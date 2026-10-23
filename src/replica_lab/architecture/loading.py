"""One offline native loader shared by E1/E2 runtime and qualification."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from .contracts import ComponentManifest, PortSpec, fingerprint
from .minicpm5 import MiniCPM5Adapter, native_abi_digest


@contextmanager
def exclusive_mps_worker(root):
    path = Path(root) / 'runs/.D2_mps_worker.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def load_local_mps(model_path, lock_path, *, profiler=None, core=None,
                   fast_memory=None):
    """Never acquire weights; check exact sealed bytes before native loading."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..model_adapter import validate_model_lock
    if os.environ.get('REPLICA_CAPACITY_GUARD') != '1':
        raise RuntimeError('Launch through scripts/run_capacity_preflight.py --entry-point')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        raise RuntimeError('Set PYTORCH_ENABLE_MPS_FALLBACK=0 before importing torch')
    if not torch.backends.mps.is_available():
        raise RuntimeError('M2 Ultra MPS GPU required; no CPU model fallback')
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        if os.environ.get(name) != '1':
            raise RuntimeError(f'{name}=1 is required')
    lock = json.loads(Path(lock_path).read_text())
    validate_model_lock(model_path, lock)
    if lock['revision'] != 'f97400052a43d642bbc6e9975e2397e3ae6a6b52':
        raise ValueError('Use the declared original MiniCPM5 source revision')
    tokenizer_files = {k:v for k,v in lock['snapshot_files_sha256'].items()
                       if k.startswith(('tokenizer', 'special_tokens', 'chat_template'))}
    manifest = ComponentManifest(lock['revision'], fingerprint(lock['snapshot_files_sha256']),
        fingerprint(tokenizer_files), lock['config_sha256'], native_abi_digest(), PortSpec(),
        core=core, fast_memory=fast_memory)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError('Source span provenance requires native fast tokenizer offsets')
    base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation='eager', use_safetensors=True)
    inventory = list(base.named_parameters())
    if len(inventory) != 381 or sum(p.numel() for _,p in inventory) != 2516756480:
        raise ValueError('Unexpected native parameter inventory')
    adapter = MiniCPM5Adapter(base, manifest).to('mps').eval()
    adapter.seal_original_identity()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if profiler:profiler.mark('model_loaded', parameters=adapter.parameter_audit())
    return adapter, tokenizer, lock


def encode_prompt(tokenizer, text, device='mps'):
    import torch
    rendered = tokenizer.apply_chat_template([{'role':'user','content':text}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(rendered, add_special_tokens=False)['input_ids']
    return torch.tensor([ids], dtype=torch.long, device=device)
