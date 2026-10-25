#!/usr/bin/env python3
"""Add a pinned Pillow processor overlay without installing into the baseline."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ('torch', 'transformers', 'numpy', 'safetensors')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query(python, code):
    return json.loads(subprocess.check_output([str(python), '-c', code], text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Use a fresh environment report path')
    baseline = ROOT / '.venv/bin/python'
    overlay = ROOT / '.venv-vision'
    locked = ROOT / 'requirements-vision-overlay.lock'
    if not locked.is_file():
        parser.error('Compile the pinned overlay lock first; never alter baseline uv.lock')
    pin_paths = [ROOT / 'uv.lock', ROOT / 'requirements-neural.txt']
    pins_before = {str(path.relative_to(ROOT)): sha(path) for path in pin_paths}
    baseline_query = 'import json,importlib.metadata as m; print(json.dumps({n:m.version(n) for n in '+repr(PACKAGES)+'}))'
    baseline_before = query(baseline, baseline_query)
    if not overlay.exists():
        subprocess.run(['uv', 'venv', '--python', str(baseline), str(overlay)], check=True)
    elif not (overlay / 'pyvenv.cfg').is_file():
        parser.error('Refusing to modify a directory that is not the dedicated overlay venv')
    subprocess.run(['uv', 'pip', 'install', '--python', str(overlay / 'bin/python'),
                    '--require-hashes', '--only-binary=:all:', '--no-deps', '-r', str(locked)], check=True)
    site = query(overlay / 'bin/python', 'import json,sysconfig; print(json.dumps(sysconfig.get_path("purelib")))')
    baseline_site = query(baseline, 'import json,sysconfig; print(json.dumps(sysconfig.get_path("purelib")))')
    pth = Path(site) / 'replica_baseline.pth'
    content = str(Path(baseline_site).resolve()) + '\n'
    if pth.exists() and pth.read_text() != content:
        raise RuntimeError('Existing baseline overlay binding differs; refusing replacement')
    if not pth.exists():
        pth.write_text(content)
    environment = query(overlay / 'bin/python', '''import json,importlib.metadata as m,torch,transformers,PIL
from transformers import SiglipImageProcessorPil
print(json.dumps({'python':__import__('sys').version,'packages':{n:m.version(n) for n in ('torch','transformers','numpy','safetensors','pillow')},'torch_path':torch.__file__,'pillow_path':PIL.__file__,'processor':SiglipImageProcessorPil.__module__+'.'+SiglipImageProcessorPil.__name__}))''')
    pins_after = {str(path.relative_to(ROOT)): sha(path) for path in pin_paths}
    if pins_after != pins_before or query(baseline, baseline_query) != baseline_before:
        raise RuntimeError('Baseline source/dependency state changed')
    if any(environment['packages'][name] != baseline_before[name] for name in PACKAGES):
        raise RuntimeError('Overlay changed a baseline neural dependency')
    if not Path(environment['pillow_path']).resolve().is_relative_to(overlay.resolve()):
        raise RuntimeError('Processor dependency is not isolated in the overlay')
    result = {'format': 'replica.vision_environment.v1', 'baseline_preserved': True,
              'baseline_pins': pins_before, 'overlay': str(overlay.relative_to(ROOT)),
              'overlay_lock_sha256': sha(locked), 'baseline_site_packages': baseline_site,
              'environment': environment, 'native_computation': False,
              'scope': 'Official PIL processor import/preprocessing dependency only; no torch/Transformers replacement or native qualification'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
