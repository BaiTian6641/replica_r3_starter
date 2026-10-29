"""Small CPU library checks; no Spark model is imported or instantiated."""
import json
from pathlib import Path
import torch
import transformers
from transformers.cache_utils import DynamicSlidingWindowLayer
from transformers.utils.chat_template_utils import _compile_jinja_template

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def describe(cache, returned):
    return {"retained_visible_tokens": cache.keys.shape[-2],
            "returned_attention_tokens": returned.shape[-2],
            "visible_key_bytes": cache.keys.numel()*cache.keys.element_size(),
            "key_backing_storage_bytes": cache.keys.untyped_storage().nbytes(),
            "cumulative_tokens": cache.get_seq_length()}


def main():
    torch.set_num_threads(1)
    cache = DynamicSlidingWindowLayer(sliding_window=512)
    # One tiny head/dimension keeps the probe under a kilobyte-scale payload.
    x = torch.zeros((1, 1, 2048, 1), dtype=torch.bfloat16, device="cpu")
    returned, _ = cache.update(x,x)
    prefill = describe(cache,returned)
    y = torch.ones((1, 1, 1, 1), dtype=torch.bfloat16, device="cpu")
    returned, _ = cache.update(y,y)
    decode = describe(cache,returned)
    assert prefill["retained_visible_tokens"] == 511
    assert prefill["key_backing_storage_bytes"] == 4096
    assert decode["key_backing_storage_bytes"] == 1024
    assert decode["cumulative_tokens"] == 2049
    template = _compile_jinja_template((ROOT/"checkpoints/sparkx2_5_bf16_source/chat_template.jinja").read_text())
    outputs = {}
    for name, kwargs in {"default": {}, "none": {"enable_thinking": False},
                         "low": {"enable_thinking": True, "reasoning_effort": "low"},
                         "high": {"enable_thinking": True, "reasoning_effort": "high"}}.items():
        text = template.render(messages=[{"role":"user","content":"test"}],add_generation_prompt=True,**kwargs)
        outputs[name] = {"opens_think":text.endswith("<think>"), "closes_think":text.endswith("</think>"),
                         "high_instruction":"thinking_effort=high" in text,
                         "low_instruction":"thinking_effort=low" in text}
    assert outputs["default"] == outputs["high"]
    assert outputs["none"]["closes_think"] and outputs["low"]["low_instruction"]
    try:
        template.render(messages=[{"role":"user","content":"test"}],add_generation_prompt=True,reasoning_effort="none")
    except Exception as exc:
        invalid = {"rejected": True, "exception_type":type(exc).__name__}
    else:
        raise AssertionError("reasoning_effort=none must not silently select no-thinking")
    result = {"scope":"CPU cache/template behavior only; no model weights or native execution",
              "torch":torch.__version__, "transformers":transformers.__version__,
              "prefill":prefill, "after_one_decode_update":decode,
              "template_modes":outputs, "invalid_reasoning_effort_none":invalid}
    (OUT/"cache_template_probe.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
