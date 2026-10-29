"""Native causal-LM readout helpers with no duplicate final normalization.

The ordinary final-layer path calls the causal-LM itself, exactly as the base
model does. Intermediate-layer readouts apply the model's final norm once before
its output projection. No hidden-state mutation, cache, or persistent write is
performed here.
"""
from __future__ import annotations

from typing import Any


def _logits(outputs: Any) -> Any:
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs[0]


def _output_projection(base: Any, hidden: Any) -> Any:
    head = getattr(base, "lm_head", None)
    if callable(head):
        return head(hidden)
    getter = getattr(base, "get_output_embeddings", None)
    if callable(getter):
        projection = getter()
        if callable(projection):
            return projection(hidden)
    # Spark2_5 ties its output projection to the input embedding.
    embedding = getattr(getattr(base, "model", None), "embedding", None)
    if embedding is not None:
        import torch.nn.functional as F
        return F.linear(hidden, embedding.weight)
    raise ValueError("native model has no supported output projection")


def native_layer_logits(base: Any, input_ids: Any, layer: int,
                        attention_mask: Any | None = None) -> Any:
    """Return last-position logits for one native layer.

    For the last decoder layer, use ``base(...)`` directly so the result is
    exactly the ordinary native model path, including its final normalization
    and tied/output-head behavior. Intermediate layers use ``base.model`` and
    apply ``model.norm`` exactly once.
    """
    model = getattr(base, "model", None)
    layers = getattr(model, "layers", None)
    if model is None or layers is None:
        raise ValueError("native base must expose model.layers")
    if not isinstance(layer, int) or isinstance(layer, bool) \
            or not 0 <= layer < len(layers):
        raise ValueError(f"readout layer outside native decoder: {layer}")
    common = {"input_ids": input_ids, "use_cache": False}
    if attention_mask is not None:
        common["attention_mask"] = attention_mask
    if layer == len(layers) - 1:
        # Keep the public last-position contract while retaining the model's
        # ordinary final norm, output projection and tied-head behavior.
        return _logits(base(**common))[:, -1:, :]
    outputs = model(**common, output_hidden_states=True)
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise ValueError("native model did not return hidden states")
    hidden = hidden_states[layer + 1][:, -1:, :]
    norm = getattr(model, "norm", None)
    if not callable(norm):
        raise ValueError("native model has no final norm")
    return _output_projection(base, norm(hidden))


__all__ = ["native_layer_logits"]
