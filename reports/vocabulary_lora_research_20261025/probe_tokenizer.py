"""Small text-only probe of the pinned Spark tokenizer; never loads a model.

Usage: .venv/bin/python reports/vocabulary_lora_research_20261025/probe_tokenizer.py /path/to/tokenizer.json
The tokenizer is public metadata; this script performs no network access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from importlib.metadata import version

from tokenizers import AddedToken, Tokenizer

EXPECTED_SHA256 = "710cce15cf3565674c499f9413997c6e8101f2bdd96245cff8f0311fb501248c"
SOURCE_URL = "https://huggingface.co/XHToken/Spark-X2.6-4B/resolve/66e3ae7bee54e0a523d692cb8ef841f35409807b/tokenizer.json"
SAMPLES = [
    "A blue cup is beside the red cup.",
    "The object is called a veltraxon.",
    "The object is called a veltraxon. The veltraxon moved behind the box.",
    "MemT-1.0 and DynamicEngram", "q_k_v_proj", "000129004287",
    "Héloïse", "naïve café", "红色杯子在蓝色杯子后面。",
    "A red car turns left, then stops.", "xqzv_7281_R",
]
BOUNDARY_CASES = [SAMPLES[2], "veltraxon", " veltraxon", " veltraxonic", "superveltraxon"]


def encode(t: Tokenizer, text: str) -> dict:
    e = t.encode(text, add_special_tokens=False)
    return {"text": text, "utf8_bytes": len(text.encode()), "token_count": len(e.ids),
            "ids": e.ids, "pieces": e.tokens,
            "roundtrip": t.decode(e.ids, skip_special_tokens=False) == text}


def probe(path: Path) -> dict:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("Input must be the pinned unmodified public Spark tokenizer")
    tokenizer = Tokenizer.from_str(raw.decode())
    data = json.loads(raw)
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    trials = []
    for spelling, single_word in ((" veltraxon", False), (" veltraxon", True), ("veltraxon", True)):
        clone = Tokenizer.from_str(raw.decode())
        added = clone.add_tokens([AddedToken(spelling, single_word=single_word,
                                            normalized=False, special=False)])
        trials.append({"spelling": spelling, "single_word": single_word,
                       "added_rows": added, "new_id": clone.token_to_id(spelling),
                       "old_id_mapping_preserved": all(clone.token_to_id(s) == i for s, i in vocab.items()),
                       "cases": [encode(clone, s) for s in BOUNDARY_CASES]})
    return {
        "kind": "illustrative_tokenizer_only_probe_not_model_quality_or_speed_benchmark",
        "reviewed_on": "2026-10-25", "tokenizers_version": version("tokenizers"),
        "source_url": SOURCE_URL, "source_sha256": digest, "source_bytes": len(raw),
        "vocabulary": {"model_entries": len(data["model"]["vocab"]),
                       "all_entries": len(vocab), "max_id": max(vocab.values()),
                       "ids_contiguous": set(vocab.values()) == set(range(131072)),
                       "special_added_entries": sum(x["special"] for x in data["added_tokens"]),
                       "reserved_named_entries": sum("unused" in s for s in vocab)},
        "normalizer": data["normalizer"], "pre_tokenizer": data["pre_tokenizer"],
        "baseline_examples": [encode(tokenizer, s) for s in SAMPLES],
        "temporary_in_memory_addition_trials": trials,
        "limitations": ["Eleven authored examples, selected for illustration; not a representative corpus",
                        "No training, embeddings, model inference, timing or quality measurement",
                        "No production tokenizer or repository source tokenizer was modified",
                        "Preserving old IDs does not preserve all segmentation or generation probabilities"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_json", type=Path)
    args = parser.parse_args()
    result = probe(args.tokenizer_json)
    out = Path(__file__).with_name("tokenizer_probe.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(out), "vocabulary": result["vocabulary"],
                      "all_baseline_roundtrips": all(x["roundtrip"] for x in result["baseline_examples"])}))
