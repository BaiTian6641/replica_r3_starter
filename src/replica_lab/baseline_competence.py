"""Pre-fit frozen-base competence. Calibration data never opens learning labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch

FORMAT = "replica.spark.baseline_competence.v1"


class CompetenceError(ValueError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def score(expected, answer):
    # Intentionally strict: extra explanation or an answer on an unknown case
    # is a failure. This scorer and the pass criteria are pinned before decode.
    return " ".join(str(expected).lower().split()) == " ".join(str(answer).lower().split())


def load_spec(protocol):
    binding = protocol.config.get("competence")
    if not binding:
        raise CompetenceError("missing separately declared pre-fit competence stage")
    path = protocol.root / binding["path"]
    if not path.is_file() or sha(path) != binding["sha256"]:
        raise CompetenceError("competence protocol hash differs")
    spec = json.loads(path.read_text())
    data = protocol.root / spec["data"]["path"]
    if sha(data) != spec["data"]["sha256"] or sha(__file__) != spec["scorer_sha256"]:
        raise CompetenceError("competence data/scorer hash differs")
    rows = json.loads(data.read_text())["cases"]
    if (len(rows) != spec["cases"] or len({r["case_id"] for r in rows}) != len(rows)
            or any(r["split"] != "pre_fit_calibration" for r in rows)
            or spec["pass_criteria"] != {"correct_min": len(rows), "false_recall_max": 0,
                                         "complete_cases": len(rows)}
            or spec["weights"] != "frozen_original_no_extensions"
            or spec["thinking_mode"] != "none"):
        raise CompetenceError("invalid competence data or predeclared criteria")
    return spec, rows


def binding_for(protocol, spec, *, context_tokens, max_new_tokens):
    return {"protocol_sha256": protocol.sha256,
            "competence_sha256": protocol.config["competence"]["sha256"],
            "data_sha256": spec["data"]["sha256"], "scorer_sha256": spec["scorer_sha256"],
            "template": spec["prompt_template"], "thinking_mode": spec["thinking_mode"],
            "tokenizer": dict(protocol.config["runtime"]["tokenizer"]),
            "source": dict(protocol.source_identity),
            "context_tokens": context_tokens, "max_new_tokens": max_new_tokens,
            "decode": spec["decode"], "criteria": spec["pass_criteria"]}


def validate_receipt(protocol, receipt, *, context_tokens=None, max_new_tokens=None):
    spec, cases = load_spec(protocol)
    binding = receipt.get("binding", {})
    expected = binding_for(protocol, spec,
        context_tokens=context_tokens or binding.get("context_tokens"),
        max_new_tokens=max_new_tokens or binding.get("max_new_tokens"))
    rows = receipt.get("rows", [])
    if (type(binding.get("context_tokens")) is not int
            or not 0 < binding["context_tokens"] <= spec["context_tokens_max"]
            or type(binding.get("max_new_tokens")) is not int
            or not 0 < binding["max_new_tokens"] <= spec["max_new_tokens"]
            or receipt.get("format") != FORMAT or binding != expected
            or len(rows) != len(cases)
            or [r.get("case_id") for r in rows] != [r["case_id"] for r in cases]
            or receipt.get("frozen_base_unchanged") is not True
            or receipt.get("optimizer_steps") != 0):
        raise CompetenceError("missing/incomplete baseline evidence or binding mismatch")
    correct = 0
    false_recall = 0
    for row, case in zip(rows, cases):
        prompt = spec["prompt_template"].format(context=case["context"], question=case["question"])
        if (row.get("complete") is not True or row.get("prompt_sha256") != canonical_sha(prompt)
                or row.get("truncated") is not False
                or len(row.get("token_ids", [])) != row.get("generated_tokens")
                or len(row.get("rendered_sha256", "")) != 64
                or not 0 < row.get("prompt_tokens", 0) <= binding["context_tokens"]
                or not 0 < row.get("generated_tokens", 0) <= binding["max_new_tokens"]):
            raise CompetenceError("incomplete baseline trajectory")
        ok = score(case["expected"], row.get("decoded", ""))
        correct += int(ok)
        false_recall += int(case["unanswerable"] and not ok)
    passed = correct >= spec["pass_criteria"]["correct_min"] and false_recall == 0
    if receipt.get("correct") != correct or receipt.get("false_recall") != false_recall:
        raise CompetenceError("baseline scores differ from raw generated answers")
    if receipt.get("passed") is not passed or not passed:
        raise CompetenceError(f"baseline competence failed: {correct}/{len(cases)}; learning stopped")
    return receipt


def run_competence(protocol, backend, output, ledger, *, require_pass=True):
    spec, cases = load_spec(protocol)
    context = ledger.context_tokens
    budget = min(ledger.max_new_tokens, spec["max_new_tokens"])
    if context > spec["context_tokens_max"] or budget <= 0:
        raise CompetenceError("undeclared competence context/generation budget")
    if any(p.requires_grad for p in backend.adapter.base.parameters()):
        raise CompetenceError("competence requires frozen original weights")
    if (backend.chat_mode or {}).get("enable_thinking") is True:
        raise CompetenceError("competence thinking mode differs")
    start = time.monotonic()
    trajectories = []
    backend.adapter.eval()
    with torch.no_grad():
        for case in cases:
            prompt = spec["prompt_template"].format(context=case["context"], question=case["question"])
            rendered = backend.render(prompt)
            tokens = backend.encode_text(rendered)
            if len(tokens) > context:
                raise CompetenceError("full calibration context does not fit; truncation refused")
            ids = torch.tensor([tokens], device=backend.device)
            generated = []
            ledger.charge("competence_cases")
            ledger.charge("generation_requests")
            for _ in range(budget):
                logits = backend.adapter.forward_text(ids, torch.ones_like(ids),
                    use_projector=False, use_ports=False, use_spine=False)
                value = int(logits[0, -1].argmax())
                generated.append(value)
                ledger.charge("competence_forwards")
                ledger.charge("autoregressive_forwards")
                ledger.charge("forward_calls")
                ledger.charge("generated_tokens")
                ledger.charge("processed_text_positions", int(ids.shape[1]))
                if value in backend.stop_token_ids:
                    break
                ids = torch.cat((ids, ids.new_tensor([[value]])), dim=1)
            decoded = backend.decode([t for t in generated if t not in backend.stop_token_ids]).strip()
            trajectories.append({"case_id": case["case_id"], "complete": True,
                "prompt_sha256": canonical_sha(prompt), "rendered_sha256": canonical_sha(rendered),
                "prompt_tokens": len(tokens), "token_ids": generated, "decoded": decoded,
                "generated_tokens": len(generated), "truncated": False})
            ledger.charge("verification_calls")
    correct = sum(score(c["expected"], r["decoded"]) for c, r in zip(cases, trajectories))
    false = sum(c["unanswerable"] and not score(c["expected"], r["decoded"])
                for c, r in zip(cases, trajectories))
    receipt = {"format": FORMAT, "binding": binding_for(protocol, spec,
                   context_tokens=context, max_new_tokens=budget),
        "rows": trajectories, "correct": correct, "false_recall": false,
        "passed": correct == len(cases) and false == 0,
        "frozen_base_unchanged": backend.adapter.base_unchanged() is True,
        "optimizer_steps": 0, "wall_seconds": time.monotonic() - start,
        "native_execution": bool(backend.native_admissible), "development_labels_opened": False,
        "sealed_final_opened": False, "engineering_diagnostics_retained": True}
    path = Path(output) / "competence/receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if require_pass:
        validate_receipt(protocol, receipt, context_tokens=context, max_new_tokens=budget)
    return receipt
