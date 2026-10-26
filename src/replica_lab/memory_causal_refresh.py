"""Frozen-writer causal query refresh with explicit oracle and hybrid controls.

Policy inputs are only the current query and observations already encoded in the
local checkpoint. Verifier labels choose no refresh action. Gold-target refresh
is a separate oracle diagnostic. A query-addressed cached-observation refresh is
a hybrid retrieval/neural-memory system, not pure learned-memory retention.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch

from .memory_diagnosis import _cosine_heads
from .memory_matched import episode_states
from .memory_update_dynamics import (
    _build_state, _codebook, _freeze, _parameter_versions, _read_weights,
    _readout_row, _recompute_ranking, _sha_file, _sha_json, _state_context,
    _state_sha, _utc_now, _variant_metrics, _write_json, _write_jsonl,
)
from .neural import SGDN2Memory, SGDN2Write

PROTOCOL_FORMAT = "replica.memory_causal_refresh.protocol.v1"
RUN_FORMAT = "replica.memory_causal_refresh.run.v1"
DECISION_FORMAT = "replica.memory_causal_refresh.decision.v1"
CHECKPOINT_FORMAT = "replica.memory_causal_refresh.checkpoint.v1"
FULL = "shared_mean/full"
CAUSAL = "shared_mean/causal_query_refresh"
SHUFFLED = "shared_mean/causal_query_refresh_shuffled_values"
DISABLED = "shared_mean/disabled"
ORACLE = "shared_mean/oracle_target_refresh_last"
VARIANT_NAMES = (FULL, CAUSAL, SHUFFLED, DISABLED, ORACLE)
ARMS = ("exact_key_value", "query_key_value", "query_key_address", "context_removed")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _variants():
    return [{"name": name, "reducer": "shared_mean"} for name in VARIANT_NAMES]


def _zero_state(primitive, device):
    return {"shared": primitive.initial_state(1, device=device)}


def _history_writes(context, history, layers, *, shuffle_values=False):
    donor_ids = sorted(record["record_id"] for record in context["records"])
    donors = {record_id: donor_ids[(index + 1) % len(donor_ids)]
              for index, record_id in enumerate(donor_ids)} if shuffle_values else {}
    result = []
    for record in history:
        writes = {}
        for layer in layers:
            write = context["writes"][(record["record_id"], layer)]
            if shuffle_values:
                donor = context["writes"][(donors[record["record_id"]], layer)]
                write = replace(write, values=donor.values)
            writes[layer] = write
        result.append(writes)
    return result


def decide_refresh(context, query, layer, *, cosine_floor, margin_floor):
    """Choose a refresh using only current-query and observed-key information."""
    query_vector = context["query_vectors"][(query["query_id"], layer)]
    scores = {
        record_id: _cosine_heads(
            query_vector[0, 0].detach().float().cpu(),
            context["writes"][(record_id, layer)].keys[0, 0].detach().float().cpu())
        for record_id in context["live_ids"]
    }
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if not ordered:
        return {"selected_record_id": None, "accepted": False, "scores": scores,
                "top_score": None, "second_score": None, "reason": "no_live_observation"}
    top_id, top = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else None
    accepted = bool(top >= cosine_floor and top - (second or 0.0) >= margin_floor)
    return {"selected_record_id": top_id if accepted else None,
            "accepted": accepted, "scores": scores, "top_score": top,
            "second_score": second,
            "reason": "query_address_accepted" if accepted else "query_address_below_threshold"}


def _decision_row(variant, context, query, layer, decision, history, *, oracle=False):
    return {
        "format": DECISION_FORMAT,
        "variant": variant,
        "policy_class": "gold_target_oracle" if oracle else "causal_current_query",
        "episode_id": context["episode"]["episode_id"],
        "state": context["state"]["name"],
        "case_id": query["query_id"],
        "layer": layer,
        "query_text_sha256": _sha_json(query["text"]),
        "selected_record_id": decision["selected_record_id"],
        "accepted": decision["accepted"],
        "reason": decision["reason"],
        "scores": decision["scores"],
        "written_record_ids": [record["record_id"] for record in history],
        "used_gold_target": oracle,
        "used_future_queries": False,
        "reread_source": False,
        "reencoded_old_observations": False,
    }


def _evaluate_variant(*, variant, context, query, labels, layers, primitive,
                      device, zero_weights, cosine_floor, margin_floor):
    query_id = query["query_id"]
    target_info = labels["targets"][query_id]
    target_id = target_info["target_entry_id"]
    kind = target_info["kind"]
    records = list(context["records"])
    decisions, readouts = [], []
    for layer in layers:
        # decide_refresh receives no labels or target identity; target_info is
        # used only by the independent evaluator below.
        if variant == ORACLE:
            selected = target_id if kind == "fact" else None
            decision = {"selected_record_id": selected, "accepted": selected is not None,
                        "scores": {}, "reason": "gold_target_oracle"}
        elif variant in (CAUSAL, SHUFFLED):
            decision = decide_refresh(context, query, layer,
                                      cosine_floor=cosine_floor,
                                      margin_floor=margin_floor)
        else:
            decision = {"selected_record_id": None, "accepted": False,
                        "scores": {}, "reason": "no_refresh"}
        selected = decision["selected_record_id"]
        if variant in (CAUSAL, SHUFFLED, ORACLE) and selected is not None:
            target = next(record for record in records if record["record_id"] == selected)
            history = records + [target]
        elif variant == DISABLED or (variant in (CAUSAL, SHUFFLED) and selected is None):
            history = []
        else:
            history = records
        states = (_zero_state(primitive, device) if not history else
                  _build_state(
                      primitive, layers, _history_writes(
                          context, history, layers, shuffle_values=variant == SHUFFLED),
                      reducer="shared_mean", gate_control=None, device=device))
        decisions.append(_decision_row(
            variant, context, query, layer, decision, history,
            oracle=variant == ORACLE))
        weights = states["shared"].weights
        state_sha = _state_sha(weights)
        final_write = history[-1]["record_id"] if history else None
        vector = context["query_vectors"][(query_id, layer)]
        if kind == "fact" and target_id is not None:
            exact = context["writes"][(target_id, layer)].keys[0, 0].detach().float().cpu()
            readouts.append(_readout_row(
                variant={"name": variant, "reducer": "shared_mean"},
                arm="exact_key_value", context=context, query=query, layer=layer,
                readout=_read_weights(weights, exact.view(1, 1, *exact.shape)),
                readout_space="value", final_write_id=final_write,
                state_sha256=state_sha))
        readouts.append(_readout_row(
            variant={"name": variant, "reducer": "shared_mean"},
            arm="query_key_value", context=context, query=query, layer=layer,
            readout=_read_weights(weights, vector), readout_space="value",
            final_write_id=final_write, state_sha256=state_sha))
        readouts.append(_readout_row(
            variant={"name": variant, "reducer": "shared_mean"},
            arm="query_key_address", context=context, query=query, layer=layer,
            readout=vector[0, 0].detach().float().cpu(), readout_space="key",
            final_write_id=final_write, state_sha256=state_sha))
        readouts.append(_readout_row(
            variant={"name": variant, "reducer": "shared_mean"},
            arm="context_removed", context=context, query=query, layer=layer,
            readout=_read_weights(zero_weights, vector), readout_space="value",
            final_write_id=final_write, state_sha256=_state_sha(zero_weights)))
    return decisions, readouts


def _pack_write(write):
    return {"keys": write.keys, "values": write.values, "erase": write.erase,
            "write_gate": write.write_gate, "decay": write.decay,
            "chunk_index": write.chunk_index, "mask": write.mask}


def _pack_context(context):
    packed = dict(context)
    packed["writes"] = [
        {"record_id": record_id, "layer": layer, "write": _pack_write(write)}
        for (record_id, layer), write in context["writes"].items()]
    packed["query_vectors"] = [
        {"query_id": query_id, "layer": layer, "vector": vector}
        for (query_id, layer), vector in context["query_vectors"].items()]
    packed["entry_vectors"] = [
        {"record_id": record_id, "layer": layer, "vector": vector}
        for (record_id, layer), vector in context["entry_vectors"].items()]
    return packed


def _unpack_context(context):
    unpacked = dict(context)
    unpacked["writes"] = {
        (item["record_id"], item["layer"]): SGDN2Write(**item["write"])
        for item in context["writes"]}
    unpacked["query_vectors"] = {
        (item["query_id"], item["layer"]): item["vector"]
        for item in context["query_vectors"]}
    unpacked["entry_vectors"] = {
        (item["record_id"], item["layer"]): item["vector"]
        for item in context["entry_vectors"]}
    return unpacked


def save_policy_checkpoint(path, contexts, *, protocol_sha256, writer_sha256):
    destination = Path(path)
    if destination.exists():
        raise FileExistsError("Use a new policy checkpoint")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": CHECKPOINT_FORMAT, "protocol_sha256": protocol_sha256,
               "writer_weights_sha256": writer_sha256,
               "contexts": [_pack_context(context) for context in contexts]}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.",
                                             suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_policy_checkpoint(path, *, protocol_sha256, writer_sha256):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    _require(isinstance(payload, dict) and payload.get("format") == CHECKPOINT_FORMAT,
             "Invalid causal policy checkpoint")
    _require(payload.get("protocol_sha256") == protocol_sha256,
             "Causal policy checkpoint protocol differs")
    _require(payload.get("writer_weights_sha256") == writer_sha256,
             "Causal policy checkpoint writer differs")
    contexts = payload.get("contexts")
    _require(isinstance(contexts, list) and contexts,
             "Causal policy checkpoint has no observed contexts")
    return [_unpack_context(context) for context in contexts]


def run_causal_refresh(model, tokenizer, protocol, *, output_dir, episodes,
                       labels_by_episode, writer_extension_receipt,
                       device="cpu", resume_checkpoint=None):
    started = _utc_now()
    started_monotonic = time.monotonic()
    _require(protocol.get("format") == PROTOCOL_FORMAT,
             "Causal refresh requires its versioned protocol")
    layers = tuple(protocol["layers"])
    thresholds = protocol["thresholds"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("decisions.jsonl", "readouts.jsonl", "rankings.jsonl",
                 "codebooks.json", "report.json"):
        _require(not (output_dir / name).exists(), f"Fresh output required: {name}")
    model.eval()
    _freeze(model)
    versions = _parameter_versions(model)
    primitive = SGDN2Memory(model.manifest.fast_memory.heads,
                            model.manifest.fast_memory.key_size,
                            model.manifest.fast_memory.value_size)
    protocol_sha = protocol["protocol_sha256"]
    writer_sha = writer_extension_receipt["weights_sha256"]
    if resume_checkpoint is None:
        contexts = []
        for episode in episodes:
            for state in episode_states(episode):
                contexts.append(_state_context(model, tokenizer, episode, state, layers))
        checkpoint_path = output_dir / "policy_checkpoint.pt"
        save_policy_checkpoint(checkpoint_path, contexts,
                               protocol_sha256=protocol_sha, writer_sha256=writer_sha)
        encoded_contexts = len(contexts)
        resumed = False
    else:
        contexts = load_policy_checkpoint(
            resume_checkpoint, protocol_sha256=protocol_sha, writer_sha256=writer_sha)
        checkpoint_path = Path(resume_checkpoint)
        encoded_contexts = 0
        resumed = True
    expected = {(episode["episode_id"], state["name"])
                for episode in episodes for state in episode_states(episode)}
    actual = {(context["episode"]["episode_id"], context["state"]["name"])
              for context in contexts}
    _require(actual == expected, "Checkpoint episode/state inventory differs")

    decisions, readouts, codebooks = [], [], {}
    with torch.no_grad():
        for context in contexts:
            episode = context["episode"]
            state = context["state"]
            labels = labels_by_episode[episode["episode_id"]][state["name"]]
            codebooks[f"{episode['episode_id']}/{state['name']}"] = _codebook(context, layers)
            zero = primitive.initial_state(1, device=device).weights
            for query in episode["queries"]:
                for variant in VARIANT_NAMES:
                    new_decisions, new_readouts = _evaluate_variant(
                        variant=variant, context=context, query=query,
                        labels=labels, layers=layers, primitive=primitive,
                        device=device, zero_weights=zero,
                        cosine_floor=thresholds["address_cosine_floor"],
                        margin_floor=thresholds["address_margin_floor"])
                    decisions.extend(new_decisions)
                    readouts.extend(new_readouts)
    _require(_parameter_versions(model) == versions,
             "Frozen writer parameters changed")
    for row in decisions + readouts:
        _require(not any(key in row for key in ("expected", "expected_answer", "gold_answer")),
                 "Raw policy/readout row contains an evaluator answer")
    _write_jsonl(output_dir / "decisions.jsonl", decisions)
    _write_jsonl(output_dir / "readouts.jsonl", readouts)
    _write_json(output_dir / "codebooks.json", {"codebooks": codebooks})
    rankings = [_recompute_ranking(row, codebooks[row["codebook_key"]], labels_by_episode[
        row["episode_id"]][row["state"]], thresholds) for row in readouts]
    _write_jsonl(output_dir / "rankings.jsonl", rankings)
    metrics = _variant_metrics(rankings, _variants())
    causal = metrics[CAUSAL]["query_key_value"]
    oracle = metrics[ORACLE]["exact_key_value"]
    causal_fact = causal["fact"]
    oracle_fact = oracle["fact"]
    _require(causal_fact.get("n") and causal["correction"].get("n")
             and causal["unaffected"].get("n") and causal["unknown"].get("n")
             and oracle_fact.get("n"),
             "Causal evaluation lacks fact/correction/unaffected/unknown coverage")
    gates = {
        "causal_top1": causal_fact["top1_rate"] >= thresholds["causal_top1_floor"],
        "correction_top1": causal["correction"]["top1_rate"] >= thresholds["correction_top1_floor"],
        "unaffected_top1": causal["unaffected"]["top1_rate"] >= thresholds["unaffected_top1_floor"],
        "unknown_false_recall": causal["unknown"]["false_recall_rate"] <= thresholds["unknown_false_recall_max"],
        "oracle_bound": oracle_fact["top1_rate"] >= thresholds["oracle_exact_top1_floor"],
        "frozen_writer": _parameter_versions(model) == versions,
        "restart_without_reencoding": not resumed or encoded_contexts == 0,
    }
    learned_pass = all(gates[name] for name in (
        "causal_top1", "correction_top1", "unaffected_top1", "unknown_false_recall"))
    mechanical_pass = gates["frozen_writer"] and gates["restart_without_reencoding"]
    decision = ("accepted" if learned_pass and mechanical_pass and gates["oracle_bound"]
                else "rejected" if gates["oracle_bound"] else "inconclusive")
    report = {
        "format": RUN_FORMAT, "status": "completed", "decision": decision,
        "started_at_utc": started,
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "protocol_sha256": protocol_sha,
        "writer_extension_receipt": writer_extension_receipt,
        "optimizer_updates": 0,
        "native": device == "mps", "device": device,
        "resumed": resumed,
        "contexts_encoded": encoded_contexts,
        "old_observations_reencoded": False,
        "checkpoint": {"path": str(checkpoint_path),
                       "sha256": _sha_file(checkpoint_path),
                       "new_checkpoint_saved": not resumed},
        "metrics": metrics,
        "gates": gates,
        "learned_causal_refresh_pass": learned_pass,
        "mechanical_pass": mechanical_pass,
        "classification": {
            "full_history": "frozen learned-memory baseline",
            "causal_refresh": "hybrid query-addressed cached-observation retrieval plus neural refresh",
            "oracle_refresh": "gold-target diagnostic upper bound; not deployable policy",
            "exact_source_or_full_context": "hybrid exact retrieval/context controls, not learned memory",
        },
        "policy_inputs": {
            "current_query": True, "cached_observed_writes": True,
            "gold_targets_for_causal_policy": False, "future_queries": False,
            "undeclared_source_rereading": False,
        },
        "raw": {
            "decisions_sha256": _sha_file(output_dir / "decisions.jsonl"),
            "readouts_sha256": _sha_file(output_dir / "readouts.jsonl"),
            "rankings_sha256": _sha_file(output_dir / "rankings.jsonl"),
            "codebooks_sha256": _sha_file(output_dir / "codebooks.json"),
        },
    }
    _write_json(output_dir / "report.json", report)
    return report
