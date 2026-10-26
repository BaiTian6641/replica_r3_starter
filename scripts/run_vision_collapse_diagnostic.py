#!/usr/bin/env python3
"""Training-only MiniCPM5 vision-collapse diagnostic (single model load).

Two stages that differ only in the six vision-projector tensors are measured on
the same sealed ``prepared04`` training rows, prompt/response builders, control
plan, teacher-forced logits and greedy generations:

* ``fresh_init_seed29``: seed-29 projector initialization from the training
  recipe, with the pinned train04 frozen sidecars copied in first;
* ``train04_extension``: the pinned train04 extension reloaded unchanged.

The run performs zero optimizer steps, zero backward passes, zero gradient
computations and zero checkpoint/extension writes.  It reads no evaluation,
dev, held-out or semantic-audit payload, downloads nothing and disables CPU
fallback.  One exclusive MPS worker, the existing capacity guard and the
existing profiler are mandatory.

Artifacts are first-order evidence only.  Nothing here promotes learned,
perception or semantic quality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replica_lab.vision_collapse_diagnostic import (  # noqa: E402
    CELL_FORMAT, CONFIG_FORMAT, CONTROL_NAMES, DIAGNOSTIC_FORMAT,
    EXPECTED_STOP_TOKENS, REPORT_FORMAT, ROW_FORMAT, DiagnosticError,
    claims_block, comparison_cells, control_plan, control_summary,
    eos_pair_analysis, feature_ceiling, feature_summary,
    first_token_analysis, first_token_ceiling_summary, fit_contract_row_report,
    fit_coverage, generation_evidence, pair_groups, pair_layout, position_nll,
    require, require_fit_coverage, require_stop_conditions, resolve_stop_tokens,
    stop_conditions_report,
)
from replica_lab.vision_curriculum import (  # noqa: E402
    VisionCurriculumError, verify_implementation_pins)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(row, allow_nan=False) + "\n"
                                  for row in rows))


def _read_json(path):
    return json.loads(Path(path).read_text())


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _check_wall(started, limit, phase):
    elapsed = time.monotonic() - started
    if elapsed > float(limit):
        raise DiagnosticError(
            f"{phase} exceeded the declared {limit}s wall limit "
            f"({elapsed:.1f}s elapsed)")


# ---------------------------------------------------------------------------
# Config, prepared/source/extension pins and training-row validation
# ---------------------------------------------------------------------------


_CONFIG_KEYS = {"format", "version", "name", "declared_at_utc", "purpose",
                "prepared", "source", "extension", "stages", "stop_tokens",
                "target_eos_token", "controls", "prompt_contract", "limits",
                "policy", "implementation_pins", "output", "report_root",
                "claims", "limitations"}


def load_config(path):
    config = _read_json(path)
    require(isinstance(config, dict), "The diagnostic config must be an object")
    require(set(config) == _CONFIG_KEYS,
            f"Diagnostic config fields differ: {sorted(set(config) ^ _CONFIG_KEYS)}")
    require(config["format"] == CONFIG_FORMAT, "Diagnostic config format differs")
    require(config["version"] == DIAGNOSTIC_FORMAT,
            "Diagnostic version differs")
    require(list(config["controls"]) == list(CONTROL_NAMES),
            "Diagnostic controls differ from the declared matched set")
    require(tuple(config["stop_tokens"]) == EXPECTED_STOP_TOKENS,
            "Diagnostic stop tokens differ from the native {1, 130073} set")
    require(config["target_eos_token"] == 1,
            "Diagnostic target EOS differs from the tokenizer </s>")
    require(config["policy"]["sealed_evaluation_access"] == 0,
            "Sealed evaluation access must be declared zero")
    require(config["limits"]["updates"] == 0
            and config["limits"]["optimizer_steps"] == 0
            and config["limits"]["gradients"] == 0
            and config["limits"]["checkpoint_writes"] == 0
            and config["limits"]["extension_writes"] == 0,
            "The diagnostic limits must forbid every write/update")
    require(config["limits"]["model_loads"] == 1,
            "The diagnostic allows exactly one model load")
    require(config["limits"]["artifact_bytes"] == 134217728,
            "The artifact ceiling differs from 128 MiB")
    require(config["limits"]["max_seconds"] == 1200,
            "The wall ceiling differs from 1200s")
    require([stage["kind"] for stage in config["stages"]]
            == ["fresh_projector", "extension"], "Diagnostic stages differ")
    require(config["policy"]["cpu_fallback"] is False
            and config["policy"]["exclusive_mps_worker"] is True
            and config["policy"]["capacity_guard_required"] is True
            and config["policy"]["profiler_required"] is True
            and config["policy"]["fresh_output"] is True,
            "The declared resource policy differs")
    require(config["claims"]["promotion"] is False
            and config["claims"]["learned_quality"] is False,
            "The diagnostic must not promote learned quality")
    return config


def verify_pins(config, root=ROOT):
    """Verify prepared, source, lock and extension bytes without loading a model."""
    root = Path(root)
    prepared = root / config["prepared"]["root"]
    require(prepared.is_dir(), "The prepared data root is missing")
    protocol_path = root / config["prepared"]["protocol"]
    manifest_path = root / config["prepared"]["manifest"]
    training_path = root / config["prepared"]["training"]
    for path, label in ((protocol_path, "prepared protocol"),
                        (manifest_path, "prepared manifest"),
                        (training_path, "training alignment")):
        require(path.is_file(), f"The {label} is missing: {path}")
    protocol_sha = _sha(protocol_path)
    require(protocol_sha == config["prepared"]["protocol_sha256"],
            "Prepared protocol bytes differ from the diagnostic pin")
    manifest_sha = _sha(manifest_path)
    require(manifest_sha == config["prepared"]["manifest_sha256"],
            "Prepared manifest bytes differ from the diagnostic pin")
    training_sha = _sha(training_path)
    require(training_sha == config["prepared"]["training_sha256"],
            "Training alignment bytes differ from the diagnostic pin")
    protocol = _read_json(protocol_path)
    manifest = _read_json(manifest_path)
    require(protocol["data"]["root"] == config["prepared"]["root"],
            "Prepared protocol data root differs")
    require(protocol["data"]["manifest"]["sha256"] == manifest_sha,
            "Prepared protocol manifest pin differs")
    require(protocol["data"]["seals"]["train_alignment_payload"] == training_sha,
            "Prepared alignment seal differs")
    counts = protocol["data"]["counts"]
    require(counts["train_examples"] == config["prepared"]["training_examples"]
            and counts["train_answers"] == config["prepared"]["answer_examples"]
            and counts["train_captions"] == config["prepared"]["caption_examples"],
            "Prepared training counts differ")
    require(manifest["counts"] == counts, "Prepared manifest counts differ")
    fit = protocol["fit_contract"]
    prompt_contract = config["prompt_contract"]
    require(fit["question_template"] == prompt_contract["question_template"]
            == protocol["evaluation"]["question_template"],
            "The question template pins differ")
    require(fit["caption_template"] == prompt_contract["caption_template"],
            "The caption template pins differ")
    require(fit["loss"] == prompt_contract["loss"]
            and fit["response_only_labels"] is True,
            "The prepared fit loss/label contract differs")
    require(int(protocol["train"]["init_seed"])
            == int(config["stages"][0]["init_seed"]),
            "The fresh init seed differs from the prepared protocol")
    limits = config["limits"]
    for name in ("max_prompt_tokens", "max_sequence_tokens",
                 "max_new_tokens", "caption_max_new_tokens"):
        require(int(protocol["limits"][name]) == int(limits[name]),
                f"The declared {name} differs from the prepared protocol")
    resource = protocol["resource_policy"]
    require(resource["device"] == config["policy"]["device"] == "mps",
            "The declared device differs")
    require(resource["cpu_fallback"] is False
            and resource["downloads"] == 0
            and resource["workers"] == 1,
            "The prepared resource policy allows fallback, downloads or workers")
    require(resource["exclusive_mps_worker"] is True,
            "The prepared resource policy lacks the exclusive MPS worker")

    lock_path = root / config["source"]["model_lock"]
    require(lock_path.is_file(), "The model lock is missing")
    lock_sha = _sha(lock_path)
    require(lock_sha == config["source"]["model_lock_sha256"],
            "Model lock bytes differ from the diagnostic pin")
    lock = _read_json(lock_path)
    require(lock["revision"] == config["source"]["model_revision"],
            "Model lock revision differs")
    require(protocol["model"]["revision"] == config["source"]["model_revision"],
            "Prepared protocol model revision differs")
    require(protocol["model"]["lock_sha256"] == lock_sha,
            "Prepared protocol model lock pin differs")
    model_path = root / config["source"]["model_path"]
    require(model_path.is_dir(), "The sealed model path is missing")
    require(_sha(model_path / "config.json") == lock["config_sha256"],
            "Local model config differs from the lock")
    require(bool(list(model_path.glob("*.safetensors"))),
            "The sealed model snapshot has no safetensors weights")
    vision_manifest = root / config["source"]["vision_source_manifest"]
    require(vision_manifest.is_file(), "The vision source manifest is missing")
    vision_manifest_sha = _sha(vision_manifest)
    require(vision_manifest_sha == config["source"]["vision_source_manifest_sha256"],
            "Vision source manifest bytes differ from the diagnostic pin")
    require(protocol["vision_source"]["manifest"]["sha256"] == vision_manifest_sha,
            "Prepared protocol vision manifest pin differs")

    extension_path = root / config["extension"]["path"]
    weights_path = extension_path / "extensions.safetensors"
    receipt_path = extension_path / "manifest.json"
    require(weights_path.is_file() and receipt_path.is_file(),
            "The pinned train04 extension is incomplete")
    receipt_sha = _sha(receipt_path)
    require(receipt_sha == config["extension"]["manifest_sha256"],
            "Extension receipt bytes differ from the diagnostic pin")
    weights_sha = _sha(weights_path)
    require(weights_sha == config["extension"]["weights_sha256"],
            "Extension weights differ from the diagnostic pin")
    require(weights_path.stat().st_size == config["extension"]["weights_bytes"],
            "Extension weight bytes differ from the diagnostic pin")
    receipt = _read_json(receipt_path)
    require(receipt["format"] == config["extension"]["format"],
            "Extension format differs")
    require(receipt["regime"] == config["extension"]["regime"],
            "Extension regime differs")
    require(receipt["base_saved"] is False,
            "The pinned extension saved base weights")
    require(receipt["weights_sha256"] == weights_sha,
            "Extension receipt weight hash differs")
    require(receipt["manifest_digest"] == config["extension"]["manifest_digest"],
            "Extension manifest digest differs")
    require(receipt["manifest"]["vision"] == protocol["vision"],
            "Extension vision pins differ from the prepared protocol")
    require(receipt["manifest"]["source_revision"]
            == config["source"]["model_revision"],
            "Extension source revision differs")

    migration_config = config["implementation_pins"]
    migrations = dict(migration_config["migrations"])
    prepared_pins = dict(protocol.get("implementation_sha256") or {})
    require(migration_config["prepared_protocol_pins_checked"] is True
            and migration_config["mode"] == "explicit_read_only_migration",
            "The diagnostic must declare explicit read-only implementation migrations")
    require(prepared_pins, "The prepared protocol lacks implementation pins")
    require(set(migrations) <= set(prepared_pins),
            "Implementation migration names an unpinned file")
    unchanged = {"implementation_sha256": {
        relative: expected for relative, expected in prepared_pins.items()
        if relative not in migrations}}
    actual = verify_implementation_pins(unchanged, root)
    for relative, migration in migrations.items():
        require(isinstance(migration, dict)
                and set(migration) == {"parent_sha256", "current_sha256", "reason"},
                f"Implementation migration is malformed: {relative}")
        expected = prepared_pins[relative]
        require(migration["parent_sha256"] == expected,
                f"Implementation migration parent differs: {relative}")
        current = migration["current_sha256"]
        require(isinstance(current, str) and len(current) == 64
                and all(character in "0123456789abcdef" for character in current)
                and current != expected,
                f"Implementation migration current pin is invalid: {relative}")
        require(isinstance(migration["reason"], str)
                and bool(migration["reason"].strip()),
                f"Implementation migration lacks a reason: {relative}")
        try:
            migrated_actual = verify_implementation_pins(
                {"implementation_sha256": {relative: current}}, root)
        except VisionCurriculumError as error:
            raise DiagnosticError(
                f"Migrated implementation bytes differ: {relative}") from error
        actual.update(migrated_actual)
    implementation = {}
    for relative in sorted(prepared_pins):
        expected = prepared_pins[relative]
        migrated = relative in migrations
        implementation[relative] = {
            "pinned_sha256": expected,
            "actual_sha256": actual[relative],
            "enforced": True,
            "matches_prepared_pin": actual[relative] == expected,
            "migrated": migrated,
            "migration": migrations.get(relative),
        }
    pins = {
        "prepared": {
            "root": config["prepared"]["root"],
            "protocol_sha256": protocol_sha,
            "manifest_sha256": manifest_sha,
            "training_sha256": training_sha,
            "training_examples": counts["train_examples"],
            "answer_examples": counts["train_answers"],
            "caption_examples": counts["train_captions"],
        },
        "source": {
            "model_path": config["source"]["model_path"],
            "model_lock_sha256": lock_sha,
            "model_revision": lock["revision"],
            "model_config_sha256": lock["config_sha256"],
            "vision_source_manifest_sha256": vision_manifest_sha,
            "vision_digest": protocol["vision_digest"],
        },
        "extension": {
            "path": config["extension"]["path"],
            "manifest_sha256": receipt_sha,
            "weights_sha256": weights_sha,
            "weights_bytes": weights_path.stat().st_size,
            "manifest_digest": receipt["manifest_digest"],
            "format": receipt["format"],
            "base_saved": receipt["base_saved"],
        },
        "implementation": {
            "enforced": True,
            "mode": migration_config["mode"],
            "reason": migration_config["reason"],
            "files": implementation,
        },
    }
    return {
        "protocol": protocol,
        "manifest": manifest,
        "pins": pins,
        "protocol_path": protocol_path,
        "training_path": training_path,
        "extension_path": extension_path,
    }


def validate_rows(rows, config, prepared_root):
    """Validate the 26 training rows and their pinned images, without labels."""
    require(isinstance(rows, list) and len(rows) == config["prepared"]["training_examples"],
            "Training row count differs from the diagnostic pin")
    seen_case: set[str] = set()
    seen_example: set[str] = set()
    kinds = {"answer": 0, "caption": 0}
    for row in rows:
        require(set(row) >= {"case_id", "example_id", "image", "image_sha256",
                             "kind", "pair_id", "question", "response",
                             "role", "world_id"},
                "A training row lacks a required field")
        case_id = str(row["case_id"])
        example_id = str(row["example_id"])
        require(case_id not in seen_case, f"Duplicate training case id: {case_id}")
        require(example_id not in seen_example,
                f"Duplicate training example id: {example_id}")
        seen_case.add(case_id)
        seen_example.add(example_id)
        require(row["kind"] in kinds, f"Unknown training kind: {row['kind']!r}")
        kinds[row["kind"]] += 1
        require(isinstance(row["response"], str) and row["response"].strip(),
                "A training response is missing")
        require(isinstance(row["question"], str) and row["question"].strip(),
                "A training question is missing")
        digest = str(row["image_sha256"])
        require(len(digest) == 64 and
                all(character in "0123456789abcdef" for character in digest),
                "A training image hash is not SHA-256")
        relative = Path(str(row["image"]))
        require(not relative.is_absolute() and ".." not in relative.parts,
                "A training image path escapes the prepared root")
        require(relative.parts and relative.parts[0] not in {"evaluation", "dev",
                                                             "heldout"},
                "A training row points outside the training payload")
        require("label" not in row and "answer" not in row,
                "Training rows must not carry evaluation labels")
        image_path = Path(prepared_root) / relative
        require(image_path.is_file(), f"Training image is missing: {relative}")
        require(_sha(image_path) == digest,
                f"Training image bytes differ from its row hash: {relative}")
    require(kinds["answer"] == config["prepared"]["answer_examples"],
            "Answer example count differs")
    require(kinds["caption"] == config["prepared"]["caption_examples"],
            "Caption example count differs")
    layout = pair_layout(rows)
    require(layout["opposite_answer_pairs"] == config["prepared"]["answer_pairs"],
            "Answer pair count differs")
    require(layout["caption_sibling_pairs"] == config["prepared"]["caption_pair_ids"],
            "Caption pair count differs")
    return layout


def load_training_rows(config, pins):
    rows = _read_jsonl(ROOT / config["prepared"]["training"])
    prepared_root = ROOT / config["prepared"]["root"]
    layout = validate_rows(rows, config, prepared_root)
    for row in rows:
        row["image_path"] = prepared_root / str(row["image"])
    return rows, layout


# ---------------------------------------------------------------------------
# Stage evaluation
# ---------------------------------------------------------------------------


class StageEvaluator:
    """One stage's cached teacher-forced cells, controls and generations."""

    def __init__(self, stage, adapter, session, tokenizer, protocol,
                 scope, rows, plan, visuals, encoder_means,
                 stops, limits, guard, profiler, started):
        import torch
        self.stage = stage
        self.name = stage["name"]
        self.adapter = adapter
        self.session = session
        self.tokenizer = tokenizer
        self.protocol = protocol
        self.scope = scope
        self.rows = rows
        self.plan = plan
        self.visuals = visuals
        self.encoder_means = encoder_means
        self.stops = stops
        self.limits = limits
        self.guard = guard
        self.profiler = profiler
        self.started = started
        self.rows_by_case = {str(row["case_id"]): row for row in rows}
        self.cells: dict[tuple[str | None, str], dict] = {}
        self.image_stats: dict[str, dict] = {}
        self.text_rms: dict[str, dict] = {}
        self.first_response: dict[str, int] = {}
        self.fit_rows: dict[str, dict] = {}
        self._torch = torch
        self.records: list[dict] = []
        self.cell_records: list[dict] = []
        self.unavailable_controls = 0

    # -- cached single computations -------------------------------------

    def _projected_stats(self, image_case_id):
        if image_case_id in self.image_stats:
            return self.image_stats[image_case_id]
        visual = self.visuals[(image_case_id, image_case_id)]
        with self._torch.no_grad():
            projected = self.adapter.project_visual(visual)[0]
            valid = visual.valid[0]
            patches = projected[valid]
            rms_all = float(patches.square().mean().sqrt().item())
            maximum = float(patches.abs().max().item())
            mean_patch = patches.mean(dim=0)
            mean_rms = float(mean_patch.square().mean().sqrt().item())
            spatial = float(patches.var(dim=0, unbiased=False).mean().item())
            mean_values = [float(value) for value in mean_patch.tolist()]
        require(bool(patches.isfinite().all()),
                "Projected visual features are nonfinite")
        stats = {
            "image_case_id": image_case_id,
            "image_sha256": visual.image_sha256,
            "tokens": int(valid.sum().item()),
            "projected_rms": rms_all,
            "projected_max": maximum,
            "mean_vector_rms": mean_rms,
            "spatial_variance": spatial,
            "mean_patch": mean_values,
        }
        self.image_stats[image_case_id] = stats
        return stats

    def _first_response_token(self, case_id):
        if case_id not in self.first_response:
            response = self.tokenizer(self.rows_by_case[case_id]["response"],
                                      add_special_tokens=False)["input_ids"]
            require(bool(response), "A training response tokenizes to nothing")
            self.first_response[case_id] = int(response[0])
        return self.first_response[case_id]

    def _text_embedding_rms(self, case_id, input_ids, prompt_tokens):
        if case_id in self.text_rms:
            return self.text_rms[case_id]
        with self._torch.no_grad():
            embeddings = self.adapter.base.get_input_embeddings()(
                input_ids)[0]
            prompt = embeddings[:prompt_tokens]
            response = embeddings[prompt_tokens:]
            record = {
                "prompt_rms": float(prompt.square().mean().sqrt().item()),
                "response_rms": float(response.square().mean().sqrt().item()),
            }
        self.text_rms[case_id] = record
        return record

    def cell(self, image_case_id, target_case_id):
        """Teacher-forced evidence for one image/target combination."""
        key = (image_case_id, target_case_id)
        if key in self.cells:
            return self.cells[key]
        from run_vision_curriculum import (
            _example_tensors, curriculum_prompt, task_graph)
        target = self.rows_by_case[target_case_id]
        prompt_text = curriculum_prompt(target["question"])
        prompt_tensor, input_ids, labels = _example_tensors(
            self.session, self.tokenizer, target,
            maximum_prompt=self.limits["max_prompt_tokens"],
            maximum_sequence=self.limits["max_sequence_tokens"])
        prompt_tokens = int(prompt_tensor.shape[1])
        visual = None
        if image_case_id is not None:
            visual = self.visuals.get((image_case_id, target_case_id))
            require(visual is not None,
                    "A teacher-forced cell lacks its task-bound visual input")
        request = self.session.prepare(
            prompt_ids=prompt_tensor,
            prompt_mask=self._torch.ones_like(prompt_tensor, dtype=self._torch.bool),
            query=prompt_text, scope=self.scope,
            task=task_graph(target["case_id"], prompt_text), training=False,
            visual=visual, enabled_ports=(), core_impl="none",
            enabled_branches=("memory",))
        try:
            with self._torch.no_grad():
                result = self.session.forward(
                    request, input_ids,
                    attention_mask=self._torch.ones_like(input_ids,
                                                         dtype=self._torch.bool),
                    labels=labels)
            require(result.logits is not None, "Teacher-forced forward returned no logits")
            require(bool(self._torch.isfinite(result.logits).all()),
                    "Teacher-forced logits are nonfinite")
            logits = result.logits[0]
            label_values = [int(value) for value in labels[0].tolist()]
            response_ids = [int(value) for value in input_ids[0, prompt_tokens:].tolist()]
            positions = position_nll(logits, label_values, prompt_tokens)
            require(positions["supervised_label_tokens"] == len(response_ids),
                    "Supervised label positions differ from the response length")
            partner = self.plan[str(target["case_id"])]["sibling_case_id"]
            opposite = (self._first_response_token(partner)
                        if partner is not None else None)
            first = first_token_analysis(
                logits[prompt_tokens - 1],
                self._first_response_token(target_case_id),
                opposite_token=opposite, top_k_size=5)
            eos = eos_pair_analysis(logits[len(label_values) - 2], self.stops)
        finally:
            request.release()
        stats = (None if image_case_id is None
                 else self._projected_stats(image_case_id))
        cell = {
            "image_case_id": image_case_id,
            "image_sha256": (None if stats is None else stats["image_sha256"]),
            "target_case_id": target_case_id,
            "prompt_tokens": prompt_tokens,
            "prompt_ids": [int(value) for value in prompt_tensor[0].tolist()],
            "prompt_mask": [1] * prompt_tokens,
            "response_ids": response_ids,
            "input_ids": [int(value) for value in input_ids[0].tolist()],
            "labels": label_values,
            "target_eos_token": response_ids[-1],
            "stop_tokens": list(self.stops),
            "positions": positions,
            "first_token": first,
            "eos": eos,
            "projected": (None if stats is None else {
                "projected_rms": stats["projected_rms"],
                "projected_max": stats["projected_max"],
                "mean_vector_rms": stats["mean_vector_rms"],
                "spatial_variance": stats["spatial_variance"],
                "tokens": stats["tokens"],
            }),
            "text_embedding": self._text_embedding_rms(
                target_case_id, input_ids, prompt_tokens),
        }
        self.cells[key] = cell
        if target_case_id not in self.fit_rows:
            fit = self.protocol["fit_contract"]
            self.fit_rows[target_case_id] = fit_contract_row_report(
                example_id=str(target["example_id"]),
                case_id=target_case_id, kind=str(target["kind"]),
                prompt_tokens=prompt_tokens, input_ids=cell["input_ids"],
                labels=label_values, eos_token=int(
                    self.tokenizer.eos_token_id),
                stop_tokens=self.stops,
                pad_token=getattr(self.tokenizer, "pad_token_id", None),
                question=str(target["question"]),
                caption_template=fit["caption_template"],
                loss=fit["loss"],
                max_sequence_tokens=self.limits["max_sequence_tokens"])
        return cell

    def _generation(self, row, control, image_case_id, cell):
        from run_vision_curriculum import curriculum_prompt, generate_row
        image_row = (None if image_case_id is None
                     else self.rows_by_case[image_case_id])
        visual = None
        if image_case_id is not None:
            visual = self.visuals[(image_case_id, str(row["case_id"]))]
        maximum_new = (int(self.limits["caption_max_new_tokens"])
                       if row["kind"] == "caption"
                       else int(self.limits["max_new_tokens"]))
        generated = generate_row(
            self.session, self.tokenizer, control=control, case=row,
            question=curriculum_prompt(row["question"]), visual=visual,
            scope=self.scope,
            maximum_prompt=self.limits["max_prompt_tokens"],
            maximum_new=maximum_new,
            image_case_id=image_case_id,
            image_sha256=(None if image_row is None
                          else image_row["image_sha256"]))
        evidence = generation_evidence(
            generated["generated_ids"], generated["generated_text"],
            stops=self.stops,
            prefill_argmax=cell["first_token"]["argmax_token"],
            max_new_tokens=maximum_new)
        expected_prompt_sha = hashlib.sha256(
            json.dumps([int(value) for value in cell["prompt_ids"]]
                       ).encode()).hexdigest()
        require(generated["prompt_sha256"] == expected_prompt_sha,
                "Generation prompt differs from the teacher-forced prompt")
        require(generated["image_case_id"] == image_case_id
                and generated["image_sha256"] == cell["image_sha256"],
                "Generation image identity differs from its cell")
        require(list(generated.get("stop_tokens", [])) == list(self.stops),
                "generate_row stop set differs from the declared set")
        require(evidence["ended_by_eos"] == generated["ended_by_eos"]
                and evidence["stop_reason"] == generated.get(
                    "stop_reason", evidence["stop_reason"]),
                "generate_row stop evidence conflicts with the declared set")
        evidence["generate_row_stop_tokens"] = generated.get("stop_tokens")
        return evidence

    # -- one stage --------------------------------------------------------

    def run(self):
        stage_started = time.monotonic()
        image_case_for = {
            "correct": lambda plan_row: plan_row["correct_case_id"],
            "sibling": lambda plan_row: plan_row["sibling_case_id"],
            "shuffled": lambda plan_row: plan_row["shuffled_case_id"],
            "noimage": lambda plan_row: None,
        }
        for position, row in enumerate(self.rows, 1):
            _check_wall(self.started, self.limits["max_seconds"],
                        "teacher-forced pass")
            if self.guard is not None and position % 5 == 0:
                self.guard.sample(index=position, event=f"{self.name}_rows")
            plan_row = self.plan[str(row["case_id"])]
            for control in CONTROL_NAMES:
                _check_wall(self.started, self.limits["max_seconds"],
                            "control generation")
                image_case_id = image_case_for[control](plan_row)
                if control == "sibling" and not plan_row["sibling_available"]:
                    self.records.append({
                        "format": ROW_FORMAT,
                        "stage": self.name,
                        "example_id": str(row["example_id"]),
                        "case_id": str(row["case_id"]),
                        "kind": str(row["kind"]),
                        "pair_id": str(row["pair_id"]),
                        "task": row.get("task"),
                        "control": control,
                        "image_role": "changed_sibling_image",
                        "image_case_id": None,
                        "image_sha256": None,
                        "pair_opposite_case_id": None,
                        "opposite_target_available": False,
                        "available": False,
                        "unavailable_reason": (
                            "no sibling row for this pair id exists in the "
                            "sealed training payload"),
                    })
                    self.unavailable_controls += 1
                    continue
                require((image_case_id is not None)
                        or control == "noimage",
                        "Only the noimage control may omit its image")
                cell = self.cell(image_case_id, str(row["case_id"]))
                generation = self._generation(row, control, image_case_id, cell)
                self.records.append({
                    "format": ROW_FORMAT,
                    "stage": self.name,
                    "example_id": str(row["example_id"]),
                    "case_id": str(row["case_id"]),
                    "kind": str(row["kind"]),
                    "pair_id": str(row["pair_id"]),
                    "task": row.get("task"),
                    "control": control,
                    "image_role": {
                        "correct": "correct_image",
                        "sibling": "changed_sibling_image",
                        "shuffled": "cross_pair_shuffled_image",
                        "noimage": "no_image",
                    }[control],
                    "image_case_id": image_case_id,
                    "image_sha256": cell["image_sha256"],
                    "pair_opposite_case_id": plan_row["sibling_case_id"],
                    "opposite_target_available": plan_row["sibling_available"],
                    "available": True,
                    "unavailable_reason": None,
                    "prompt_tokens": cell["prompt_tokens"],
                    "prompt_ids": cell["prompt_ids"],
                    "prompt_mask": cell["prompt_mask"],
                    "response_ids": cell["response_ids"],
                    "labels": cell["labels"],
                    "label_shift": 1,
                    "target_eos_token": cell["target_eos_token"],
                    "stop_tokens": cell["stop_tokens"],
                    "positions": cell["positions"],
                    "first_token": cell["first_token"],
                    "eos": cell["eos"],
                    "projected": cell["projected"],
                    "text_embedding": cell["text_embedding"],
                    "generation": generation,
                    "fit_contract": {
                        key: self.fit_rows[str(row["case_id"])][key]
                        for key in ("prompt_tokens", "response_tokens",
                                    "supervised_label_tokens",
                                    "shifted_supervision_tokens",
                                    "loss_tokens", "label_shift",
                                    "label_positions", "predictor_positions",
                                    "response_eos_token", "stop_tokens",
                                    "eos_in_stop_set", "pad_token",
                                    "pad_positions_before_eos",
                                    "template_checked", "response_only_labels",
                                    "loss", "contract_valid")
                    },
                })
        groups = pair_groups(self.rows)
        for full in [*groups["answer"], *groups["caption"]]:
            _check_wall(self.started, self.limits["max_seconds"],
                        "image/target comparisons")
            for cell_spec in comparison_cells(full, self.plan):
                target_case_id = str(cell_spec["target_case_id"])
                cell = self.cell(cell_spec["image_case_id"], target_case_id)
                self.cell_records.append({
                    "format": CELL_FORMAT,
                    "stage": self.name,
                    "pair_id": cell_spec["pair_id"],
                    "kind": cell_spec["kind"],
                    "pair_complete": cell_spec["pair_complete"],
                    "image_role": cell_spec["image_role"],
                    "target_role": cell_spec["target_role"],
                    "image_case_id": cell_spec["image_case_id"],
                    "target_case_id": target_case_id,
                    "image_sha256": cell["image_sha256"],
                    "positions": cell["positions"],
                    "first_token": cell["first_token"],
                    "eos": cell["eos"],
                    "projected": cell["projected"],
                })
        coverage = fit_coverage(
            [self.fit_rows[case_id] for case_id in sorted(self.fit_rows)],
            [str(row["example_id"]) for row in self.rows])
        require_fit_coverage(coverage, expected_kinds={
            "answer": self.protocol["data"]["counts"]["train_answers"],
            "caption": self.protocol["data"]["counts"]["train_captions"],
        })
        text_rows = [
            {"case_id": record["case_id"],
             "gold_token": record["first_token"]["gold_token"],
             "gold_rank": record["first_token"]["gold_rank"],
             "argmax_token": record["first_token"]["argmax_token"]}
            for record in self.records
            if record["control"] == "noimage" and record["kind"] == "answer"]
        text_ceiling = first_token_ceiling_summary(
            text_rows,
            expected_cases=self.protocol["data"]["counts"]["train_answers"])
        feature = feature_summary(
            self.name, self.image_stats, self.encoder_means, self.plan,
            [self.text_rms[case_id]["response_rms"]
             for case_id in sorted(self.text_rms)])
        self.profiler.mark(f"{self.name}_complete",
                           rows=len(self.records), cells=len(self.cells))
        return {
            "stage": self.name,
            "kind": self.stage["kind"],
            "updates": 0,
            "records": self.records,
            "rows": len(self.rows),
            "rows_with_all_four_controls": len(self.rows) - sum(
                1 for row in self.rows
                if not self.plan[str(row["case_id"])]["sibling_available"]),
            "computed_control_records": (len(self.records)
                                        - self.unavailable_controls),
            "unavailable_controls": self.unavailable_controls,
            "cells": self.cell_records,
            "teacher_forced_cells": len(self.cells),
            "fit_rows": [self.fit_rows[case_id]
                         for case_id in sorted(self.fit_rows)],
            "fit_coverage": coverage,
            "control_summary": control_summary(self.records),
            "text_ceiling": text_ceiling,
            "feature_diagnostics": feature,
            "image_stats": self.image_stats,
            "text_embedding": self.text_rms,
            "duration_seconds": time.monotonic() - stage_started,
        }


# ---------------------------------------------------------------------------
# Native execution
# ---------------------------------------------------------------------------


def _snapshot_new_parameters(adapter):
    return {name: parameter.detach().cpu().clone()
            for name, parameter in adapter.new_named_parameters()}


def _changed(before, after):
    import torch
    return sorted(name for name in before
                  if not torch.equal(before[name], after[name]))


def _failure_report(output, exc, *, stop_report=None):
    payload = {
        "format": REPORT_FORMAT + ".failure",
        "generated_at_utc": _now(),
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "stop_conditions": stop_report,
        "claims": claims_block(),
    }
    try:
        _write_json(Path(output) / "failure_report.json", payload)
    except Exception:  # noqa: BLE001 - never mask the original failure
        pass


def execute(args, config, pins):
    import torch
    from replica_lab.architecture.loading import (
        exclusive_mps_worker, load_local_mps)
    from replica_lab.profiling import RunProfiler
    from run_vision_curriculum import (
        Guard, feature_service, freeze_projector_only, initialize_projector,
        make_session, protocol_scope, projector_state_sha256,
        verify_source_and_spec)
    from replica_lab.vision_qualification import validate_extension_inventory

    protocol = pins["protocol"]
    output = Path(args.output)
    require(Path(args.model_path).resolve()
            == (ROOT / config["source"]["model_path"]).resolve(),
            "CLI model path differs from the pinned sealed source")
    require(Path(args.model_lock).resolve()
            == (ROOT / config["source"]["model_lock"]).resolve(),
            "CLI model lock differs from the pinned lock")
    rows, layout = load_training_rows(config, pins)
    plan = control_plan(rows)
    limits = config["limits"]
    started = time.monotonic()
    guard = None
    (output / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False) + "\n")
    (output / "protocol.json").write_bytes(pins["protocol_path"].read_bytes())
    try:
        guard = Guard(protocol)
        scope = protocol_scope(protocol)
        with exclusive_mps_worker(ROOT):
            profiler = RunProfiler("mps", output)
            profiler.start()
            try:
                _, spec, _ = verify_source_and_spec(protocol, "mps")
                adapter, tokenizer, _ = load_local_mps(
                    args.model_path, args.model_lock, vision=spec,
                    profiler=profiler)
                guard.sample(index=0, event="model_loaded")
                profiler.mark("model_loaded")
                stops, stop_source = resolve_stop_tokens(
                    getattr(adapter.base.config, "eos_token_id", None),
                    getattr(tokenizer, "eos_token_id", None))
                require(tuple(stops) == EXPECTED_STOP_TOKENS,
                        f"Model stop set {stops} differs from {EXPECTED_STOP_TOKENS}")
                require(int(tokenizer.eos_token_id)
                        == int(config["target_eos_token"]),
                        "Tokenizer EOS differs from the declared target EOS")
                adapter.eval()
                freeze_projector_only(adapter)
                session = make_session(adapter, tokenizer, output)
                service = feature_service(protocol, "mps")
                guard.sample(index=0, event="feature_service")
                profiler.mark("feature_service")

                # Observe every needed (image, target-task) pair exactly once.
                # MemorySession.prepare binds visual.task_id to the request task,
                # so a sibling or shuffled image must be observed under the
                # target case's task id.  Features are frozen and stage-invariant.
                rows_by_case = {str(row["case_id"]): row for row in rows}
                groups = pair_groups(rows)
                visual_pairs = set()
                for row in rows:
                    case_id = str(row["case_id"])
                    plan_row = plan[case_id]
                    visual_pairs.add((case_id, case_id))
                    if plan_row["sibling_available"]:
                        visual_pairs.add((str(plan_row["sibling_case_id"]),
                                          case_id))
                    visual_pairs.add((str(plan_row["shuffled_case_id"]),
                                      case_id))
                for group in [*groups["answer"], *groups["caption"]]:
                    for cell_spec in comparison_cells(group, plan):
                        visual_pairs.add((str(cell_spec["image_case_id"]),
                                          str(cell_spec["target_case_id"])))
                visuals = {}
                for image_case, target_case in sorted(visual_pairs):
                    image_row = rows_by_case[image_case]
                    visual = service.observe(
                        str(image_row["image_path"]),
                        source_id=protocol["vision_source"]["source_id"],
                        task_id=target_case, scope=scope, observed_at=0,
                        source_epoch=session.ledger.epoch)
                    require(visual.producer == spec.digest,
                            "Observed visual producer differs")
                    require(visual.task_id == target_case,
                            "Observed visual task binding differs")
                    require(visual.image_sha256 == image_row["image_sha256"],
                            "Observed image hash differs from its row")
                    require(bool(visual.features.isfinite().all()),
                            "Visual features are nonfinite")
                    visuals[(image_case, target_case)] = visual
                encoder_means = {}
                for row in rows:
                    case_id = str(row["case_id"])
                    visual = visuals[(case_id, case_id)]
                    with torch.no_grad():
                        valid = visual.valid[0]
                        mean_patch = visual.features[0][valid].mean(dim=0)
                        encoder_means[case_id] = [
                            float(value) for value in mean_patch.tolist()]
                guard.sample(index=0, event="encoder_features")
                profiler.mark("encoder_features", images=len(rows),
                              bound_visuals=len(visuals))

                # Load the pinned extension once so the fresh stage can share
                # every frozen sidecar; only the projector differs.
                receipt = validate_extension_inventory(
                    pins["extension_path"],
                    expected_names=[name for name, _ in
                                    adapter.new_named_parameters()],
                    vision_pins=protocol["vision"])
                require(receipt["weights_sha256"]
                        == config["extension"]["weights_sha256"],
                        "Validated extension weights differ")
                adapter.load_extension(pins["extension_path"])
                extension_snapshot = _snapshot_new_parameters(adapter)
                extension_projector_sha = projector_state_sha256(
                    adapter.vision_projector)

                # Stage A: fresh seed-29 projector on matched frozen sidecars.
                initialize_projector(adapter.vision_projector,
                                     int(config["stages"][0]["init_seed"]))
                fresh_snapshot = _snapshot_new_parameters(adapter)
                fresh_changed = _changed(extension_snapshot, fresh_snapshot)
                fresh_frozen_changed = [name for name in fresh_changed
                                        if not name.startswith("vision_projector.")]
                require(not fresh_frozen_changed,
                        f"Fresh stage changed frozen tensors: {fresh_frozen_changed}")
                fresh_projector_sha = projector_state_sha256(
                    adapter.vision_projector)
                require(fresh_projector_sha != extension_projector_sha,
                        "Fresh initialization reproduced the trained projector")
                require(fresh_projector_sha not in set(
                    protocol["train"]["forbidden_parent_weights_sha256"]),
                    "Fresh initialization matches the rejected parent")
                profiler.mark("fresh_init", changed=fresh_changed)
                fresh = StageEvaluator(
                    config["stages"][0], adapter, session, tokenizer,
                    protocol, scope, rows, plan, visuals,
                    encoder_means, stops, limits, guard, profiler, started).run()
                fresh["projector_state_sha256"] = fresh_projector_sha
                fresh["frozen_component_source"] = "pinned_train04_extension"
                fresh["changed_extension_tensors"] = fresh_changed
                fresh["frozen_extension_tensors_unchanged"] = (
                    len(extension_snapshot) - len(fresh_changed))

                # Stage B: reload the pinned extension bitwise.
                adapter.load_extension(pins["extension_path"])
                reload_snapshot = _snapshot_new_parameters(adapter)
                require(not _changed(extension_snapshot, reload_snapshot),
                        "The pinned extension reload is not bitwise")
                require(projector_state_sha256(adapter.vision_projector)
                        == extension_projector_sha,
                        "Extension projector reload is not bitwise")
                profiler.mark("extension_reload")
                extension = StageEvaluator(
                    config["stages"][1], adapter, session, tokenizer,
                    protocol, scope, rows, plan, visuals,
                    encoder_means, stops, limits, guard, profiler, started).run()
                extension["projector_state_sha256"] = extension_projector_sha
                extension["changed_extension_tensors"] = _changed(
                    fresh_snapshot, reload_snapshot)
                extension["frozen_extension_tensors_unchanged"] = (
                    len(reload_snapshot) - len(extension["changed_extension_tensors"]))
                require(set(extension["changed_extension_tensors"])
                        <= {"vision_projector.0.weight", "vision_projector.0.bias",
                            "vision_projector.1.weight", "vision_projector.1.bias",
                            "vision_projector.3.weight", "vision_projector.3.bias"},
                        "Stages differ outside the six projector tensors")
                require(fresh["frozen_extension_tensors_unchanged"]
                        == extension["frozen_extension_tensors_unchanged"],
                        "Frozen tensor counts differ between stages")

                # Feature ceiling uses frozen 768-d encoder means only, over
                # the five complete opposite-answer pairs.
                answer_groups = [group for group in pair_groups(rows)["answer"]
                                 if group["complete"]]
                require(len(answer_groups) == 5,
                        "The feature ceiling requires five complete answer pairs")
                ceiling_pairs = []
                for group in answer_groups:
                    members = {case_id.rsplit("-", 1)[-1]: case_id
                               for case_id in group["case_ids"]}
                    ceiling_pairs.append({"pair_id": group["pair_id"],
                                          "a_case_id": members["a"],
                                          "b_case_id": members["b"]})
                ceiling = feature_ceiling(ceiling_pairs, encoder_means,
                                         expected_pairs=5)
                text_invariant = (
                    fresh["text_ceiling"]["rows"] == extension["text_ceiling"]["rows"])
                profiler.mark("diagnostics_complete")
            except BaseException:
                profiler.stop()
                raise
            summary = profiler.stop()

        stage_payload = {"fresh_init_seed29": fresh,
                         "train04_extension": extension}
        require(fresh["fit_rows"] == extension["fit_rows"],
                "Fit contract rows differ between stages")
        require(fresh["fit_coverage"] == extension["fit_coverage"],
                "Fit contract coverage differs between stages")
        rows_all = fresh["records"] + extension["records"]
        cells_all = fresh["cells"] + extension["cells"]
        fit_rows = fresh["fit_rows"]
        _write_jsonl(output / "rows.jsonl", rows_all)
        _write_jsonl(output / "cells.jsonl", cells_all)
        _write_jsonl(output / "fit_contract_rows.jsonl", fit_rows)
        _write_json(output / "fit_contract_coverage.json", fresh["fit_coverage"])
        features = {
            "encoder_mean_patch": encoder_means,
            "stages": {
                name: {
                    "image_stats": payload["image_stats"],
                    "text_embedding": payload["text_embedding"],
                    "summary": payload["feature_diagnostics"],
                }
                for name, payload in stage_payload.items()
            },
        }
        _write_json(output / "features.json", features)

        data_bytes = sum(path.stat().st_size for path in output.rglob("*")
                         if path.is_file())
        stop_report = stop_conditions_report(
            wall_seconds=time.monotonic() - started,
            wall_limit_seconds=limits["max_seconds"],
            artifact_bytes=data_bytes,
            artifact_limit_bytes=limits["artifact_bytes"],
            updates=0, optimizer_steps=0, gradients_computed=0,
            checkpoint_writes=0, extension_writes=0, model_loads=1,
            extension_loads=2, downloads=0, cpu_fallback=False,
            sealed_evaluation_access=0,
            guard_stop_reason=guard.report().get("stop_reason"))
        require_stop_conditions(stop_report)
        report = {
            "format": REPORT_FORMAT,
            "diagnostic": config["version"],
            "declared_at_utc": config["declared_at_utc"],
            "generated_at_utc": _now(),
            "command": [sys.executable, *sys.argv],
            "pins": pins["pins"],
            "policy_declared": config["policy"],
            "limits": limits,
            "generation_stop_set": {
                "stop_tokens": list(stops),
                "source": stop_source,
                "expected": list(EXPECTED_STOP_TOKENS),
                "target_eos_token": int(config["target_eos_token"]),
                "evidence": ("Generate the stop evidence from this set; "
                             "tokenizer.eos_token_id alone drops <|im_end|>."),
            },
            "pair_layout": layout,
            "controls": {
                "names": list(CONTROL_NAMES),
                "selection": ("deterministic by sorted pair id; no expected "
                              "target text participates"),
                "rows": len(rows),
                "rows_with_all_four_controls": sum(
                    1 for entry in plan.values()
                    if entry["sibling_available"]),
                "sibling_unavailable_case_ids": sorted(
                    case_id for case_id, entry in plan.items()
                    if not entry["sibling_available"]),
                "per_row": plan,
            },
            "stages": {
                name: {
                    "kind": payload["kind"],
                    "updates": 0,
                    "projector_state_sha256": payload["projector_state_sha256"],
                    "frozen_component_source": payload.get(
                        "frozen_component_source", "pinned_train04_extension"),
                    "changed_extension_tensors": payload[
                        "changed_extension_tensors"],
                    "frozen_extension_tensors_unchanged": payload[
                        "frozen_extension_tensors_unchanged"],
                    "rows": payload["rows"],
                    "rows_with_all_four_controls": payload[
                        "rows_with_all_four_controls"],
                    "computed_control_records": payload[
                        "computed_control_records"],
                    "unavailable_controls": payload["unavailable_controls"],
                    "evidence_records": len(payload["records"]),
                    "teacher_forced_cells": payload["teacher_forced_cells"],
                    "comparison_cells": len(payload["cells"]),
                    "control_summary": payload["control_summary"],
                    "text_ceiling": payload["text_ceiling"],
                    "feature_diagnostics": payload["feature_diagnostics"],
                    "fit_coverage": payload["fit_coverage"],
                    "duration_seconds": payload["duration_seconds"],
                }
                for name, payload in stage_payload.items()
            },
            "text_ceiling_stage_invariant": text_invariant,
            "feature_ceiling": ceiling,
            "fit_contract": {
                "coverage": fresh["fit_coverage"],
                "expected_kinds": {
                    "answer": protocol["data"]["counts"]["train_answers"],
                    "caption": protocol["data"]["counts"]["train_captions"],
                },
                "rows_retained": len(fit_rows),
                "contract": protocol["fit_contract"],
                "scope": ("per-row prompt mask, response labels, shifted "
                          "supervised positions and target EOS; every row, "
                          "captions included, retained exactly once"),
            },
            "resource": {
                "guard": guard.report(),
                "profile": summary,
            },
            "stop_conditions": stop_report,
            "claims": claims_block(),
            "limitations": config["limitations"],
        }
        # Stabilize the recorded artifact-byte estimate to include this report.
        for _ in range(3):
            candidate = json.dumps(report, indent=2, allow_nan=False) + "\n"
            total = data_bytes + len(candidate.encode())
            report["artifact_bytes_estimate_including_report"] = total
            report["stop_conditions"] = {
                **report["stop_conditions"],
                "artifact_bytes_estimate_including_report": total,
                "artifact_bytes_scope": (
                    "data payloads plus this report; manifest.json records the "
                    "final total including itself"),
            }
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")
        total = sum(path.stat().st_size for path in output.rglob("*")
                    if path.is_file())
        if total > int(limits["artifact_bytes"]):
            raise DiagnosticError(
                f"Artifact bytes {total} exceed the declared "
                f"{limits['artifact_bytes']} ceiling")
        manifest = {
            "format": REPORT_FORMAT + ".artifacts",
            "generated_at_utc": _now(),
            "total_bytes": total,
            "files_exclude": "manifest.json",
            "limit_bytes": limits["artifact_bytes"],
            "files": {
                str(path.relative_to(output)): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "manifest.json"
            },
        }
        _write_json(output / "manifest.json", manifest)
        final_bytes = sum(path.stat().st_size for path in output.rglob("*")
                          if path.is_file())
        if final_bytes > int(limits["artifact_bytes"]):
            raise DiagnosticError(
                f"Final artifact bytes {final_bytes} exceed the declared "
                f"{limits['artifact_bytes']} ceiling")
        return report
    except BaseException as exc:
        elapsed = time.monotonic() - started
        stop_report = stop_conditions_report(
            wall_seconds=elapsed,
            wall_limit_seconds=limits["max_seconds"],
            artifact_bytes=sum(path.stat().st_size
                               for path in Path(args.output).rglob("*")
                               if path.is_file()),
            artifact_limit_bytes=limits["artifact_bytes"],
            updates=0, optimizer_steps=0, gradients_computed=0,
            checkpoint_writes=0, extension_writes=0, model_loads=1,
            extension_loads=2, downloads=0, cpu_fallback=False,
            sealed_evaluation_access=0,
            guard_stop_reason=(guard.report().get("stop_reason")
                               if guard is not None else None))
        _failure_report(output, exc, stop_report=stop_report)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/vision_collapse_diagnostic_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps",), default="mps")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if os.environ.get("REPLICA_CAPACITY_GUARD") != "1":
        parser.error("Use the capacity guard")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("CPU fallback must be disabled")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            parser.error(f"{name}=1 is required")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    if args.device != config["policy"]["device"]:
        parser.error("The declared diagnostic device must be mps")
    require(args.output not in (ROOT, Path("/")), "Refusing an unsafe output path")
    if args.output.exists() and (not args.output.is_dir()
                                 or any(args.output.iterdir())):
        parser.error("Use fresh output")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        pins = verify_pins(config)
        execute(args, config, pins)
    except BaseException as exc:
        if not (args.output / "failure_report.json").exists():
            _failure_report(args.output, exc)
        raise


if __name__ == "__main__":
    main()
