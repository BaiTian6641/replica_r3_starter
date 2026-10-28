"""Shared, preparation-only helpers for the DC-P1 and DC-R1 protocols.

Every function here is deterministic and standard-library only.  Nothing in this
module loads MiniCPM5, a tokenizer, torch, an optimizer or a native runtime.
The helpers deliberately keep four concerns separate:

* configuration and resource-guard validation;
* sealed-fixture hashing and fresh-path handling;
* answer-index/source/KV shortcut rejection for generated-answer scoring;
* fail-closed review/launch receipt handling.

They are shared by the protocol-specific builders, validators, scorers and
preflight commands introduced on 27 October 2026.  They do not modify, import or
reinterpret any historical runner.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]

COMMON_FORMAT = "replica.dc_protocol_common.v1"
PREFLIGHT_FORMAT = "replica.dc_protocol.preflight_receipt.v1"
AUTHORIZATION_FORMAT = "replica.dc_protocol.authorization.v1"
WORK_BUDGET_FORMAT = "replica.dc_protocol.work_budget.v1"

# Unchanged fleet-wide resource policy.  These are deliberately duplicated from
# the sealed behavioral-reference protocol rather than imported from a runner.
GUARD_BYTES = {
    "peak_driver_bytes_max": 53687091200,
    "sustained_swap_growth_bytes_max": 536870912,
    "min_host_free_fraction": 0.15,
    "exclusive_worker": True,
    "cpu_fallback": False,
    "mps_fallback": False,
}

BASELINE_OPTIMIZER = "NativeFusedAdamW"
BASELINE_OPTIMIZER_ALIAS = "adamw_fused"
BASELINE_MODEL = {
    "repo_id": "openbmb/MiniCPM5-2B",
    "revision": "f97400052a43d642bbc6e9975e2397e3ae6a6b52",
    "parameter_count": 2516756480,
    "model_path": "checkpoints/minicpm5_source",
    "lock_path": "configs/minicpm5_sealed_lock.local.json",
    "local_bytes_sealed": True,
    "upstream_authenticity_independently_verified": False,
}

TOKENIZER_FILES = {
    "tokenizer.json":
        "3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81",
    "tokenizer_config.json":
        "e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b",
}
TOKENIZER_BINDING = {
    "repository": BASELINE_MODEL["repo_id"],
    "revision": BASELINE_MODEL["revision"],
    "files": dict(TOKENIZER_FILES),
    "load_mode": "offline_local_files_only",
    "trust_remote_code": False,
    "fast_tokenizer": True,
    "decode_policy": {
        "version": "replica.dc.generated_decode.v1",
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
        "strip_outer_whitespace": True,
        "terminal_stop_tokens_excluded_before_decode": True,
        "answer_selection_allowed": False,
        "candidate_answers_used": False,
        "empty_generation_credit": False,
        "target_eos_token": 1,
        "stop_tokens": [1, 130073],
    },
}

# A generated-answer row that carries any of these fields is rejected before a
# score is computed.  The list is intentionally broad: an index into a closed
# answer vocabulary is still an answer-index bypass even if it is called a
# "route", "slot" or "candidate".
ANSWER_SHORTCUT_FIELDS = frozenset({
    "answer", "answer_id", "answer_index", "answer_label", "answer_logit",
    "answer_position", "answer_token", "candidate_answer", "candidate_answers",
    "correct", "correct_index", "expected", "expected_answer", "expected_token",
    "gold", "gold_answer", "gold_id", "gold_index", "gold_token", "label",
    "label_index", "target", "target_answer", "target_id", "target_index",
    "target_token", "value", "value_index",
})
EVALUATION_CASE_FORBIDDEN_FIELDS = frozenset({
    "answer", "answer_id", "answer_index", "expected", "expected_answer",
    "expected_token", "gold", "gold_answer", "gold_id", "gold_index",
    "label", "source", "source_context", "source_text", "target",
    "target_answer", "target_id", "target_index", "target_token", "value",
})
ROUTING_FORBIDDEN_FIELDS = frozenset({
    "answer", "answer_id", "answer_index", "answer_label", "expected",
    "expected_answer", "gold", "gold_answer", "gold_id", "gold_index",
    "label", "target", "target_answer", "target_id", "target_index",
    "value", "value_index", "world_answer", "record_answer",
})
SOURCE_FREE_FIELDS = {
    "source_context_present": False,
    "source_kv_loaded": False,
    "source_prefill_tokens": 0,
    "source_kv_reused": False,
    "prompt_source_spans": 0,
}


class ProtocolError(ValueError):
    """One fail-closed preparation check failed."""

    def __init__(self, check: str, detail: str):
        super().__init__(f"{check}: {detail}")
        self.check = check
        self.detail = detail

    def as_dict(self) -> dict[str, str]:
        return {"check": self.check, "detail": self.detail}


def require(condition: bool, check: str, detail: str) -> None:
    if not condition:
        raise ProtocolError(check, detail)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_hex64(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def read_json(path: str | Path) -> dict:
    path = Path(path)
    require(path.is_file(), "read", f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProtocolError("read", f"unreadable JSON {path}: {error}") from error
    require(isinstance(value, dict), "read", f"JSON root is not an object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    require(path.is_file(), "read", f"missing JSONL file: {path}")
    rows: list[dict] = []
    for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ProtocolError(
                "read", f"{path}:{number} is not JSON: {error}") from error
        require(isinstance(row, dict), "read",
                f"{path}:{number} is not a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(canonical(row) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def require_fresh_path(path: str | Path, check: str = "fresh_output") -> Path:
    target = Path(path)
    require(not target.exists(), check,
            f"path is not fresh; refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def normalize_answer(value: Any) -> str:
    require(isinstance(value, str), "answer", "answer must be a string")
    return " ".join(value.strip().casefold().split())


def find_forbidden_fields(
        value: Any, forbidden: Iterable[str], *, prefix: str = "") -> list[str]:
    """Return dotted paths whose final key is forbidden, recursively."""
    forbidden_set = set(forbidden)
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).casefold() in forbidden_set:
                found.append(path)
            found.extend(find_forbidden_fields(
                child, forbidden_set, prefix=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_forbidden_fields(
                child, forbidden_set, prefix=f"{prefix}[{index}]"))
    return found


def validate_guards(config: Mapping[str, Any],
                    path: Sequence[str] = ("resource_guard",)) -> None:
    node: Any = config
    for key in path:
        require(isinstance(node, Mapping) and key in node, "resource_guard",
                f"missing {'.'.join(path)}")
        node = node[key]
    require(isinstance(node, Mapping), "resource_guard",
            "resource_guard must be an object")
    for key, expected in GUARD_BYTES.items():
        require(node.get(key) == expected, "resource_guard",
                f"{key}={node.get(key)!r} differs from sealed {expected!r}")


def validate_baseline_model(config: Mapping[str, Any]) -> None:
    node = config.get("baseline_model")
    require(isinstance(node, Mapping), "baseline_model",
            "baseline_model must be an object")
    for key, expected in BASELINE_MODEL.items():
        require(node.get(key) == expected, "baseline_model",
                f"{key}={node.get(key)!r} differs from sealed {expected!r}")
    optimizer = config.get("optimizer_policy", {})
    require(optimizer.get("primary") == BASELINE_OPTIMIZER, "optimizer_policy",
            f"primary optimizer must remain {BASELINE_OPTIMIZER}")
    require(optimizer.get("default_name") == BASELINE_OPTIMIZER_ALIAS,
            "optimizer_policy", f"default name must remain {BASELINE_OPTIMIZER_ALIAS}")
    require(optimizer.get("changed") is False, "optimizer_policy",
            "DC experiments may not change the optimizer")
    require(optimizer.get("mps_fallback_enabled") is False, "optimizer_policy",
            "MPS fallback must remain disabled")


def validate_tokenizer_binding(binding: Any) -> dict:
    require(isinstance(binding, Mapping), "tokenizer_binding",
            "tokenizer binding must be an object")
    for key in ("repository", "revision", "files", "load_mode",
                "trust_remote_code", "fast_tokenizer", "decode_policy"):
        require(key in binding, "tokenizer_binding", f"missing {key}")
    require(binding["repository"] == TOKENIZER_BINDING["repository"],
            "tokenizer_binding", "repository differs from sealed MiniCPM5")
    require(binding["revision"] == TOKENIZER_BINDING["revision"],
            "tokenizer_binding", "revision differs from sealed MiniCPM5")
    require(binding["load_mode"] == "offline_local_files_only",
            "tokenizer_binding", "offline local_files_only is mandatory")
    require(binding["trust_remote_code"] is False, "tokenizer_binding",
            "trust_remote_code must be false")
    require(binding["fast_tokenizer"] is True, "tokenizer_binding",
            "the sealed fast tokenizer is required")
    files = binding["files"]
    require(isinstance(files, Mapping) and set(files) == set(TOKENIZER_FILES),
            "tokenizer_binding", "tokenizer file set must be exactly sealed set")
    for name, digest in TOKENIZER_FILES.items():
        require(files.get(name) == digest, "tokenizer_binding",
                f"{name} hash differs from sealed tokenizer")
    policy = binding["decode_policy"]
    require(isinstance(policy, Mapping), "tokenizer_binding",
            "decode_policy must be an object")
    for key, expected in TOKENIZER_BINDING["decode_policy"].items():
        require(policy.get(key) == expected, "tokenizer_binding",
                f"decode_policy.{key} differs from sealed policy")
    return dict(binding)


def tokenizer_binding_sha(binding: Mapping[str, Any]) -> str:
    return sha256_json(validate_tokenizer_binding(binding))


def validate_tokenizer_directory(binding: Mapping[str, Any],
                                 directory: str | Path) -> None:
    directory = Path(directory)
    for name, expected in binding["files"].items():
        path = directory / name
        require(path.is_file(), "tokenizer_files", f"missing {path}")
        require(sha256_file(path) == expected, "tokenizer_files",
                f"{name} bytes differ from the sealed tokenizer")


def verify_decoded_text(
        prediction: Mapping[str, Any], binding: Mapping[str, Any],
        decode_fn: Callable[[Sequence[int]], str],
        *, check: str = "generated_decode") -> str:
    """Decode generated ids and require an exact match to the posted text.

    ``terminal_stop_tokens_excluded_before_decode`` is represented explicitly:
    the runner must remove every terminal stop id before handing ids to the
    decoder.  A trailing stop token in the posted ids is therefore a protocol
    violation, not something this helper silently repairs.
    """
    ids = prediction.get("generated_ids")
    text = prediction.get("generated_text")
    require(isinstance(ids, list) and all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0
        for item in ids), check, "generated_ids must be nonnegative ints")
    require(isinstance(text, str), check, "generated_text must be a string")
    policy = binding["decode_policy"]
    stops = set(policy["stop_tokens"])
    require(not any(item in stops for item in ids), check,
            "terminal stop ids must be excluded before decode")
    decoded = decode_fn(ids)
    require(isinstance(decoded, str), check, "decoder must return a string")
    if policy["strip_outer_whitespace"]:
        decoded = decoded.strip()
    if policy["skip_special_tokens"] is False:
        raise ProtocolError(check, "sealed decode policy expects skip_special_tokens")
    require(decoded == text, check,
            "decoded ids do not reproduce generated_text")
    require(prediction.get("tokenizer_binding_sha256")
            == tokenizer_binding_sha(binding), check,
            "prediction tokenizer binding differs from the sealed binding")
    require(prediction.get("decode_policy") == policy, check,
            "prediction decode policy differs from the sealed policy")
    if text == "":
        require(prediction.get("empty_generation") is True, check,
                "empty text must be declared as an empty generation")
    else:
        require(prediction.get("empty_generation") is False, check,
                "nonempty text cannot be declared empty")
    return text


def assert_no_answer_shortcut(row: Mapping[str, Any], *,
                              check: str = "answer_shortcut") -> None:
    found = find_forbidden_fields(row, ANSWER_SHORTCUT_FIELDS)
    require(not found, check,
            "answer-bearing fields are forbidden in this row: "
            + ", ".join(sorted(found)))


def assert_routing_is_answer_free(routing: Any) -> None:
    require(isinstance(routing, Mapping), "routing_metadata",
            "routing_metadata must be an object")
    found = find_forbidden_fields(routing, ROUTING_FORBIDDEN_FIELDS)
    require(not found, "routing_metadata",
            "answer-bearing routing metadata is forbidden: "
            + ", ".join(sorted(found)))


def assert_source_free_prediction(row: Mapping[str, Any]) -> None:
    for key, expected in SOURCE_FREE_FIELDS.items():
        require(row.get(key) == expected, "source_free_evaluation",
                f"{key}={row.get(key)!r}; expected {expected!r}")


def assert_prompt_matches(case: Mapping[str, Any],
                          prediction: Mapping[str, Any]) -> None:
    require(prediction.get("case_id") == case.get("case_id"),
            "case_binding", "prediction case_id does not match the case")
    require(prediction.get("prompt") == case.get("prompt"),
            "case_binding", "prediction prompt differs from the sealed case")
    require(prediction.get("prompt_sha256")
            == sha256_json({"prompt": case.get("prompt")}),
            "case_binding", "prediction prompt hash is not the sealed case hash")


def assert_target_absent(prompt: str, target: str, *,
                         check: str = "target_leak") -> None:
    normalized_prompt = normalize_answer(prompt)
    normalized_target = normalize_answer(target)
    require(normalized_target not in normalized_prompt, check,
            "the evaluation prompt contains the sealed target")


def paired_bootstrap_ci(
        differences: Sequence[float], *, seed: int = 20261027,
        resamples: int = 2000, alpha: float = 0.05) -> dict[str, float]:
    """Deterministic percentile bootstrap over paired source-group differences."""
    require(bool(differences), "paired_bootstrap", "no paired differences")
    require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in differences), "paired_bootstrap",
            "differences must be numeric")
    rng = random.Random(seed)
    count = len(differences)
    if count == 1:
        value = float(differences[0])
        return {"mean": value, "low": value, "high": value,
                "resamples": 1, "alpha": alpha}
    samples = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(count):
            total += float(differences[rng.randrange(count)])
        samples.append(total / count)
    samples.sort()
    low_index = max(0, int((alpha / 2.0) * resamples) - 1)
    high_index = min(resamples - 1, int((1.0 - alpha / 2.0) * resamples))
    return {
        "mean": sum(float(item) for item in differences) / count,
        "low": samples[low_index],
        "high": samples[high_index],
        "resamples": resamples,
        "alpha": alpha,
    }


def check_work_budget(accounting: Mapping[str, Any],
                      budget: Mapping[str, Any], *,
                      check: str = "work_budget") -> dict:
    """Fail closed on any physical-work counter above its declared ceiling."""
    require(isinstance(accounting, Mapping), check, "accounting must be an object")
    require(isinstance(budget, Mapping), check, "budget must be an object")
    observed: dict[str, int] = {}
    for name, raw in accounting.items():
        require(name.endswith("_count") or name.endswith("_calls")
                or name.endswith("_tokens") or name.endswith("_updates")
                or name.endswith("_seconds") or name in {
                    "model_loads", "tokenizer_loads", "downloads",
                    "retries", "workers", "optimizer_steps",
                    "backward_calls", "forward_calls", "generated_tokens"},
                check, f"unrecognized accounting counter {name!r}")
        require(isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0,
                check, f"{name} must be a nonnegative integer")
        observed[name] = raw
    ceilings = budget.get("physical_limits")
    require(isinstance(ceilings, Mapping), check,
            "budget.physical_limits is required")
    for name, value in observed.items():
        ceiling_name = f"max_{name}"
        require(ceiling_name in ceilings, check,
                f"no declared physical ceiling for {name!r}")
        ceiling = ceilings[ceiling_name]
        require(isinstance(ceiling, int) and not isinstance(ceiling, bool)
                and ceiling >= 0, check,
                f"{ceiling_name} must be a nonnegative integer")
        require(value <= ceiling, check,
                f"{name}={value} exceeds {ceiling_name}={ceiling}")
    return {
        "format": WORK_BUDGET_FORMAT,
        "observed": dict(sorted(observed.items())),
        "ceilings": {f"max_{name}": ceilings[f"max_{name}"]
                     for name in sorted(observed)},
        "within_budget": True,
    }


def preflight_receipt(
        *, config_path: str | Path, manifest_path: str | Path,
        config: Mapping[str, Any], protocol_id: str,
        manifest_sha256: str, authorization_path: str | Path | None,
        launch: bool) -> tuple[dict, int]:
    """Write no file here; return a deterministic receipt and exit code."""
    require(config.get("protocol_id") == protocol_id, "preflight",
            "config protocol_id differs from the command")
    require(config.get("classification", {}).get("preparation_only") is True,
            "preflight", "preparation_only must be true")
    require(config.get("classification", {}).get("native_execution_allowed")
            is False, "preflight", "native execution must be false in preparation")
    validate_guards(config)
    validate_baseline_model(config)
    launch_declaration = config.get("launch", {})
    output_root = launch_declaration.get("output_root")
    require(isinstance(output_root, str) and bool(output_root),
            "fresh_output", "launch.output_root is required")
    require(not (ROOT / output_root).exists(), "fresh_output",
            f"native output root already exists; refusing reuse: {output_root}")
    config_sha = sha256_file(config_path)
    manifest = read_json(manifest_path)
    require(manifest.get("protocol_id") == protocol_id, "preflight",
            "manifest protocol_id differs")
    require(manifest.get("manifest_sha256") == manifest_sha256,
            "preflight", "manifest self hash differs")
    for name, digest in manifest.get("files", {}).items():
        path = Path(manifest_path).parent / name
        require(path.is_file(), "preflight", f"missing prepared file {path}")
        require(sha256_file(path) == digest, "preflight",
                f"prepared file changed: {path}")
    base = {
        "format": PREFLIGHT_FORMAT,
        "protocol_id": protocol_id,
        "status": "review_ready",
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "prepared_manifest_path": str(manifest_path),
        "prepared_manifest_sha256": manifest_sha256,
        "native_execution": False,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "downloads": 0,
        "heldout_opened": False,
        "launch_requested": bool(launch),
        "launch_requirements": deepcopy(config.get("launch", {})),
    }
    if not launch:
        base["status"] = "review_ready"
        base["native_execution_status"] = "refused_preparation_only"
        base["exit_code"] = 0
        return base, 0

    # Launch is fail-closed even with an authorization receipt: this package
    # contains no native runner and must not invent one.
    require(authorization_path is not None, "authorization",
            "a separate authorization receipt is required before native work")
    authorization = read_json(authorization_path)
    base["authorization_receipt_sha256"] = sha256_file(authorization_path)
    require(authorization.get("format") == AUTHORIZATION_FORMAT,
            "authorization", "authorization format differs")
    require(authorization.get("protocol_id") == protocol_id,
            "authorization", "authorization protocol differs")
    require(authorization.get("config_sha256") == config_sha,
            "authorization", "authorization did not bind the exact config bytes")
    require(authorization.get("prepared_manifest_sha256") == manifest_sha256,
            "authorization", "authorization did not bind the exact fixture")
    require(authorization.get("native_execution_allowed") is True,
            "authorization", "native_execution_allowed is not true")
    require(authorization.get("single_attempt") is True,
            "authorization", "exactly one attempt must be authorized")
    require(isinstance(authorization.get("authorized_by"), str)
            and bool(authorization["authorized_by"].strip()),
            "authorization", "authorized_by must be a nonempty string")
    require(is_hex64(authorization.get("receipt_sha256")),
            "authorization", "receipt_sha256 must be a sealed digest")
    base["status"] = "refused_native_runner_absent"
    base["native_execution_status"] = (
        "authorization shape accepted, but no native runner exists in this "
        "preparation package; a separate runner and new review are mandatory")
    base["exit_code"] = 2
    return base, 2


def validate_manifest(root: str | Path, config: Mapping[str, Any], *,
                      expected_files: Sequence[str],
                      manifest_format: str,
                      protocol_id: str,
                      require_config_binding: bool = True,
                      skip_hash_files: Sequence[str] = ()) -> dict:
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("format") == manifest_format, "manifest",
            "manifest format differs")
    require(manifest.get("protocol_id") == protocol_id, "manifest",
            "manifest protocol_id differs")
    require(manifest.get("native_computation") is False, "manifest",
            "manifest must declare no native computation")
    require(manifest.get("model_loaded") is False, "manifest",
            "manifest must declare no model load")
    require(manifest.get("tokenizer_loaded") is False, "manifest",
            "manifest must declare no tokenizer load")
    require(manifest.get("downloads") == 0, "manifest",
            "manifest must declare zero downloads")
    require(manifest.get("heldout_opened") is False, "manifest",
            "manifest must declare closed held-out labels")
    require(isinstance(manifest.get("files"), Mapping), "manifest",
            "manifest files must be an object")
    skipped = set(skip_hash_files)
    require(skipped <= set(expected_files), "manifest",
            f"files deferred for later access are not declared: {sorted(skipped)}")
    require(set(manifest["files"]) == set(expected_files), "manifest",
            "manifest file set differs from the declared file set")
    for name, digest in manifest["files"].items():
        path = root / name
        require(path.is_file(), "manifest", f"missing prepared file {path}")
        if name in skipped:
            continue
        require(sha256_file(path) == digest, "manifest",
                f"prepared file hash differs: {name}")
    body = {key: value for key, value in manifest.items()
            if key != "manifest_sha256"}
    require(manifest.get("manifest_sha256") == sha256_json(body), "manifest",
            "manifest self hash differs")
    if require_config_binding:
        prepared = config.get("prepared", {})
        require(prepared.get("manifest_sha256") == manifest["manifest_sha256"],
                "config", "config does not bind the prepared manifest")
    return manifest


def seal_manifest(root: str | Path, body: Mapping[str, Any]) -> dict:
    root = Path(root)
    sealed = dict(body)
    sealed.pop("manifest_sha256", None)
    sealed["files"] = {
        name: sha256_file(root / name) for name in sorted(body["files"])}
    sealed["manifest_sha256"] = sha256_json(sealed)
    write_json(root / "manifest.json", sealed)
    return sealed


# ---------------------------------------------------------------------------
# v2 additions (27 October 2026)
#
# Everything below is additive: no v1 helper above changed behavior, no v1
# limit was overwritten and no v1 receipt format was reused for a different
# meaning.  The additions provide
#
# * a canonical physical-work counter registry shared by DC-P1 v2 and DC-R1 v2;
# * a fail-closed v2 budget checker with per-arm ceilings;
# * generic self-digested evidence seals;
# * offline git commit binding for an externally issued authorization.
# ---------------------------------------------------------------------------

WORK_BUDGET_V2_FORMAT = "replica.dc_protocol.work_budget.v2"

# Canonical execution-point counters.  A counter that is not in this registry
# must at least carry one of the declared unit suffixes; anything else is
# refused instead of being silently ignored.
WORK_COUNTERS = frozenset({
    "model_loads", "tokenizer_loads", "downloads", "retries", "workers",
    "training_forwards", "backward_calls", "optimizer_steps",
    "forward_calls", "generation_requests", "prefill_forwards",
    "decode_forwards", "autoregressive_forwards", "generation_prefill_forwards",
    "generation_decode_forwards", "generated_tokens",
    "processed_positions", "processed_text_positions",
    "parity_probe_forwards", "parity_processed_positions",
    "control_evaluations", "checkpoint_writes", "restarts",
    "restart_model_loads", "restart_forwards", "restart_backward_calls",
    "temporary_storage_bytes", "live_storage_bytes",
    "layer_readout_streams",
    "restart_tokenizer_loads", "restart_training_forwards",
    "restart_optimizer_steps", "restart_processed_text_positions",
    "per_event_checkpoint_writes", "artifact_checkpoint_writes",
    "retained_checkpoint_artifacts", "live_recovery_checkpoints",
    "storage_temp_overlap_blocks", "recovery_reloads",
})
COUNTER_SUFFIXES = (
    "_count", "_calls", "_tokens", "_updates", "_seconds", "_forwards",
    "_positions", "_loads", "_steps", "_bytes", "_writes", "_requests",
    "_streams", "_artifacts", "_blocks", "_reloads",
)


def is_work_counter(name: Any) -> bool:
    """True when ``name`` is a declared physical-work counter."""
    return (isinstance(name, str)
            and (name in WORK_COUNTERS or name.endswith(COUNTER_SUFFIXES)))


def _require_counter_value(name: str, value: Any, check: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            check, f"{name} must be a nonnegative integer")
    return int(value)


def check_work_budget_v2(accounting: Mapping[str, Any],
                         budget: Mapping[str, Any], *,
                         arm: str | None = None,
                         check: str = "work_budget_v2") -> dict:
    """Fail closed on any physical-work counter above its declared ceiling.

    The counters are execution points, not requests: one teaching event is a
    training forward plus a backward plus an optimizer step, and one generated
    answer is a prefill forward plus one decode forward per emitted token.  A
    caller charges the counters before the work and re-checks after it, so work
    stops before a ceiling is crossed rather than after.

    ``arm`` additionally enforces ``budget["per_arm_limits"][arm]``.  An arm
    that declares no block is an error: per-arm work is never inferred.
    """
    require(isinstance(accounting, Mapping), check, "accounting must be an object")
    require(isinstance(budget, Mapping), check, "budget must be an object")
    observed: dict[str, int] = {}
    for name, raw in accounting.items():
        require(is_work_counter(name), check,
                f"unrecognized accounting counter {name!r}")
        observed[str(name)] = _require_counter_value(str(name), raw, check)
    ceilings = budget.get("physical_limits")
    require(isinstance(ceilings, Mapping), check,
            "budget.physical_limits is required")
    arm_ceilings: Mapping[str, Any] = {}
    if arm is not None:
        per_arm = budget.get("per_arm_limits")
        require(isinstance(per_arm, Mapping), check,
                "budget.per_arm_limits is required when an arm is checked")
        require(arm in per_arm, check, f"no declared per-arm budget for {arm!r}")
        arm_ceilings = per_arm[arm]
        require(isinstance(arm_ceilings, Mapping), check,
                f"per_arm_limits[{arm!r}] must be an object")

    def ceiling_for(name: str, block: Mapping[str, Any],
                    scope: str) -> int:
        key = f"max_{name}"
        require(key in block, check,
                f"no declared {scope} physical ceiling for {name!r}")
        return _require_counter_value(
            key, block[key], check)

    checked: dict[str, int] = {}
    for name, value in observed.items():
        ceiling = ceiling_for(name, ceilings, "run")
        require(value <= ceiling, check,
                f"{name}={value} exceeds max_{name}={ceiling}")
        checked[f"max_{name}"] = ceiling
        if arm is not None and f"max_{name}" in arm_ceilings:
            arm_ceiling = _require_counter_value(
                f"max_{name}", arm_ceilings[f"max_{name}"], check)
            require(value <= arm_ceiling, check,
                    f"{name}={value} exceeds arm {arm!r} max_{name}="
                    f"{arm_ceiling}")
    return {
        "format": WORK_BUDGET_V2_FORMAT,
        "arm": arm,
        "observed": dict(sorted(observed.items())),
        "ceilings": dict(sorted(checked.items())),
        "within_budget": True,
    }


def seal_record(body: Mapping[str, Any], *,
                digest_field: str = "receipt_sha256") -> dict:
    """Return a copy of ``body`` carrying its own canonical digest."""
    sealed = {key: value for key, value in body.items() if key != digest_field}
    sealed[digest_field] = sha256_json(sealed)
    return sealed


def verify_sealed_record(record: Any, *, digest_field: str = "receipt_sha256",
                        check: str = "seal") -> dict:
    """Fail closed unless ``record`` is a self-consistent sealed body."""
    require(isinstance(record, Mapping), check, "record must be an object")
    digest = record.get(digest_field)
    require(is_hex64(digest), check,
            f"{digest_field} must be a sealed sha256 digest")
    body = {key: value for key, value in record.items() if key != digest_field}
    require(sha256_json(body) == digest, check,
            f"{digest_field} does not match the record body")
    return dict(record)


def sha256_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    """Digest a JSONL-shaped row sequence without writing it to disk."""
    return sha256_json([dict(row) for row in rows])


def git_show_bytes(commit: str, path: str) -> bytes | None:
    """Return the committed bytes of ``path`` at ``commit``, or None."""
    import subprocess

    require(isinstance(commit, str) and len(commit) == 40
            and all(character in "0123456789abcdef" for character in commit),
            "commit_binding", "protocol_commit must be a full 40-hex git commit")
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{commit}:{path}"],
            check=False, capture_output=True, timeout=60)
    except (OSError, ValueError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def require_committed_bytes(commit: str, path: str, expected_sha256: str, *,
                            verifier: Callable[[str, str], bytes | None] | None = None,
                            check: str = "commit_binding") -> str:
    """Require the exact committed bytes of ``path`` at ``commit``.

    ``verifier`` lets CPU tests bind an in-memory commit without a repository;
    the default reads the real worktree history and never touches the network.
    """
    require(is_hex64(expected_sha256), check,
            f"expected digest for {path} is not a sealed sha256")
    read = verifier or git_show_bytes
    payload = read(commit, path)
    require(payload is not None, check,
            f"{path} is absent from protocol commit {commit}")
    require(sha256_bytes(payload) == expected_sha256, check,
            f"{path} bytes at {commit} differ from the sealed digest")
    return expected_sha256
