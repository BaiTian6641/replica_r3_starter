"""Self-contained CPU tests for the paired image-dependence diagnostic."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replica_lab.contracts import fingerprint  # noqa: E402
from replica_lab.vision_dependence import (  # noqa: E402
    FINALIZE_RECEIPT_ROOT, IMPLEMENTATION_FILES, MODEL_ROW_FIELDS,
    PRIMARY_READOUT, SPLIT_DESCRIPTOR_POOLS, TASKS,
    VisionDependenceError, assemble_dev_report, assemble_native_run_report,
    audit_dataset, build_protocol, clustered_bootstrap, derangement_indices,
    evaluate_records, finalize_evaluation, fit_readouts, fit_ridge,
    infer_task, load_config, load_feature_bundle, predict_primary,
    publish_native_artifacts, question_derangement, records_from_bundle,
    regional_pooling, require_fresh_output, validate_config,
    validate_native_run_report, verify_data_pins, verify_implementation_pins,
    validate_protocol, verify_vision_binding, world_clusters,
    write_feature_bundle, write_protocol,
)

BASE_CONFIG = json.loads(
    (ROOT / "configs/vision_dependence_v2.json").read_text(encoding="utf-8"))
TEMPLATE = json.loads(
    (ROOT / "configs/minicpm5_vision_v0_qualification_v3.json").read_text(
        encoding="utf-8"))


def _small_config(data_root: str = "data/vd2_test") -> dict:
    config = copy.deepcopy(BASE_CONFIG)
    config["seed"] = 101
    config["worlds_per_task"] = 8
    config["data_root"] = data_root
    return validate_config(config)


@pytest.fixture(scope="session")
def small_config() -> dict:
    return _small_config()


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory, small_config) -> Path:
    root = tmp_path_factory.mktemp("vd2_data") / "corpus"
    from replica_lab.vision_dependence import generate_dataset
    generate_dataset(small_config, root)
    return root


@pytest.fixture(scope="session")
def dataset(dataset_root):
    return audit_dataset(dataset_root, config=_small_config())


def _png_path(root: Path, asset_id: str) -> Path:
    return root / "assets" / f"{asset_id}.png"


# ---------------------------------------------------------------------------
# Data contract.


def test_pairs_share_question_and_fork_exactly_one_verified_field(dataset):
    scene = {item["asset_id"]: item["scene"] for item in dataset.scenes}
    cases = {case["case_id"]: case for case in dataset.cases}
    for pair in dataset.pairs:
        first = pair["sides"]["a"]
        second = pair["sides"]["b"]
        first_case = cases[first["asset_id"]]
        second_case = cases[second["asset_id"]]
        assert first_case["question"] == second_case["question"] == pair["question"]
        assert first_case["answer"] != second_case["answer"]
        assert pair["answers_differ"] is True
        first_scene, second_scene = scene[first["asset_id"]], scene[second["asset_id"]]
        assert (first_scene["width"], first_scene["height"]) == (
            second_scene["width"], second_scene["height"])
        assert first_scene["layout"]["template"] == second_scene["layout"]["template"]
        task = pair["task"]
        change = pair["decisive_change"]
        if task == "color":
            assert change["kind"] == "color_changed"
            left = {obj["id"]: obj for obj in first_scene["objects"]}
            right = {obj["id"]: obj for obj in second_scene["objects"]}
            assert set(left) == set(right)
            for object_id in left:
                for key in ("shape", "bbox", "center"):
                    assert left[object_id][key] == right[object_id][key]
                if object_id == change["object_id"]:
                    assert left[object_id]["color"] == change["from"]
                    assert right[object_id]["color"] == change["to"]
                else:
                    assert left[object_id]["color"] == right[object_id]["color"]
            assert {change["from"], change["to"]} == {
                first_case["answer"], second_case["answer"]}
            assert pair["combination"]["answers"] == pair["answers"] == {
                "a": first_case["answer"], "b": second_case["answer"]}
            for side, side_scene, side_case in (("a", first_scene, first_case),
                                                ("b", second_scene, second_case)):
                target = {obj["id"]: obj for obj in side_scene["objects"]}["t0"]
                assert pair["combination"]["answer_descriptors"][side] == [
                    target["color"], target["shape"]]
        elif task == "shape":
            assert change["kind"] == "shape_changed"
            left = {obj["id"]: obj for obj in first_scene["objects"]}
            right = {obj["id"]: obj for obj in second_scene["objects"]}
            assert set(left) == set(right)
            for object_id in left:
                assert left[object_id]["color"] == right[object_id]["color"]
                assert left[object_id]["center"] == right[object_id]["center"]
                if object_id == change["object_id"]:
                    assert left[object_id]["shape"] == change["from"]
                    assert right[object_id]["shape"] == change["to"]
                else:
                    assert left[object_id]["shape"] == right[object_id]["shape"]
        elif task == "count":
            assert change["kind"] in {"object_added", "object_removed"}
            before = {(obj["shape"], obj["color"], tuple(obj["center"]))
                      for obj in first_scene["objects"]}
            after = {(obj["shape"], obj["color"], tuple(obj["center"]))
                     for obj in second_scene["objects"]}
            assert len(before.symmetric_difference(after)) == 1
            assert abs(int(first_case["answer"])
                       - int(second_case["answer"])) == 1
        elif task == "presence":
            assert change["kind"] in {"object_added", "object_removed"}
            before = {(obj["shape"], obj["color"], tuple(obj["center"]))
                      for obj in first_scene["objects"]}
            after = {(obj["shape"], obj["color"], tuple(obj["center"]))
                     for obj in second_scene["objects"]}
            assert before != after
            assert len(before.symmetric_difference(after)) == 1
            assert len(before) + len(after) > 1
            assert (first_case["answer"], second_case["answer"]) in {
                ("yes", "no"), ("no", "yes")}
        elif task == "spatial":
            assert change["kind"] == "position_changed"
            left = {obj["id"]: obj for obj in first_scene["objects"]}
            right = {obj["id"]: obj for obj in second_scene["objects"]}
            assert set(left) == set(right)
            moved = []
            for object_id in left:
                assert left[object_id]["shape"] == right[object_id]["shape"]
                assert left[object_id]["color"] == right[object_id]["color"]
                if left[object_id]["center"] != right[object_id]["center"]:
                    moved.append(object_id)
            assert moved == [change["object_id"]]
            assert (first_case["answer"], second_case["answer"]) in {
                ("yes", "no"), ("no", "yes")}
        else:
            pytest.fail(f"unexpected task {task}")
        first_png = _png_path(dataset.root, first["asset_id"]).read_bytes()
        second_png = _png_path(dataset.root, second["asset_id"]).read_bytes()
        from replica_lab.visual_data import decode_rgb_png
        assert decode_rgb_png(first_png)[2] != decode_rgb_png(second_png)[2]


def test_labels_balanced_and_train_covers_every_evaluation_label(dataset):
    manifest = dataset.manifest
    for task in TASKS:
        train = set(manifest["label_coverage"][task]["train"])
        for split in ("dev", "final"):
            assert set(manifest["label_coverage"][task][split]) <= train
        for split in ("train", "dev", "final"):
            balance = manifest["label_balance"][task][split]
            assert balance["max_minus_min"] <= 1
            assert sum(balance["histogram"].values()) > 0
    for split in ("train", "dev", "final"):
        for task in TASKS:
            side_histogram = {"a": {}, "b": {}}
            for side in ("a", "b"):
                counts = {}
                for pair in dataset.pairs:
                    if pair["split"] != split or pair["task"] != task:
                        continue
                    label = pair["sides"][side]["answer"]
                    counts[label] = counts.get(label, 0) + 1
                side_histogram[side] = counts
            expected = set(manifest["label_coverage"][task][split])
            for side in ("a", "b"):
                counts = side_histogram[side]
                assert set(counts) == expected, (split, task, side)
                values = list(counts.values())
                assert max(values) - min(values) <= 1, (split, task, side)
    count_labels = manifest["label_balance"]["count"]["train"]["labels"]
    assert len(set(count_labels)) >= 3
    assert not (len(count_labels) == 1 and count_labels[0] == "4")
    assert set(count_labels) == {"2", "3", "4", "5"}
    for split in ("train", "dev", "final"):
        histogram = manifest["label_balance"]["presence"][split]["histogram"]
        assert histogram["yes"] == histogram["no"]
        spatial = manifest["label_balance"]["spatial"][split]["histogram"]
        assert spatial["yes"] == spatial["no"]
        sides = {"a": {"yes": 0, "no": 0}, "b": {"yes": 0, "no": 0}}
        for pair in dataset.pairs:
            if pair["split"] != split or pair["task"] != "presence":
                continue
            for side in ("a", "b"):
                sides[side][pair["sides"][side]["answer"]] += 1
        assert sides["a"] == sides["b"]
        assert sides["a"]["yes"] == sides["a"]["no"]


def test_worlds_and_combinations_are_disjoint(dataset):
    worlds = dataset.worlds
    by_split = {split: {world["world_id"] for world in worlds
                        if world["split"] == split}
                for split in ("train", "dev", "final")}
    assert by_split["train"].isdisjoint(by_split["dev"])
    assert by_split["train"].isdisjoint(by_split["final"])
    assert by_split["dev"].isdisjoint(by_split["final"])
    combos = {split: {fingerprint(world["combination"]) for world in worlds
                      if world["split"] == split}
              for split in by_split}
    assert combos["train"].isdisjoint(combos["dev"])
    assert combos["train"].isdisjoint(combos["final"])
    assert combos["dev"].isdisjoint(combos["final"])
    for world in worlds:
        assert len(world["pair_ids"]) == 1
        assert len(world["asset_ids"]) == 2
        pair = next(item for item in dataset.pairs
                    if item["pair_id"] == world["pair_ids"][0])
        assert pair["combination"] == world["combination"]
        assert pair["split"] == world["split"]
        assert pair["task"] == world["task"]
    for split in by_split:
        for task in TASKS:
            count = sum(1 for world in worlds
                        if world["split"] == split and world["task"] == task)
            assert 8 <= count <= 12
    assert len({asset["sha256"] for asset in dataset.manifest["assets"]}) \
        == len(dataset.manifest["assets"])
    assert len({asset["scene_sha256"] for asset in dataset.manifest["assets"]}) \
        == len(dataset.manifest["assets"])


def test_target_distractor_and_layout_combinations_are_held_out(dataset):
    pools = {split: {tuple(item) for item in
                     dataset.manifest["descriptor_pools"][split]}
             for split in ("train", "dev", "final")}
    assert pools["train"].isdisjoint(pools["dev"])
    assert pools["train"].isdisjoint(pools["final"])
    assert pools["dev"].isdisjoint(pools["final"])
    scenes = {item["asset_id"]: item["scene"] for item in dataset.scenes}
    used = {split: {(obj["color"], obj["shape"])
                    for asset in dataset.manifest["assets"]
                    if asset["split"] == split
                    for obj in scenes[asset["asset_id"]]["objects"]}
            for split in ("train", "dev", "final")}
    for split in used:
        assert used[split] <= pools[split]
    assert used["train"].isdisjoint(used["dev"])
    assert used["train"].isdisjoint(used["final"])
    assert used["dev"].isdisjoint(used["final"])
    target_sets = {task: {split: set() for split in ("train", "dev", "final")}
                   for task in TASKS}
    distractor_sets = {task: {split: set() for split in ("train", "dev", "final")}
                       for task in TASKS}
    for pair in dataset.pairs:
        for side in ("a", "b"):
            scene = scenes[pair["sides"][side]["asset_id"]]
            for obj in scene["objects"]:
                descriptor = (obj["color"], obj["shape"])
                target = target_sets[pair["task"]][pair["split"]]
                distractor = distractor_sets[pair["task"]][pair["split"]]
                if obj["role"] == "target":
                    target.add(descriptor)
                else:
                    distractor.add(descriptor)
    for task in TASKS:
        for kind in (target_sets, distractor_sets):
            assert kind[task]["train"].isdisjoint(kind[task]["dev"])
            assert kind[task]["train"].isdisjoint(kind[task]["final"])
            assert kind[task]["dev"].isdisjoint(kind[task]["final"])
        per_task_layouts = {split: {world["combination"]["layout"]
                                    for world in dataset.worlds
                                    if world["split"] == split
                                    and world["task"] == task}
                            for split in ("train", "dev", "final")}
        assert per_task_layouts["train"].isdisjoint(per_task_layouts["dev"])
        assert per_task_layouts["train"].isdisjoint(per_task_layouts["final"])
        assert per_task_layouts["dev"].isdisjoint(per_task_layouts["final"])
    compositions = {split: {fingerprint(pair["distractor_composition"])
                            for pair in dataset.pairs
                            if pair["split"] == split}
                    for split in ("train", "dev", "final")}
    assert compositions["train"].isdisjoint(compositions["dev"])
    assert compositions["train"].isdisjoint(compositions["final"])
    assert compositions["dev"].isdisjoint(compositions["final"])
    assert dataset.manifest["distractor_compositions"]["disjoint"] is True
    assert dataset.manifest["counts"]["worlds_per_pair"] == 1
    assert dataset.manifest["counts"]["pairs_per_world"] == 1


def test_model_rows_carry_no_label_or_task_fields(dataset):
    for row in dataset.rows:
        assert set(row) == MODEL_ROW_FIELDS
        for forbidden in ("answer", "label", "task", "check", "expected",
                          "combination", "scene_sha256"):
            assert forbidden not in row
        encoded = json.dumps(row)
        assert '"answer"' not in encoded
        assert '"label"' not in encoded
        assert '"task"' not in encoded
        case = dataset.case(row["case_id"])
        assert row["question"] == case["question"]
        assert row["image_sha256"] == dataset.asset(row["asset_id"])["sha256"]


def test_new_corpus_stays_out_of_the_v1_case_set(dataset):
    manifest = dataset.manifest
    assert manifest["provenance"]["excludes_v1_cases"] is True
    assert manifest["provenance"]["v1_overlap"] == 0
    v1_path = ROOT / "data/vision_preparation_v1/v1/verifier_cases.jsonl"
    if v1_path.is_file():
        v1_ids = {json.loads(line)["case_id"] for line in
                  v1_path.read_text(encoding="utf-8").splitlines() if line}
        assert {case["case_id"] for case in dataset.cases}.isdisjoint(v1_ids)
    assert all(not case["case_id"].startswith("v1-") for case in dataset.cases)
    assert all(asset["asset_id"].startswith("vd2-")
               for asset in dataset.manifest["assets"])


def test_single_singleton_merge_keeps_a_bijection_with_unique_donors():
    question_a = "How many red circle objects are in the image?"
    question_b = "How many blue square objects are in the image?"
    records = [{"question": question_a, "task": "count"}] + [
        {"question": question_b, "task": "count"} for _ in range(3)]
    order, stats = question_derangement(records, 3)
    assert sorted(int(value) for value in order) == list(range(4))
    assert len(set(int(value) for value in order)) == 4
    assert not np.any(order == np.arange(4))
    donors = [records[int(value)]["question"] for value in order]
    assert donors.count(question_a) == 1
    assert donors.count(question_b) == 3
    same_question = sum(
        records[int(donor)]["question"] == records[receiver]["question"]
        for receiver, donor in enumerate(order))
    assert stats["same_question_rows"] == same_question == 2
    assert stats["same_task_only_rows"] == 2
    assert stats["fixed_points"] == 0 and stats["cross_task_donors"] == 0


def test_deterministic_regeneration_and_tamper_rejection(tmp_path, small_config):
    from replica_lab.vision_dependence import generate_dataset
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_dataset(small_config, first)
    generate_dataset(small_config, second)
    first_files = {path.relative_to(first): path.read_bytes()
                   for path in sorted(first.rglob("*")) if path.is_file()}
    second_files = {path.relative_to(second): path.read_bytes()
                    for path in sorted(second.rglob("*")) if path.is_file()}
    assert first_files.keys() == second_files.keys()
    for relative, data in first_files.items():
        assert data == second_files[relative], f"non-deterministic file: {relative}"

    def copy(name):
        target = tmp_path / name
        shutil.copytree(first, target)
        return target

    asset_tampered = copy("asset_tampered")
    asset = sorted((asset_tampered / "assets").glob("*.png"))[0]
    asset.write_bytes(asset.read_bytes() + b"tamper")
    with pytest.raises(VisionDependenceError):
        audit_dataset(asset_tampered)

    rows_tampered = copy("rows_tampered")
    rows_path = rows_tampered / "model_rows.jsonl"
    rows_path.write_text(rows_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(VisionDependenceError):
        audit_dataset(rows_tampered)

    manifest_tampered = copy("manifest_tampered")
    manifest_path = manifest_tampered / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["counts"]["worlds"] += 1
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(VisionDependenceError):
        audit_dataset(manifest_tampered)


# ---------------------------------------------------------------------------
# Readouts.


def test_high_dimensional_dual_ridge_matches_primal_solution():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(18, 256))
    labels = np.array([0, 1, 2] * 6)
    targets = np.eye(3)[labels]
    fitted = fit_ridge(features, targets, penalty=1.0)
    assert fitted["dual"] is True
    assert fitted["dimension"] == 256 and fitted["samples"] == 18
    mean = fitted["mean"]
    scale = features.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    design = (features - mean) / scale
    centred = targets - targets.mean(axis=0, keepdims=True)
    primal = np.linalg.solve(design.T @ design + np.eye(256),
                             design.T @ centred)
    assert np.allclose(fitted["weights"], primal, atol=1e-8)
    from replica_lab.vision_dependence import ridge_scores
    assert ridge_scores(fitted, features).shape == (18, 3)
    assert np.isfinite(fitted["weights"]).all()


QUESTION = {
    "count": "How many red circle objects are in the image?",
    "presence": "Is there a red circle in the image?",
    "color": "What color is the circle?",
    "shape": "What shape is the red object?",
    "spatial": "Is the red circle to the left of the blue square?",
}
LABELS = {
    "count": ("2", "3", "4"),
    "presence": ("no", "yes"),
    "color": ("red", "blue", "green"),
    "shape": ("circle", "square", "triangle"),
    "spatial": ("no", "yes"),
}


def _synthetic_records():
    rng = np.random.default_rng(11)
    all_labels = sorted({label for values in LABELS.values() for label in values})
    index = {label: position for position, label in enumerate(all_labels)}
    records = []
    for split, worlds in (("train", 8), ("dev", 4)):
        for task in TASKS:
            for world in range(worlds):
                label = LABELS[task][world % len(LABELS[task])]
                for side in range(2):
                    indicator = np.zeros(len(all_labels))
                    indicator[index[label]] = 5.0
                    question = indicator + rng.normal(scale=0.05,
                                                     size=len(all_labels))
                    records.append({
                        "case_id": f"syn-{split}-{task}-{world}-{side}",
                        "world_id": f"syn-{split}-{task}-w{world}",
                        "split": split,
                        "task": task,
                        "label": label,
                        "question": QUESTION[task],
                        "pooled": rng.normal(size=8),
                        "regions": rng.normal(size=(4, 8)),
                        "question_embedding": question,
                    })
    return records


def _readout_config(resamples: int = 100) -> dict:
    readout = copy.deepcopy(BASE_CONFIG["readout"])
    readout["bootstrap"]["resamples"] = resamples
    return readout


def test_question_only_can_outperform_image_and_never_supports_by_default():
    records = _synthetic_records()
    evaluation = evaluate_records(records, readout=_readout_config(),
                                  split="dev", seed=101)
    question = evaluation["readouts"]["question_only"]["accuracy"]
    primary = evaluation["readouts"][PRIMARY_READOUT]["accuracy"]
    assert question > 0.95
    assert primary < question
    verdict = evaluation["decision"]["verdict"]
    assert evaluation["decision"]["supported"] is False
    assert verdict in {"failing_primary_has_no_incremental_image_signal",
                      "inconclusive_primary_gain_not_established"}
    assert verdict != "frozen_features_support_readout"
    gains = evaluation["gains"]
    assert gains["over_question_only"]["point"] < 0.0
    assert evaluation["shuffle"]["fixed_points"] == 0
    assert {record["world_id"] for record in records
            if record["split"] == "final"}.isdisjoint(
        evaluation["evaluation_worlds"])


def test_derangement_has_no_fixed_points_and_is_deterministic():
    for count in range(2, 41):
        for seed in (0, 1, 7919):
            order = derangement_indices(count, seed)
            assert sorted(int(value) for value in order) == list(range(count))
            assert not np.any(order == np.arange(count))
            assert np.array_equal(order, derangement_indices(count, seed))
    with pytest.raises(VisionDependenceError):
        derangement_indices(1, 0)


def test_clustered_bootstrap_resamples_complete_pair_worlds():
    records = []
    for world in range(4):
        for side in (0, 1):
            records.append({"world_id": f"w{world}", "task": "count",
                            "value": float(side)})
    clusters = world_clusters(records)
    assert len(clusters) == 4
    assert all(len(indices) == 2 for indices in clusters.values())
    values = np.array([record["value"] for record in records])
    metric = {"pair_mean": values}
    result = clustered_bootstrap(records, metric, resamples=500,
                                 confidence=0.95, seed=3)
    interval = result["metrics"]["pair_mean"]
    assert interval["point"] == pytest.approx(0.5)
    assert interval["lower"] == pytest.approx(0.5)
    assert interval["upper"] == pytest.approx(0.5)
    per_task = clustered_bootstrap(records, metric, resamples=500,
                                   confidence=0.95, seed=3, task="count")
    assert per_task["clusters"] == 4


def test_primary_routing_uses_question_text_only():
    signature = inspect.signature(predict_primary)
    assert set(signature.parameters) == {"primary", "questions", "regions"}
    for task, question in QUESTION.items():
        assert infer_task(question) == task
    assert infer_task("What is this?") is None
    records = _synthetic_records()
    fitted = fit_readouts(records, penalty=1.0)
    assert fitted["primary"]["routing"] == "question_text_only"
    eval_records = [record for record in records if record["split"] == "dev"]
    regions = np.stack([record["regions"] for record in eval_records])
    predictions = predict_primary(fitted["primary"],
                                  [record["question"] for record in eval_records],
                                  regions)
    assert len(predictions) == len(eval_records)
    assert all(isinstance(value, str) for value in predictions)


def test_regional_pooling_preserves_quadrant_layout():
    patches = np.zeros((36, 6), dtype=np.float32)
    for token in range(36):
        patches[token] = token
    pooled = regional_pooling(patches, (2, 2))
    assert pooled.shape == (4, 6)
    side = 6
    expected = []
    for rows in ((0, 3), (3, 6)):
        for columns in ((0, 3), (3, 6)):
            block = np.arange(36).reshape(side, side)[
                rows[0]:rows[1], columns[0]:columns[1]]
            expected.append(block.reshape(-1).mean())
    assert np.allclose(pooled[:, 0], expected)
    with pytest.raises(VisionDependenceError):
        regional_pooling(np.zeros((35, 6)), (2, 2))


def test_decision_requires_both_gains_and_strictly_positive_bounds():
    decision = copy.deepcopy(BASE_CONFIG["readout"]["decision"])

    def metrics(gain_q, gain_s, lower_q, lower_s):
        return {
            "gain_over_question_only": {"point": gain_q, "lower": lower_q,
                                         "upper": 1.0},
            "gain_over_shuffled_image": {"point": gain_s, "lower": lower_s,
                                          "upper": 1.0},
        }

    from replica_lab.vision_dependence import decide
    supported = decide(metrics(0.06, 0.07, 0.01, 0.02), decision)
    assert supported["supported"] is True
    assert supported["verdict"] == "image_dependence_supported_by_primary_readout"
    at_bound = decide(metrics(0.06, 0.06, 0.0, 0.01), decision)
    assert at_bound["supported"] is False
    assert at_bound["verdict"] == "inconclusive_primary_gain_not_established"
    small = decide(metrics(0.04, 0.06, 0.01, 0.02), decision)
    assert small["supported"] is False
    assert small["verdict"] == "inconclusive_primary_gain_not_established"
    negative = decide(metrics(-0.01, 0.06, -0.05, 0.01), decision)
    assert negative["supported"] is False
    assert negative["verdict"] == "failing_primary_has_no_incremental_image_signal"
    for verdict in (supported, at_bound, small, negative):
        assert verdict["verdict"] != "frozen_features_support_readout"
        assert verdict["default_positive_verdict"] is False


# ---------------------------------------------------------------------------
# One-shot finalizer.


def _feature_rows(dataset):
    return [{
        "case_id": row["case_id"],
        "split": dataset.case(row["case_id"])["split"],
        "world_id": dataset.case(row["case_id"])["world_id"],
        "image_sha256": row["image_sha256"],
        "question": row["question"],
    } for row in dataset.rows]


def _build_run_artifacts(base: Path, dataset, protocol, protocol_path: Path,
                         name: str, seed: int):
    from replica_lab.vision_dependence import generate_dataset  # noqa: F401
    run_dir = base / "runs" / name
    rng = np.random.default_rng(seed)
    count = len(dataset.rows)
    rows = _feature_rows(dataset)
    pins = write_feature_bundle(
        run_dir, rows=rows,
        image_pooled=rng.normal(size=(count, 8)),
        image_regions=rng.normal(size=(count, 4, 8)),
        question=rng.normal(size=(count, 12)),
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        data=protocol["data"])
    bundle = load_feature_bundle(
        run_dir, expected_index_sha256=pins["index"]["sha256"])
    records = records_from_bundle(dataset, bundle)
    evaluation = evaluate_records(records, readout=_readout_config(),
                                  split="dev", seed=protocol["seed"])
    predictions_path = run_dir / "dev_predictions.json"
    predictions_path.write_text(
        json.dumps({"rows": evaluation["predictions"]}, sort_keys=True) + "\n",
        encoding="utf-8")
    predictions_sha = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    report_pins = {
        "features": pins["features"], "index": pins["index"],
        "predictions": {"path": str(predictions_path),
                        "sha256": predictions_sha},
    }
    dev_report = assemble_dev_report(
        protocol=protocol, protocol_path=protocol_path,
        feature_pins=report_pins, predictions_sha256=predictions_sha,
        evaluation=evaluation,
        model={"revision": protocol["model"]["revision"],
               "lock_sha256": protocol["model"]["lock_sha256"],
               "vision_digest": protocol["vision_digest"]},
        frozen_audit={"test": "synthetic-cpu"},
        created_at="2026-10-25T00:00:00+00:00")
    feature_bytes = (Path(pins["features"]["path"]).stat().st_size
                     + Path(pins["index"]["path"]).stat().st_size)
    run_report = assemble_native_run_report(
        protocol=protocol, protocol_path=protocol_path, dev_report=dev_report,
        output=run_dir, feature_pins=report_pins, dev_evaluation=evaluation,
        model=dev_report["model"], frozen_audit={"test": "synthetic-cpu"},
        workload_declared=validate_protocol(protocol),
        elapsed_seconds=1.0, feature_bytes=feature_bytes,
        started_at_utc="2026-10-25T00:00:00+00:00",
        packages={}, extra={})
    publish_native_artifacts(run_dir, dev_report=dev_report,
                            run_report=run_report,
                            max_seconds=protocol["limits"]["max_seconds"],
                            max_feature_bytes=protocol["limits"]["max_feature_bytes"])
    dev_report_path = run_dir / "dev_report.json"
    return {"features": run_dir / "features.npz", "run_dir": run_dir,
            "dev_report": dev_report_path,
            "dev_hash": hashlib.sha256(dev_report_path.read_bytes()).hexdigest()}


@pytest.fixture(scope="session")
def finalize_workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("vd2_finalize") / "repo"
    for relative in IMPLEMENTATION_FILES:
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    config = _small_config("data/vd2")
    from replica_lab.vision_dependence import generate_dataset
    dataset = generate_dataset(config, root / config["data_root"])
    protocol = build_protocol(config, root=root, template=TEMPLATE)
    protocol_path = root / config["data_root"] / "protocol.json"
    write_protocol(protocol_path, protocol)
    first = _build_run_artifacts(root, dataset, protocol, protocol_path,
                                 "dev01", 5)
    second = _build_run_artifacts(root, dataset, protocol, protocol_path,
                                  "dev02", 6)
    return {"root": root, "protocol_path": protocol_path, "dataset": dataset,
            "protocol": protocol, "first": first, "second": second}


def test_finalizer_requires_exact_dev_hash_and_refuses_repeat(
        tmp_path, finalize_workspace, monkeypatch):
    workspace = finalize_workspace
    monkeypatch.chdir(workspace["root"])
    first = workspace["first"]
    second = workspace["second"]
    receipt_root = workspace["root"] / FINALIZE_RECEIPT_ROOT
    output = tmp_path / "final"
    with pytest.raises(VisionDependenceError):
        finalize_evaluation(
            protocol_path=workspace["protocol_path"],
            features_path=first["features"],
            dev_report_path=first["dev_report"],
            dev_report_sha256="0" * 64, output=output)
    assert not receipt_root.exists() or not any(receipt_root.iterdir())
    report = finalize_evaluation(
        protocol_path=workspace["protocol_path"],
        features_path=first["features"],
        dev_report_path=first["dev_report"],
        dev_report_sha256=first["dev_hash"], output=output)
    assert report["scores_final"] is True
    assert report["final_scored"] is True
    assert report["evaluated_once"] is True
    assert report["dev_report_sha256"] == first["dev_hash"]
    assert report["evaluation"]["split"] == "final"
    assert report["evaluation"]["primary_readout"] == PRIMARY_READOUT
    assert (output / "final_report.json").is_file()
    receipts = list(receipt_root.glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["dev_report_sha256"] == first["dev_hash"]
    assert receipt["protocol_sha256"] == hashlib.sha256(
        workspace["protocol_path"].read_bytes()).hexdigest()
    assert receipt["data_manifest_sha256"] == (
        workspace["protocol"]["data"]["manifest"]["sha256"])
    assert receipt["status"] == "completed"
    assert receipt["features_index_sha256"] == (
        json.loads(first["dev_report"].read_text())["features"]["index"]["sha256"])
    assert receipt["final_report_sha256"] == hashlib.sha256(
        (output / "final_report.json").read_bytes()).hexdigest()
    assert "completed_at_utc" in receipt
    # A second native output with its own valid dev report cannot reuse the
    # same protocol/data holdout.
    second_output = tmp_path / "final_second_dir"
    with pytest.raises(VisionDependenceError):
        finalize_evaluation(
            protocol_path=workspace["protocol_path"],
            features_path=second["features"],
            dev_report_path=second["dev_report"],
            dev_report_sha256=second["dev_hash"], output=second_output)
    assert not (second_output / "final_report.json").exists()
    # Nor can the same run be finalised into another output directory.
    again = tmp_path / "final_again"
    with pytest.raises(VisionDependenceError):
        finalize_evaluation(
            protocol_path=workspace["protocol_path"],
            features_path=first["features"],
            dev_report_path=first["dev_report"],
            dev_report_sha256=first["dev_hash"], output=again)
    assert not (again / "final_report.json").exists()


def test_protocol_rejects_tampered_data_and_implementation_pins(finalize_workspace):
    protocol = json.loads(Path(finalize_workspace["protocol_path"]).read_text())
    root = finalize_workspace["root"]
    data_tampered = copy.deepcopy(protocol)
    data_tampered["data"]["model_rows"]["sha256"] = "0" * 64
    with pytest.raises(VisionDependenceError):
        verify_data_pins(data_tampered, root=root)
    pins_tampered = copy.deepcopy(protocol)
    relative = sorted(pins_tampered["implementation_sha256"])[0]
    pins_tampered["implementation_sha256"][relative] = "0" * 64
    with pytest.raises(VisionDependenceError):
        verify_implementation_pins(pins_tampered, root)


def test_feature_bundle_rejects_value_tampering(tmp_path, finalize_workspace):
    source = finalize_workspace["first"]["run_dir"]
    target = tmp_path / "bundle"
    target.mkdir()
    shutil.copy2(source / "features.npz", target / "features.npz")
    shutil.copy2(source / "features_index.json", target / "features_index.json")
    with np.load(target / "features.npz", allow_pickle=False) as stored:
        arrays = {name: np.array(stored[name]) for name in stored.files}
    arrays["image_pooled"][0, 0] += 1.0
    np.savez(target / "features.npz", **arrays)
    with pytest.raises(VisionDependenceError):
        load_feature_bundle(target)

# ---------------------------------------------------------------------------
# Review regressions: committed 12-world audit, pools, shuffle, receipts,
# failed-run ordering, count shortcut, data-root, vision binding.


def _committed_dataset():
    root = ROOT / "data/vision_dependence_v2"
    assert root.is_dir(), "committed v2 artifacts are required"
    return audit_dataset(root, config=load_config(
        ROOT / "configs/vision_dependence_v2.json"))


def _count_total_labels(dataset):
    scenes = {item["asset_id"]: item["scene"] for item in dataset.scenes}
    mapping = {}
    for pair in dataset.pairs:
        if pair["task"] != "count":
            continue
        for side in ("a", "b"):
            asset_id = pair["sides"][side]["asset_id"]
            total = len(scenes[asset_id]["objects"])
            label = pair["sides"][side]["answer"]
            mapping.setdefault(total, set()).add(label)
    return mapping


def test_committed_twelve_world_artifacts_are_audited():
    dataset = _committed_dataset()
    counts = dataset.manifest["counts"]
    assert counts["worlds"] == 180
    assert counts["pairs"] == 180
    assert counts["assets"] == 360
    assert counts["model_rows"] == 360
    assert counts["worlds_per_pair"] == 1
    assert counts["pairs_per_world"] == 1
    for split in ("train", "dev", "final"):
        for task in TASKS:
            assert counts["worlds_by_task_split"][task][split] == 12
    pools = {split: {tuple(item) for item in
                     dataset.manifest["descriptor_pools"][split]}
             for split in ("train", "dev", "final")}
    assert pools["train"].isdisjoint(pools["dev"])
    assert pools["train"].isdisjoint(pools["final"])
    assert pools["dev"].isdisjoint(pools["final"])
    compositions = {split: set(
        dataset.manifest["distractor_compositions"]["keys"][split])
        for split in ("train", "dev", "final")}
    assert compositions["train"].isdisjoint(compositions["dev"])
    assert compositions["train"].isdisjoint(compositions["final"])
    assert compositions["dev"].isdisjoint(compositions["final"])
    mapping = _count_total_labels(dataset)
    assert any(len(labels) > 1 for labels in mapping.values()), mapping


def test_count_total_objects_does_not_determine_the_label():
    dataset = _committed_dataset()
    mapping = _count_total_labels(dataset)
    scenes = {item["asset_id"]: item["scene"] for item in dataset.scenes}
    row_counts = {}
    for pair in dataset.pairs:
        if pair["task"] != "count":
            continue
        for side in ("a", "b"):
            total = len(scenes[pair["sides"][side]["asset_id"]]["objects"])
            row_counts[total] = row_counts.get(total, 0) + 1
    # The full mapping is eliminated: some totals carry multiple labels.
    assert any(len(labels) > 1 for labels in mapping.values())
    assert not all(len(labels) == 1 for labels in mapping.values())
    # Totals 3 and 8 remain partially informative on 12/72 count images.
    assert mapping[3] == {"2"} and mapping[8] == {"5"}
    assert row_counts[3] + row_counts[8] == 12
    for split in ("train", "dev", "final"):
        split_mapping = {}
        for pair in dataset.pairs:
            if pair["split"] != split or pair["task"] != "count":
                continue
            for side in ("a", "b"):
                total = len(scenes[pair["sides"][side]["asset_id"]]["objects"])
                split_mapping.setdefault(total, set()).add(
                    pair["sides"][side]["answer"])
        assert any(len(labels) > 1 for labels in split_mapping.values())


def _assert_shuffle_compatible(records, seed=17):
    order, stats = question_derangement(records, seed)
    assert stats["fixed_points"] == 0
    assert stats["cross_task_donors"] == 0
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(record["question"], []).append(index)
    for receiver, donor_index in enumerate(order):
        donor = records[int(donor_index)]
        assert donor["task"] == records[receiver]["task"]
        if len(groups[records[receiver]["question"]]) >= 2:
            assert donor["question"] == records[receiver]["question"]
    return order, stats


def test_shuffle_donors_stay_within_task_and_question_where_possible():
    dataset = _committed_dataset()
    for split in ("dev", "final"):
        records = [{"question": case["question"], "task": case["task"]}
                   for case in dataset.cases if case["split"] == split]
        order, stats = _assert_shuffle_compatible(records)
        assert stats["same_question_rows"] == len(records)
        assert stats["same_task_only_rows"] == 0
        assert len(set(int(value) for value in order)) == len(records)
    # A singleton question still receives a same-task donor rather than a
    # cross-task one.
    generic = [
        {"question": "How many red circle objects are in the image?",
         "task": "count"},
        {"question": "How many blue square objects are in the image?",
         "task": "count"},
        {"question": "What color is the circle?", "task": "color"},
        {"question": "What color is the circle?", "task": "color"},
    ]
    order, stats = _assert_shuffle_compatible(generic)
    assert stats["same_task_only_rows"] == 2


def test_vision_binding_checks_both_specs_against_the_protocol(finalize_workspace):
    from types import SimpleNamespace
    protocol = finalize_workspace["protocol"]
    fields = dict(protocol["vision"])
    adapter = SimpleNamespace(**fields, digest=protocol["vision_digest"])
    service = SimpleNamespace(**fields, digest=protocol["vision_digest"])
    verify_vision_binding(protocol, adapter_spec=adapter, service_spec=service)
    bad_service = SimpleNamespace(**fields, digest="0" * 64)
    with pytest.raises(VisionDependenceError):
        verify_vision_binding(protocol, adapter_spec=adapter,
                              service_spec=bad_service)
    bad_adapter = SimpleNamespace(**{**fields, "patch_size": 32},
                                  digest=protocol["vision_digest"])
    with pytest.raises(VisionDependenceError):
        verify_vision_binding(protocol, adapter_spec=bad_adapter,
                              service_spec=service)


def test_custom_data_root_protocol_generation(finalize_workspace):
    from replica_lab.vision_dependence import generate_dataset
    root = finalize_workspace["root"]
    config = _small_config("custom/vd2")
    generate_dataset(config, root / "custom/vd2")
    protocol = build_protocol(config, root=root, template=TEMPLATE,
                              data_root="custom/vd2")
    assert protocol["data"]["root"] == "custom/vd2"
    verify_data_pins(protocol, root=root)
    assert protocol["config_sha256"] == fingerprint(config)


def test_repeated_native_output_is_refused(tmp_path):
    output = require_fresh_output(tmp_path / "native")
    (output / "partial.txt").write_text("x", encoding="utf-8")
    with pytest.raises(VisionDependenceError):
        require_fresh_output(output)


def _publish_fixture(workspace, tmp_path, name, elapsed=1.0, feature_bytes=1):
    from replica_lab.vision_dependence import _json_bytes
    first = workspace["first"]
    dev_report = json.loads(Path(first["dev_report"]).read_text())
    output = tmp_path / name
    run_report = assemble_native_run_report(
        protocol=workspace["protocol"],
        protocol_path=workspace["protocol_path"],
        dev_report=dev_report, output=output,
        feature_pins=dev_report["features"],
        dev_evaluation=dev_report["evaluation"],
        model=dev_report["model"], frozen_audit={},
        workload_declared=validate_protocol(workspace["protocol"]),
        elapsed_seconds=elapsed, feature_bytes=feature_bytes,
        started_at_utc="2026-10-25T00:00:00+00:00", packages={}, extra={})
    assert run_report["dev_report"]["sha256"] == hashlib.sha256(
        _json_bytes(dev_report)).hexdigest()
    return output, dev_report, run_report


def test_publish_native_artifacts_checks_before_publishing(finalize_workspace,
                                                          tmp_path):
    output, dev_report, run_report = _publish_fixture(
        finalize_workspace, tmp_path, "timeout", elapsed=99.0)
    with pytest.raises(VisionDependenceError):
        publish_native_artifacts(
            output, dev_report=dev_report, run_report=run_report,
            max_seconds=10, max_feature_bytes=1024)
    assert not (output / "run_report.json").exists()
    assert not (output / "dev_report.json").exists()
    output_big, dev_report_big, run_report_big = _publish_fixture(
        finalize_workspace, tmp_path, "too_big", feature_bytes=4096)
    with pytest.raises(VisionDependenceError):
        publish_native_artifacts(
            output_big, dev_report=dev_report_big, run_report=run_report_big,
            max_seconds=10, max_feature_bytes=1024)
    assert not (output_big / "run_report.json").exists()
    assert not (output_big / "dev_report.json").exists()
    output_ok, dev_report_ok, run_report_ok = _publish_fixture(
        finalize_workspace, tmp_path, "ok")
    published = publish_native_artifacts(
        output_ok, dev_report=dev_report_ok, run_report=run_report_ok,
        max_seconds=10, max_feature_bytes=1024)
    assert published["status"] == "completed"
    assert (output_ok / "run_report.json").is_file()
    assert (output_ok / "dev_report.json").is_file()


def test_failed_dev_write_leaves_run_report_without_dev_report(
        finalize_workspace, tmp_path, monkeypatch):
    import replica_lab.vision_dependence as module
    output, dev_report, run_report = _publish_fixture(
        finalize_workspace, tmp_path, "order")
    real_write = module._atomic_write

    def failing_write(path, data):
        if Path(path).name == "dev_report.json":
            raise OSError("simulated dev publication failure")
        return real_write(path, data)

    monkeypatch.setattr(module, "_atomic_write", failing_write)
    with pytest.raises(OSError):
        publish_native_artifacts(
            output, dev_report=dev_report, run_report=run_report,
            max_seconds=10, max_feature_bytes=1024)
    assert (output / "run_report.json").is_file()
    assert not (output / "dev_report.json").exists()


def test_finalizer_rejects_failed_or_missing_native_run_reports(
        finalize_workspace, tmp_path, monkeypatch):
    from replica_lab.vision_dependence import generate_dataset  # noqa: F401
    workspace = finalize_workspace
    monkeypatch.chdir(workspace["root"])
    dataset = workspace["dataset"]
    protocol = workspace["protocol"]
    protocol_path = workspace["protocol_path"]

    def build(name, seed):
        return _build_run_artifacts(workspace["root"], dataset, protocol,
                                    protocol_path, name, seed)

    failed = build("failed_status", 21)
    run_path = failed["run_dir"] / "run_report.json"
    run = json.loads(run_path.read_text())
    run["status"] = "failed"
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    output = tmp_path / "failed_final"
    with pytest.raises(VisionDependenceError):
        finalize_evaluation(
            protocol_path=protocol_path, features_path=failed["features"],
            dev_report_path=failed["dev_report"],
            dev_report_sha256=failed["dev_hash"], output=output)
    assert not (output / "final_report.json").exists()

    updates = build("nonzero_updates", 22)
    run_path = updates["run_dir"] / "run_report.json"
    run = json.loads(run_path.read_text())
    run["optimizer_updates"] = 1
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    output = tmp_path / "updates_final"
    with pytest.raises(VisionDependenceError):
        finalize_evaluation(
            protocol_path=protocol_path, features_path=updates["features"],
            dev_report_path=updates["dev_report"],
            dev_report_sha256=updates["dev_hash"], output=output)
    assert not (output / "final_report.json").exists()

    missing = build("missing_run", 23)
    (missing["run_dir"] / "run_report.json").unlink()
    output = tmp_path / "missing_final"
    with pytest.raises(VisionDependenceError):
        finalize_evaluation(
            protocol_path=protocol_path, features_path=missing["features"],
            dev_report_path=missing["dev_report"],
            dev_report_sha256=missing["dev_hash"], output=output)
    assert not (output / "final_report.json").exists()

    wrong_hash = build("wrong_dev_pin", 24)
    run_path = wrong_hash["run_dir"] / "run_report.json"
    run = json.loads(run_path.read_text())
    run["dev_report"]["sha256"] = "0" * 64
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    with pytest.raises(VisionDependenceError):
        validate_native_run_report(
            dev_report_path=wrong_hash["dev_report"],
            dev_report_sha256=wrong_hash["dev_hash"],
            protocol=protocol, protocol_path=protocol_path,
            feature_pins=json.loads(
                wrong_hash["dev_report"].read_text())["features"])

def _isolated_workspace(tmp_path, name="repo"):
    root = tmp_path / name
    for relative in IMPLEMENTATION_FILES:
        source = ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    from replica_lab.vision_dependence import generate_dataset
    config = _small_config("data/vd2")
    dataset = generate_dataset(config, root / config["data_root"])
    protocol = build_protocol(config, root=root, template=TEMPLATE)
    protocol_path = root / config["data_root"] / "protocol.json"
    write_protocol(protocol_path, protocol)
    return {"root": root, "config": config, "dataset": dataset,
            "protocol": protocol, "protocol_path": protocol_path}


def _set_status(run):
    run["status"] = "failed"


def _set_elapsed(run):
    run["elapsed_seconds"] = 10 ** 9


def _set_feature_bytes(run):
    run["feature_bytes"] = 10 ** 9


def _set_data(run):
    run["data"]["counts"]["worlds"] += 1


def _set_device(run):
    run["device"] = "cpu"


def _set_downloads(run):
    run["downloads"] = 1


def _set_fallback(run):
    run["fallback_disabled"] = False


def _set_updates(run):
    run["optimizer_updates"] = 1


def _set_workload(run):
    run["workload_actual"]["optimizer_updates"] = 1


def _set_dev_hash(run):
    run["dev_report"]["sha256"] = "0" * 64


def _set_final_split(run):
    run["scored_splits"] = ["dev", "final"]


def _set_feature_collection(run):
    run["feature_collection_splits"] = ["train", "dev"]


RUN_REPORT_MUTATIONS = [
    ("status", _set_status, "did not complete"),
    ("elapsed", _set_elapsed, "exceeded max_seconds"),
    ("feature_bytes", _set_feature_bytes, "exceeded max_feature_bytes"),
    ("data", _set_data, "data pins differ"),
    ("device", _set_device, "device differs"),
    ("downloads", _set_downloads, "declares downloads"),
    ("fallback", _set_fallback, "fallback policy"),
    ("updates", _set_updates, "zero updates"),
    ("workload", _set_workload, "workload differs"),
    ("dev_hash", _set_dev_hash, "exact dev report"),
    ("scored_splits", _set_final_split, "scored splits"),
    ("feature_collection", _set_feature_collection,
     "feature collection splits"),
]


@pytest.mark.parametrize("name,mutate,message", RUN_REPORT_MUTATIONS,
                         ids=[item[0] for item in RUN_REPORT_MUTATIONS])
def test_finalizer_rechecks_every_run_invariant_in_isolation(
        tmp_path, name, mutate, message, monkeypatch):
    workspace = _isolated_workspace(tmp_path, name=f"repo_{name}")
    monkeypatch.chdir(workspace["root"])
    artifacts = _build_run_artifacts(
        workspace["root"], workspace["dataset"], workspace["protocol"],
        workspace["protocol_path"], f"run_{name}", 31)
    run_path = artifacts["run_dir"] / "run_report.json"
    run = json.loads(run_path.read_text())
    mutate(run)
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    output = tmp_path / f"final_{name}"
    with pytest.raises(VisionDependenceError, match=message):
        finalize_evaluation(
            protocol_path=workspace["protocol_path"],
            features_path=artifacts["features"],
            dev_report_path=artifacts["dev_report"],
            dev_report_sha256=artifacts["dev_hash"], output=output)
    assert not (output / "final_report.json").exists()
    receipts = workspace["root"] / FINALIZE_RECEIPT_ROOT
    assert not receipts.exists() or not any(receipts.iterdir())


def test_finalizer_requires_run_report_in_isolation(tmp_path, monkeypatch):
    workspace = _isolated_workspace(tmp_path, name="repo_missing")
    monkeypatch.chdir(workspace["root"])
    artifacts = _build_run_artifacts(
        workspace["root"], workspace["dataset"], workspace["protocol"],
        workspace["protocol_path"], "run_missing", 32)
    (artifacts["run_dir"] / "run_report.json").unlink()
    output = tmp_path / "final_missing"
    with pytest.raises(VisionDependenceError, match="run_report.json is required"):
        finalize_evaluation(
            protocol_path=workspace["protocol_path"],
            features_path=artifacts["features"],
            dev_report_path=artifacts["dev_report"],
            dev_report_sha256=artifacts["dev_hash"], output=output)
    assert not (output / "final_report.json").exists()


def test_receipt_is_completed_only_after_a_final_report(tmp_path, monkeypatch):
    import replica_lab.vision_dependence as module
    workspace = _isolated_workspace(tmp_path, name="repo_crash")
    monkeypatch.chdir(workspace["root"])
    artifacts = _build_run_artifacts(
        workspace["root"], workspace["dataset"], workspace["protocol"],
        workspace["protocol_path"], "run_crash", 33)
    output = tmp_path / "final_crash"
    real_write = module._write_exact

    def fail_final_report(path, data):
        if Path(path).name == "final_report.json":
            raise OSError("simulated final report write failure")
        return real_write(path, data)

    monkeypatch.setattr(module, "_write_exact", fail_final_report)
    with pytest.raises(OSError):
        finalize_evaluation(
            protocol_path=workspace["protocol_path"],
            features_path=artifacts["features"],
            dev_report_path=artifacts["dev_report"],
            dev_report_sha256=artifacts["dev_hash"], output=output)
    assert not (output / "final_report.json").exists()
    receipts = list((workspace["root"] / FINALIZE_RECEIPT_ROOT).glob("*.json"))
    assert len(receipts) == 1
    reserved = json.loads(receipts[0].read_text())
    assert reserved["status"] == "reserved_before_scoring"
    assert reserved["features_index_sha256"]
    assert "final_report_sha256" not in reserved
    monkeypatch.setattr(module, "_write_exact", real_write)
    with pytest.raises(VisionDependenceError, match="Refusing repeat evaluation"):
        finalize_evaluation(
            protocol_path=workspace["protocol_path"],
            features_path=artifacts["features"],
            dev_report_path=artifacts["dev_report"],
            dev_report_sha256=artifacts["dev_hash"], output=output)
    assert not (output / "final_report.json").exists()

