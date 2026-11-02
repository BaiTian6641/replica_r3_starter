"""Regression for fold-local probe targets and original pair identities."""
import torch

from probe_slot_decodability_v15 import probe as primal_probe
from analyze_spark_reader_query_v15 import probe as dual_probe


def test_shuffled_targets_and_original_pair_identity_match_in_both_probes():
    answers = ["red", "blue"] * 4
    identities = [
        {"pair_index": pair, "side": side, "world_id": f"world-{pair}-{side}"}
        for pair in range(4) for side in (0, 1)
    ]
    features = torch.tensor([
        [2.0, 0.0, 1.0, 0.0], [0.0, 2.0, 1.0, 0.0],
        [2.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0, 1.0],
        [2.0, 0.0, 1.0, 1.0], [0.0, 2.0, 1.0, 1.0],
        [2.0, 0.0, 2.0, 1.0], [0.0, 2.0, 2.0, 1.0],
    ])
    shuffle = [1, 0, 3, 2, 5, 4, 7, 6]
    for probe in (primal_probe, dual_probe):
        baseline = probe(features, answers, identities=identities)
        assert sum(row["correct"] for row in baseline) == 8
        assert [row["world_id"] for row in baseline] == [row["world_id"] for row in identities]
        shuffled = probe(features, answers, shuffled=shuffle, identities=identities)
        # Every label is swapped. Score against the permuted held-out labels.
        assert sum(row["correct"] for row in shuffled) == 8
        assert [row["actual"] for row in shuffled] == ["blue", "red"] * 4
        assert [row["expected_answer"] for row in shuffled] == answers
        assert [row["pair_index"] for row in shuffled] == [0, 0, 1, 1, 2, 2, 3, 3]
