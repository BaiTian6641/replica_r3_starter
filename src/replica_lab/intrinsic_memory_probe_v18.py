"""Ridge probes with folds defined by original group identities."""
import torch

LABELS = ("red", "blue", "green", "gold")


def probe(features, answers, *, identities, shuffled=None, solver="dual"):
    if identities is None or len(identities) != len(answers) or len(features) != len(answers):
        raise ValueError("original probe identities must align with rows")
    if features.ndim != 2:
        raise ValueError("finite matrix of probe features required")
    x_all = features.detach().to(device="cpu").to(dtype=torch.float64)
    if not bool(x_all.isfinite().all()):
        raise ValueError("finite matrix of probe features required")
    groups, seen = {}, set()
    for index, identity in enumerate(identities):
        group = identity.get("group_id", identity.get("pair_index"))
        row_id = identity.get("world_id")
        if group is None or row_id is None or row_id in seen:
            raise ValueError("unique rows and original group identities required")
        seen.add(row_id)
        groups.setdefault(group, []).append(index)
    if len(groups) < 2:
        raise ValueError("at least two independent groups required")
    if any(answer not in LABELS for answer in answers):
        raise ValueError("filter unsupported answers before probing")
    if shuffled is not None and (len(shuffled) != len(answers)
            or sorted(shuffled) != list(range(len(answers)))):
        raise ValueError("shuffled-label control must be a permutation")
    one_hot = torch.eye(len(LABELS), dtype=x_all.dtype, device=x_all.device)[
        [LABELS.index(answer) for answer in answers]]
    if shuffled is not None:
        one_hot = one_hot[shuffled]
    predictions = [None] * len(answers)
    for group, held in groups.items():
        train = [i for i in range(len(answers)) if i not in held]
        center, prior = x_all[train].mean(0), one_hot[train].mean(0)
        x = x_all[train] - center
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        y = one_hot[train] - prior
        if solver == "dual":
            weights = x.T @ torch.linalg.solve(x @ x.T + torch.eye(len(train), device=x.device), y)
        elif solver == "primal":
            weights = torch.linalg.solve(x.T @ x + torch.eye(x.shape[1], device=x.device), x.T @ y)
        else:
            raise ValueError("unknown probe solver")
        for index in held:
            value = x_all[index] - center
            scores = value / value.norm().clamp_min(1e-12) @ weights + prior
            predicted, target = LABELS[int(scores.argmax())], LABELS[int(one_hot[index].argmax())]
            predictions[index] = {**identities[index], "actual": target,
                "expected_answer": answers[index], "predicted": predicted,
                "correct": predicted == target, "held_out_group": group,
                "training_groups": [g for g in groups if g != group]}
    assert all(row is not None for row in predictions)
    return predictions
