"""Explicit separation between execution completion and learning success."""
from __future__ import annotations

from collections.abc import Iterable


def learning_success_evidence(*, declared_updates: int, execution_complete: bool,
                              optimizer_steps: int,
                              changed_trainable_names: Iterable[str],
                              restored_parent: bool = False,
                              retry_optimizer_steps: int = 0,
                              no_op_optimizer_steps: int = 0) -> dict:
    """Report physical learning work separately from evaluation state.

    A positive-update arm succeeds only when every declared event and every
    explicitly accounted retry completed and at least one declared trainable
    tensor actually changed.  Restoring the parent afterward does not erase
    that training evidence.  Zero-update controls are not applicable; a no-op
    or zero-learning-rate optimizer is a failed positive-update arm, not a
    successful execution alias.
    """
    declared = int(declared_updates)
    steps = int(optimizer_steps)
    retries = int(retry_optimizer_steps)
    no_ops = int(no_op_optimizer_steps)
    raw_names = list(changed_trainable_names)
    if any(not isinstance(name, str) or not name for name in raw_names):
        raise ValueError("changed trainable names must be nonempty strings")
    names = sorted(set(raw_names))
    valid_counts = (declared >= 0 and steps >= 0 and retries >= 0
                    and no_ops >= 0 and no_ops <= retries <= steps)
    if declared == 0:
        passed = False
        status = ("not_applicable_zero_update_control"
                  if execution_complete and steps == retries == no_ops == 0
                  else "failed")
    else:
        # Retried successful events are already included in declared updates;
        # only completed no-op calls add physical steps beyond that count.
        passed = (valid_counts and bool(execution_complete)
                  and steps == declared + no_ops and bool(names))
        status = "passed" if passed else "failed"
    return {
        "criterion": (
            "declared_steps_plus_noops_with_retries_and_changed_trainable_tensor"),
        "declared_updates": declared,
        "execution_complete": bool(execution_complete),
        "optimizer_steps": steps,
        "retry_optimizer_steps": retries,
        "no_op_optimizer_steps": no_ops,
        "changed_trainable_names": names,
        "learning_success": passed,
        "restored_parent": bool(restored_parent),
        "status": status,
    }


__all__ = ["learning_success_evidence"]
