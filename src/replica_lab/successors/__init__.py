"""Separately versioned Replica successor contracts (isolated CPU engineering).

This package is intentionally outside the contextual-memory v12 modules and the
sealed v3-v11 runtime bytes. It contains:

- :mod:`replica_lab.successors.formation` - completed-interaction formation (S2a)
- :mod:`replica_lab.successors.chunks` - chunk-start/barrier formation (S2b)
- :mod:`replica_lab.successors.tiers` - hot/warm/cold validity and accounting
- :mod:`replica_lab.successors.adapters` - adapters to the existing core/task/
  identity interfaces
- :mod:`replica_lab.successors.study` - isolated CPU write-enabled/disabled
  later-answer comparison
- :mod:`replica_lab.successors.protocols` - later behavioral protocol documents

Every durable format carries a distinct ``replica.successors.*.v1`` schema. No
module here reads or writes ``src/replica_lab/runtime_pilot.py``. Native memory,
persistent-P updates, capacity expansion, nested adaptation and RSI remain
gated; these modules prove CPU contracts only.
"""

__all__ = [
    "formation",
    "chunks",
    "tiers",
    "adapters",
    "study",
    "protocols",
]
