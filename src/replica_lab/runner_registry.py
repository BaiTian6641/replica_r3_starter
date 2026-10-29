"""Retired-runner launch quarantine, separate from historical replay."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/retired_runners.json"


class RetiredRunnerError(ValueError):
    """A historical runner was used for new native execution."""

    def __init__(self, entry: dict):
        self.entry = dict(entry)
        super().__init__(entry["reason"])

    def as_dict(self) -> dict:
        return {"check": "retired_runner", "detail": str(self),
                "replacement": self.entry["replacement"]}


def _relative_runner(path: str | Path) -> str:
    candidate = Path(path)
    try:
        resolved = (candidate.resolve() if candidate.is_absolute()
                    else (ROOT / candidate).resolve())
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def retirement_for(path: str | Path) -> dict | None:
    relative = _relative_runner(path)
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for entry in payload["entries"]:
        if entry["path"] == relative:
            return dict(entry)
    return None


def require_not_retired(path: str | Path, *,
                        historical_replay: bool = False) -> None:
    """Refuse new launches; explicit historical replay still verifies authorization."""
    entry = retirement_for(path)
    replay_allowed = (historical_replay
                      and entry is not None
                      and entry.get("historical_replay_allowed") is True)
    if (entry is not None and entry.get("new_native_launch_allowed") is False
            and not replay_allowed):
        raise RetiredRunnerError(entry)


__all__ = ["RetiredRunnerError", "require_not_retired", "retirement_for"]
