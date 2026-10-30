"""Incrementally replayed v13 work ledger; v12 journals remain readable and unchanged.

Own durable appends update verified cached state in O(1) for reserve/finish.
External appends use prefix authentication plus the complete v12 replay path.
Replacement, truncation, torn tails and uncertain work remain fail-closed.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import os

from .intrinsic_memory_budget_v12 import (
    BudgetError, COUNTERS, WorkLedger, _amounts, _canonical)

FORMAT = "replica.intrinsic_memory.reservations.incremental.v13"


class IncrementalWorkLedger(WorkLedger):
    """A drop-in reader/writer for the v12 append-only journal format."""

    def _install_cache(self, doc):
        stat = self.path.stat()
        self._cache = {
            "key": (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns),
            "doc": doc,
            "sha256": hashlib.sha256(self.path.read_bytes()),
        }

    def _cached_read(self):
        stat = self.path.stat()
        cache = self._cache
        if cache is not None:
            device, inode, size, mtime = cache["key"]
            if (stat.st_dev, stat.st_ino) != (device, inode):
                raise BudgetError("reservation journal was replaced")
            if stat.st_size < size:
                raise BudgetError("reservation journal was truncated")
            if stat.st_size == size:
                if stat.st_mtime_ns != mtime:
                    raise BudgetError("reservation journal was modified in place")
                return cache["doc"]
            with self.path.open("rb") as stream:
                prefix = stream.read(size)
                if len(prefix) != size:
                    raise BudgetError("reservation journal prefix is unavailable")
            if hashlib.sha256(prefix).digest() != cache["sha256"].digest():
                raise BudgetError("reservation journal prefix differs")
            doc = self._read()  # external append: complete verified replay
            self._install_cache(doc)
            return doc
        doc = self._read()
        self._install_cache(doc)
        return doc

    def snapshot(self, *, cached=False):
        with self._shared():
            doc = self._cached_read()  # always enforce the cached high-water boundary
            if cached:
                return doc
            fresh = self._read()
            self._install_cache(fresh)
            return fresh

    @staticmethod
    def _apply(doc, record):
        """Apply one already durable record with the same checks as v12 replay."""
        kind = record.get("t")
        plan, stages, events = doc["plan"], doc["stages"], doc["events"]
        if kind == "enter":
            stage = record["stage"]
            if stage not in plan or stage in stages:
                raise BudgetError("stage missing or already consumed")
            if any(value["status"] == "active" for value in stages.values()):
                raise BudgetError("two stages active in one journal")
            doc["reserved_wall_seconds"] += plan[stage]["wall_seconds"]
            stages[stage] = {"status": "active", "started": record["started"],
                "deadline": record["deadline"], "deadline_unix": record["deadline_unix"],
                "reserved": {key: 0 for key in COUNTERS}}
        elif kind == "reserve":
            stage = record["stage"]
            state = stages.get(stage)
            amounts = _amounts(record["reserved"], where="incremental replay")
            if state is None or state["status"] != "active":
                raise BudgetError("reservation outside an active stage")
            if any(state["reserved"][key] + value > plan[stage][key]
                   for key, value in amounts.items()):
                raise BudgetError("stage work budget exceeded in journal")
            if record["id"] != len(events):
                raise BudgetError("reservation id order differs")
            for key, value in amounts.items():
                state["reserved"][key] += value
                doc["reserved"][key] += value
            events.append({"id": record["id"], "stage": stage,
                "operation": record["operation"], "reserved": amounts,
                "status": "reserved", "started": record["started"], "pid": record["pid"]})
        elif kind == "finish":
            identifier = record.get("id")
            event = (events[identifier] if type(identifier) is int
                     and 0 <= identifier < len(events) else None)
            if event is None or event["status"] != "reserved":
                raise BudgetError("terminal record does not match a reservation")
            if record["status"] not in ("completed", "failed"):
                raise BudgetError("invalid terminal status")
            actual = _amounts(record.get("actual") or {}, where="actual")
            if any(key not in event["reserved"] or value > event["reserved"][key]
                   for key, value in actual.items()):
                raise BudgetError("actual work exceeds reservation")
            event.update(status=record["status"], actual=actual,
                         detail=record.get("detail"),
                         elapsed_seconds=record.get("elapsed_seconds"))
        elif kind == "leave":
            state = stages.get(record["stage"])
            if state is None or state["status"] != "active":
                raise BudgetError("stage already closed")
            if any(event["stage"] == record["stage"] and event["status"] == "reserved"
                   for event in events):
                raise BudgetError("stage has uncertain work")
            state.update(status=record["status"],
                         elapsed_seconds=record.get("elapsed_seconds"))
        elif kind == "recover":
            for identifier in record["ids"]:
                event = (events[identifier] if type(identifier) is int
                         and 0 <= identifier < len(events) else None)
                if event is None or event["status"] != "reserved":
                    raise BudgetError("recovery record does not match a reservation")
                event.update(status="failed", actual={},
                    detail={"error": "BudgetError",
                            "message": "process terminated before terminal record; charged"},
                    elapsed_seconds=None)
        elif kind == "torn":
            if (type(record.get("bytes")) is not int or record["bytes"] <= 0
                    or not isinstance(record.get("sha256"), str)):
                raise BudgetError("invalid torn-tail record")
            doc["torn_records"].append({"bytes": record["bytes"],
                                        "sha256": record["sha256"]})
        else:
            raise BudgetError("unknown or non-append journal record")

    @contextmanager
    def _transaction(self):
        lock = self.path.with_suffix(".lock")
        if lock.is_symlink() or self.path.is_symlink():
            raise BudgetError("substituted ledger")
        with lock.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                doc = self._cached_read()

                def write(record):
                    line = (_canonical(record) + "\n").encode()
                    try:
                        written = self._write(record)
                        self._apply(doc, record)
                        cache = self._cache
                        cache["sha256"].update(line)
                        stat = self.path.stat()
                        doc["journal"].update(
                            appended_bytes=doc["journal"]["appended_bytes"] + written,
                            append_calls=doc["journal"]["append_calls"] + 1,
                            torn_tail=False)
                        cache["key"] = (stat.st_dev, stat.st_ino,
                                        stat.st_size, stat.st_mtime_ns)
                        return written
                    except BaseException:
                        # The append may be partial or the cached state uncertain.
                        self._cache = None
                        raise

                yield doc, write
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
