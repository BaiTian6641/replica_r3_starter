"""Explicit native memory/core architecture; the legacy adapter remains reproducible."""
from .contracts import (PortSpec, CoreSpec, CoreState, EvidencePin, DecisionFrame,
                        RequestContext, MemoryRead, PendingWrite, TaskGraph, ComponentManifest,
                        SourceSpan, SenseFrame)
from .minicpm5 import MiniCPM5Adapter
