"""Explicit native memory-port architecture; the legacy adapter remains reproducible."""
from .contracts import (PortSpec, EvidencePin, DecisionFrame, RequestContext, MemoryRead,
                        PendingWrite, TaskGraph, ComponentManifest, SourceSpan, SenseFrame)
from .minicpm5 import MiniCPM5Adapter
