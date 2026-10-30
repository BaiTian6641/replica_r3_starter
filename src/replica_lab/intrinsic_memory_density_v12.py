"""Invocation-based density/cost instrumentation for the separately versioned v12 candidate.

Two responsibilities live here, both new and independent of the sealed v3-v11
runners and of the five v12 producer modules:

* ``SiteInvocationProfiler`` binds every *site dispatch* to the one physical
  reader group that actually ran.  Hooks are registered exactly once per unique
  module object.  A matched layout with two canonical groups invoked from 36
  sites therefore has two hook pairs, not 72.  Forward and recompute passes get
  distinct identities; recomputation is charged as computation but cannot create
  retrieval, admission, snapshot or durable-write effects.
* ``DensityProducerBoundary`` is a narrow, versioned adapter over the reviewed
  v12 producer.  It validates the documented public surface, the frozen-source
  feature policy, placement, canonical group aliasing and exact optimizer
  ownership without importing or assuming the producer's private internals.

Neither class measures native performance.  A single-forward tiny-fixture
duration is an engineering cost sample, not a Spark latency, memory or density
qualification.  Physical allocator counters are reported only when the device
backend exposes them; deterministic activation bytes are always reported.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading
import time

import torch

IMPLEMENTATION = "replica.intrinsic_memory.density.v12"
PROFILER_FORMAT = IMPLEMENTATION + ".profiler"
PRODUCER_BOUNDARY_FORMAT = IMPLEMENTATION + ".producer_boundary"
LEDGER_FORMAT = IMPLEMENTATION + ".cost_ledger"
PRODUCER_API_VERSION = 1
PLACEMENT = "block_input_query_to_post_block_residual"
WIDTH = 256
SLOTS = 8
PHASES = ("forward", "recompute")
COMPARISONS = ("fixed_per_site", "matched_unique_parameters")

# Side-effect counters that a recomputation must not change.  Source tokenizer or
# frozen-source forwards may legitimately recur; retrieval/writes/copies may not.
READ_WRITE_KEYS = (
    "retrieval_calls", "admission_calls", "correction_calls", "snapshot_copies",
    "snapshot_bytes", "durable_write_bytes", "durable_write_ns",
    "source_revalidations", "value_slots_read",
)

PRODUCER_METHODS = (
    "forward", "training_values", "training_from_stored", "snapshot", "validate",
    "admit", "read", "correct", "save", "load", "close", "modules",
    "assert_optimizer_ownership",
)
PRODUCER_ATTRIBUTES = (
    "spec", "reader", "adapter", "encoder", "optimizer", "cost", "source_identity",
)
PRODUCER_API = {
    "version": PRODUCER_API_VERSION,
    "required_methods": list(PRODUCER_METHODS),
    "required_attributes": list(PRODUCER_ATTRIBUTES),
    "feature_provider": ["spec", "provenance", "release"],
    "encoder": ["release", "provider"],
    "optimizer": "NativeFusedAdamW",
    "placement": PLACEMENT,
}


class DensityError(RuntimeError):
    """Density instrumentation, boundary or accounting contract failed."""


class ProducerBoundaryViolation(DensityError):
    """The reviewed producer does not expose the documented v12 density surface."""


def tensor_bytes(value):
    """Deterministic tensor payload bytes; auxiliary metadata is ignored."""
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    return 0


def _device_allocator_bytes(device=None):
    """Current allocation on the actual reader device, or None on CPU.

    A CPU fixture must not report a GPU allocator merely because the installed
    torch build exposes one.  Never sampled or estimated.
    """
    kind = getattr(device, "type", None)
    if kind == "cuda" and torch.cuda.is_available():
        return int(torch.cuda.memory_allocated(device))
    if kind == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and mps.is_available():
            try:
                return int(mps.current_allocated_memory())
            except (AttributeError, RuntimeError):
                return None
    return None


def _int_cost(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _integer_ledger(mapping):
    result = {}
    for name, value in sorted(mapping.items()):
        converted = _int_cost(value)
        if converted is not None:
            result[name] = converted
    return result


def cost_snapshot(path):
    """Capture the producer/memory cost ledgers without invoking any operation."""
    producer = getattr(path, "producer", path)
    memory_cost = getattr(producer, "cost", {})
    if not isinstance(memory_cost, dict):
        raise DensityError("producer cost ledger must be a mapping")
    provider = getattr(getattr(producer, "encoder", None), "provider", None)
    provider_cost = getattr(provider, "cost", None)
    if provider_cost is not None and not isinstance(provider_cost, dict):
        raise DensityError("source-feature cost ledger must be a mapping")
    return {
        "format": LEDGER_FORMAT,
        "memory": _integer_ledger(memory_cost),
        "source": (_integer_ledger(provider_cost)
                   if provider_cost is not None else {}),
    }


def ledger_delta(before, after):
    """Per-ledger differences; missing keys are treated as zero."""
    if not (isinstance(before, dict) and isinstance(after, dict)):
        raise DensityError("two cost snapshots are required")
    result = {"format": LEDGER_FORMAT, "memory": {}, "source": {}}
    for ledger in ("memory", "source"):
        keys = set(before.get(ledger, {})) | set(after.get(ledger, {}))
        result[ledger] = {key: int(after.get(ledger, {}).get(key, 0))
                          - int(before.get(ledger, {}).get(key, 0))
                          for key in sorted(keys)}
    return result


def assert_no_read_write_effects(before, after, *, keys=READ_WRITE_KEYS):
    """Raise if a declared recomputation introduced a memory side effect."""
    delta = ledger_delta(before, after)
    changed = {key: delta["memory"].get(key, 0) for key in keys
               if delta["memory"].get(key, 0) != 0}
    # Source tokenization/forward recomputation is legitimate work; it may not
    # run backwards and is reported separately from durable read/write effects.
    for key in ("requests", "forwards", "cache_misses"):
        value = delta["source"].get(key, 0)
        if value < 0:
            raise DensityError("source-feature counter moved backwards: " + key)
        if value:
            changed["source." + key] = value
    if changed:
        raise DensityError("recomputation created retrieval/write/copy effects: "
                           + ", ".join(sorted(changed)))
    return delta


def physical_groups(reader):
    """Map every declared site to exactly one unique physical reader module."""
    spec = getattr(reader, "spec", None)
    sites = getattr(spec, "sites", None)
    if (not isinstance(sites, tuple) or not sites
            or tuple(sorted(set(sites))) != sites):
        raise DensityError("ordered unique reader sites are required")
    ports = getattr(reader, "ports", None)
    if ports is None:
        raise DensityError("reader must expose site ports")
    mapping, modules = {}, {}
    for site in sites:
        module = ports[str(site)]
        if not isinstance(module, torch.nn.Module):
            raise DensityError(f"site {site} does not resolve to a reader module")
        mapping[site] = id(module)
        modules[id(module)] = module
    if len(modules) < 1:
        raise DensityError("no physical reader group")
    names = {}
    for name, module in reader.named_modules():
        names.setdefault(id(module), name)
    return mapping, modules, names


def parameter_audit(path):
    """Exact intended/optimizer ownership inventory for one producer path."""
    producer = getattr(path, "producer", path)
    try:
        modules = producer.modules()
        groups = producer.optimizer.param_groups
    except AttributeError as error:
        raise DensityError("producer lacks modules()/optimizer groups") from error
    if not isinstance(modules, dict) or not modules:
        raise DensityError("producer modules() must return a nonempty mapping")
    intended, intended_names = [], {}
    for name, module in modules.items():
        for key, value in module.named_parameters():
            intended.append(value)
            intended_names[id(value)] = name + "." + key
    owned = [value for group in groups for value in group["params"]]
    base = list(producer.adapter.base.parameters())
    base_ids = {id(value) for value in base}
    owned_ids = {id(value) for value in owned}
    intended_ids = {id(value) for value in intended}
    reader = producer.reader
    reader_parameters = list(reader.parameters())
    reader_ids = {id(value) for value in reader_parameters}
    mapping, physical, physical_names = physical_groups(reader)
    state_tensors = len(reader.state_dict())
    physical_state_tensors = sum(len(module.state_dict()) for module in physical.values())
    reader_groups = {"groups": len(physical),
                     "site_to_group": {str(site): mapping[site] for site in mapping},
                     "canonical_owner_names": sorted(
                         {physical_names.get(gid) for gid in physical
                          if physical_names.get(gid) is not None}),
                     "state_tensors": state_tensors,
                     "physical_state_tensors": physical_state_tensors,
                     "alias_duplicated_state_tensors": state_tensors - physical_state_tensors}
    audit = {
        "intended_parameter_tensors": len(intended),
        "intended_unique_parameter_objects": len(intended_ids),
        "duplicate_intended_registration": len(intended) - len(intended_ids),
        "intended_parameters": sum(value.numel() for value in intended),
        "intended_requires_grad_tensors": sum(bool(value.requires_grad) for value in intended),
        "optimizer_parameter_tensors": len(owned),
        "optimizer_parameters": sum(value.numel() for value in owned),
        "optimizer_groups": len(groups),
        "duplicate_ownership": len(owned) - len(owned_ids),
        "missing_ownership": len(intended_ids - owned_ids),
        "extra_ownership": len(owned_ids - intended_ids),
        "base_parameter_tensors": len(base),
        "base_parameters": sum(value.numel() for value in base),
        "base_parameters_trainable": sum(bool(value.requires_grad) for value in base),
        "base_parameters_optimizer_owned": sum(id(value) in owned_ids for value in base),
        "reader_parameter_tensors": len(reader_parameters),
        "reader_parameters": sum(value.numel() for value in reader_parameters),
        "reader_unique_parameter_objects": len(reader_ids),
        "reader_groups": reader_groups,
        "intended_names": [intended_names[id(value)] for value in intended],
        "owner_group_names": [group.get("group_name") for group in groups],
    }
    audit["ownership_exact"] = bool(
        audit["duplicate_ownership"] == 0 and audit["missing_ownership"] == 0
        and audit["extra_ownership"] == 0
        and audit["duplicate_intended_registration"] == 0
        and audit["base_parameters_optimizer_owned"] == 0
        and audit["base_parameters_trainable"] == 0
        and audit["reader_unique_parameter_objects"] == len(reader_ids)
        and audit["reader_groups"]["alias_duplicated_state_tensors"] == 0)
    audit["all_intended_trainable"] = (
        audit["intended_requires_grad_tensors"] == audit["intended_parameter_tensors"])
    return audit


def _group_name(reader, group_id):
    for name, module in reader.named_modules():
        if id(module) == group_id:
            return name
    return None


class SiteInvocationProfiler:
    """One hook pair per physical reader group, one event per site dispatch.

    Construction validates that every layout site resolves to a module, then
    registers hooks only for the distinct module objects.  ``instrument`` wraps a
    ``LayerMemorySnapshot`` in a proxy whose ``delta`` supplies site/pass identity
    while the hooks measure the real group computation.  Only one thread may use
    an instance: attribution is a small deterministic stack, not global state.
    """

    def __init__(self, path_or_reader, *, expected_groups=None,
                 cache_bytes_source=None):
        self.reader = getattr(path_or_reader, "reader", path_or_reader)
        self._cache_source = cache_bytes_source
        self._provider = getattr(getattr(path_or_reader, "encoder", None), "provider", None)
        self._owner_thread = threading.get_ident()
        self.site_to_group, self.groups, self.group_names = physical_groups(self.reader)
        self.sites = tuple(self.reader.spec.sites)
        if expected_groups is not None and len(self.groups) != expected_groups:
            raise DensityError(
                f"expected {expected_groups} physical groups, found {len(self.groups)}")
        self._hooks = []
        self._events = []
        self._site_counts = {str(site): 0 for site in self.sites}
        self._group_counts = {}
        self._phase_counts = {phase: 0 for phase in PHASES}
        self._forward_ids = set()
        self._recompute_of = {}
        self._snapshot_by_forward = {}
        self._active = None
        self._open = None
        self._last_event = None
        self._unattributed = 0
        self._duplicate_refusals = 0
        self._registration_counts = {}
        self._backward = {"calls": 0, "processed_positions": 0}
        self._register()

    # -- hooks -----------------------------------------------------------
    def _register(self):
        for group_id, module in self.groups.items():
            if getattr(module, "_density_profiler_owner", None) is not None:
                self._duplicate_refusals += 1
                raise DensityError("reader group already has density hooks; duplicate registration refused")
            pre = module.register_forward_pre_hook(self._pre)
            post = module.register_forward_hook(self._post)
            module._density_profiler_owner = self
            self._hooks.extend((pre, post))
            self._registration_counts[group_id] = {
                "group_name": self.group_names.get(group_id),
                "pre_hooks": 1, "post_hooks": 1,
            }

    def _check_thread(self):
        if threading.get_ident() != self._owner_thread:
            raise DensityError("one profiler instance is single-threaded")

    def _pre(self, module, inputs):
        self._check_thread()
        if self._active is None:
            self._unattributed += 1
            raise DensityError("physical group ran without a site dispatch")
        site, expected_group = self._active
        if id(module) != expected_group:
            raise DensityError("site dispatch resolved to a different physical group")
        if self._open is not None:
            raise DensityError("nested reader invocation is not a declared site dispatch")
        parameter = next(module.parameters(), None)
        device = parameter.device if parameter is not None else None
        event = {
            "sequence": len(self._events) + 1,
            "site": site,
            "group": id(module),
            "group_name": self.group_names.get(id(module)),
            "phase": self._pass_phase,
            "forward_id": self._pass_forward,
            "recompute_of": self._pass_recompute_of,
            "sequence_id": self._pass_sequence,
            "chunk_id": self._pass_chunk,
            "input_bytes": tensor_bytes(inputs),
            "availability_nonzero": (bool(inputs[2].any())
                                     if len(inputs) > 2 and torch.is_tensor(inputs[2]) else None),
            "allocator_before": _device_allocator_bytes(device),
            "cache_before": self._cache_bytes(),
            "_device": device,
            "_start_ns": time.perf_counter_ns(),
        }
        self._open = event

    def _post(self, module, inputs, output):
        if self._open is None or self._active is None:
            raise DensityError("reader group returned without a site dispatch")
        event = self._open
        event["group_wall_ns"] = time.perf_counter_ns() - event.pop("_start_ns")
        event["output_bytes"] = tensor_bytes(output)
        event["output_requires_grad"] = bool(torch.is_tensor(output) and output.requires_grad)
        device = event.pop("_device", None)
        event["device"] = str(device) if device is not None else None
        event["allocator_after"] = _device_allocator_bytes(device)
        event["cache_after"] = self._cache_bytes()
        self._events.append(event)
        self._group_counts[str(event["group"])] = self._group_counts.get(str(event["group"]), 0) + 1
        self._phase_counts[event["phase"]] += 1
        self._last_event = event
        self._open = None

    def _cache_bytes(self):
        if self._cache_source is not None:
            value = self._cache_source()
            if _int_cost(value) is not None:
                return _int_cost(value)
        provider_cost = getattr(self._provider, "cost", None)
        if isinstance(provider_cost, dict):
            value = provider_cost.get("cache_bytes")
            if _int_cost(value) is not None:
                return _int_cost(value)
        return None

    # -- pass/site identity ---------------------------------------------
    @contextmanager
    def _dispatch(self, site, expected_group):
        if self._active is not None:
            raise DensityError("concurrent or nested site dispatch")
        self._active = (site, expected_group)
        try:
            yield
        finally:
            self._active = None
            self._open = None

    def instrument(self, snapshot, *, forward_id=None, phase="forward",
                   recompute_of=None, sequence_id=0, chunk_id=0,
                   active_sites=None):
        """Wrap one shared snapshot for one pass without copying its tensors."""
        self._check_thread()
        if phase not in PHASES:
            raise DensityError("undeclared profiling phase")
        if recompute_of is not None and phase != "recompute":
            raise DensityError("recompute_of is only valid for a recompute pass")
        if phase == "recompute":
            if recompute_of not in self._forward_ids:
                raise DensityError("recompute references an unknown forward identity")
            original = self._snapshot_by_forward[str(recompute_of)]
            if id(snapshot) != original:
                raise DensityError("recompute must reuse the forward's one snapshot object")
            if forward_id is None:
                forward_id = f"recompute-{len(self._recompute_of) + 1}"
            if str(forward_id) in self._recompute_of:
                raise DensityError("duplicate recompute identity")
        else:
            if forward_id is None:
                forward_id = f"forward-{len(self._forward_ids) + 1}"
            if str(forward_id) in self._forward_ids:
                raise DensityError("duplicate forward identity")
            self._forward_ids.add(str(forward_id))
            self._snapshot_by_forward[str(forward_id)] = id(snapshot)
        sites = snapshot.spec.sites if active_sites is None else tuple(active_sites)
        if (not isinstance(sites, tuple) or not sites
                or tuple(sorted(set(sites))) != sites
                or not set(sites) <= set(self.sites)):
            raise DensityError("invalid active-site mask")
        if phase == "recompute":
            self._recompute_of[str(forward_id)] = str(recompute_of)
        return _ProfiledSnapshot(self, snapshot, sites, forward_id=str(forward_id),
                                 phase=phase,
                                 recompute_of=(str(recompute_of)
                                               if recompute_of is not None else None),
                                 sequence_id=sequence_id, chunk_id=chunk_id)

    def forward(self, path, ids, mask, batch, *, scope, active_sites=None,
                enabled=True, forward_id=None, phase="forward", recompute_of=None,
                snapshot=None, sequence_id=0, chunk_id=0):
        """Snapshot once, profile every dispatch, then run the frozen backbone."""
        if not enabled or batch is None or active_sites == ():
            return path.forward(ids, mask, batch, scope=scope,
                                enabled=enabled, active_sites=active_sites)
        if snapshot is None:
            snapshot = path.snapshot(ids, batch, scope=scope, active_sites=active_sites)
        proxy = self.instrument(snapshot, forward_id=forward_id, phase=phase,
                                recompute_of=recompute_of, sequence_id=sequence_id,
                                chunk_id=chunk_id, active_sites=active_sites)
        return path.adapter.forward_text(
            ids, mask, use_projector=False, use_ports=False, use_spine=False,
            intrinsic_memory=proxy)

    def record_backward(self, *, calls=1, processed_positions=0):
        """Charge an observed backward/update, never infer one from reader count."""
        if (type(calls) is not int or calls < 0
                or type(processed_positions) is not int or processed_positions < 0):
            raise DensityError("invalid backward work charge")
        self._backward["calls"] += calls
        self._backward["processed_positions"] += processed_positions
        return dict(self._backward)

    # -- reporting -------------------------------------------------------
    def summary(self):
        self._check_thread()
        hooks_now = {str(gid): (2 if getattr(module, "_density_profiler_owner", None) is self
                                else 0) for gid, module in self.groups.items()}
        if any(count != 2 for count in hooks_now.values()):
            raise DensityError("a physical group does not have exactly one pre/post hook pair")
        if self._unattributed:
            raise DensityError("unattributed group invocations were observed")
        if self._active is not None or self._open is not None:
            raise DensityError("summary requested during an open dispatch")
        group_invocations = sum(self._group_counts.values())
        site_invocations = sum(self._site_counts.values())
        if site_invocations != group_invocations:
            raise DensityError("site dispatches and physical group invocations differ")
        allocator = [event for event in self._events
                     if event["allocator_before"] is not None]
        cache = [value for event in self._events
                 for value in (event["cache_before"], event["cache_after"])
                 if value is not None]
        events = [dict(event) for event in self._events]
        return {
            "format": PROFILER_FORMAT,
            "implementation": IMPLEMENTATION,
            "sites": list(self.sites),
            "site_to_group": {str(site): group for site, group in self.site_to_group.items()},
            "group_names": {str(gid): name for gid, name in self.group_names.items()
                            if gid in self.groups},
            "physical_groups": len(self.groups),
            "hook_registrations": [
                {**value, "group": str(gid), "hooks_now": hooks_now[str(gid)]}
                for gid, value in sorted(self._registration_counts.items())],
            "callbacks_per_physical_group": 2,
            "duplicate_registration_refusals": self._duplicate_refusals,
            "total_site_invocations": site_invocations,
            "site_invocations": {site: count for site, count in self._site_counts.items()},
            "group_invocations": {group: count for group, count in self._group_counts.items()},
            "phase_site_invocations": {phase: sum(
                event["phase"] == phase for event in self._events) for phase in PHASES},
            "forward_ids": sorted(self._forward_ids),
            "recompute_passes": dict(sorted(self._recompute_of.items())),
            "timing": {
                "unit": "nanoseconds_monotonic",
                "group_wall_ns_total": sum(event["group_wall_ns"] for event in self._events),
                "group_wall_ns_by_phase": {
                    phase: sum(event["group_wall_ns"] for event in self._events
                               if event["phase"] == phase) for phase in PHASES},
                "dispatch_wall_ns_total": sum(event.get("dispatch_wall_ns", 0)
                                               for event in self._events),
                "single_forward_fixture_only": True,
            },
            "backward_work": {**self._backward,
                              "caller_measured": True},
            "activation_bytes": {
                "input_total": sum(event["input_bytes"] for event in self._events),
                "output_total": sum(event["output_bytes"] for event in self._events),
                "by_phase": {phase: {
                    "input": sum(event["input_bytes"] for event in self._events
                                 if event["phase"] == phase),
                    "output": sum(event["output_bytes"] for event in self._events
                                  if event["phase"] == phase),
                } for phase in PHASES},
            },
            "allocator": {
                "available": bool(allocator),
                "backend": ("cuda_or_mps" if allocator else "unavailable_on_cpu"),
                "max_current_allocated_bytes": (max(
                    event["allocator_after"] for event in allocator)
                    if allocator else None),
            },
            "cache_bytes": {
                "available": bool(cache),
                "max_observed": max(cache) if cache else None,
                "min_observed": min(cache) if cache else None,
            },
            "memory_side_effects_not_created_by_profiler": {
                "retrieval_calls": 0, "admission_calls": 0,
                "snapshot_copies": 0, "durable_write_bytes": 0,
            },
            "events": events,
        }

    def close(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        for module in self.groups.values():
            if getattr(module, "_density_profiler_owner", None) is self:
                del module._density_profiler_owner


class _ProfiledSnapshot:
    """Read-only snapshot facade that attributes each delta to one site."""

    def __init__(self, profiler, snapshot, sites, *, forward_id, phase,
                 recompute_of, sequence_id, chunk_id):
        self._profiler = profiler
        self._snapshot = snapshot
        self.spec = snapshot.spec
        self.values = snapshot.values
        self.availability = snapshot.availability
        self.active_sites = sites
        self.forward_id = forward_id
        self.phase = phase
        self.recompute_of = recompute_of
        self.sequence_id = sequence_id
        self.chunk_id = chunk_id

    def delta(self, site, hidden):
        profiler = self._profiler
        if site not in self.spec.sites:
            raise DensityError("disabled intrinsic site")
        expected = profiler.site_to_group.get(site)
        if expected is None:
            raise DensityError("site has no physical group")
        before = len(profiler._events)
        profiler._pass_phase = self.phase
        profiler._pass_forward = self.forward_id
        profiler._pass_recompute_of = self.recompute_of
        profiler._pass_sequence = self.sequence_id
        profiler._pass_chunk = self.chunk_id
        start = time.perf_counter_ns()
        with profiler._dispatch(site, expected):
            output = self._snapshot.delta(site, hidden)
        elapsed = time.perf_counter_ns() - start
        if len(profiler._events) != before + 1:
            raise DensityError("site dispatch did not produce exactly one group invocation")
        event = profiler._events[-1]
        if event["site"] != site or event["group"] != expected:
            raise DensityError("group invocation was attributed to another site")
        event["dispatch_wall_ns"] = elapsed
        profiler._site_counts[str(site)] += 1
        return output


class DensityProducerBoundary:
    """Versioned, duck-typed boundary over the reviewed v12 producer.

    No v12 module is imported here.  Binding refuses a missing/renamed surface,
    a different placement, a non-canonical feature contract, non-aliased matched
    groups, or duplicate/missing optimizer ownership.  It does not repair any of
    those conditions.
    """

    def __init__(self, producer, *, comparison, expected_groups=None):
        if comparison not in COMPARISONS:
            raise ProducerBoundaryViolation("undeclared density comparison")
        missing = [name for name in PRODUCER_METHODS
                   if not callable(getattr(producer, name, None))]
        missing += [name for name in PRODUCER_ATTRIBUTES
                    if not hasattr(producer, name)]
        if missing:
            raise ProducerBoundaryViolation(
                "reviewed producer is missing density API names: " + ", ".join(sorted(missing)))
        if not isinstance(producer.cost, dict):
            raise ProducerBoundaryViolation("producer cost ledger must be a mapping")
        if not isinstance(producer.source_identity, str) or not producer.source_identity:
            raise ProducerBoundaryViolation("explicit source identity is required")
        spec = producer.spec
        if (getattr(spec, "placement", None) != PLACEMENT
                or getattr(spec, "width", None) != WIDTH
                or getattr(spec, "slots", None) != SLOTS):
            raise ProducerBoundaryViolation("declared v12 placement/width/slots differ")
        provider = getattr(producer.encoder, "provider", None)
        feature_spec = getattr(provider, "spec", None)
        identity = getattr(feature_spec, "identity", None)
        if not callable(identity):
            raise ProducerBoundaryViolation("frozen-source feature spec is missing")
        feature = identity()
        if (feature.get("input_policy") != "canonical_span_only_no_template_no_special_tokens"
                or feature.get("output_policy") != "last_decoder_hidden_after_final_norm"
                or feature.get("original_updates") != 0
                or feature.get("hidden_size") != spec.hidden_size
                or getattr(feature_spec, "max_tokens", 257) > 256
                or getattr(feature_spec, "cache_bytes", 32 * 1024**2 + 1) > 32 * 1024**2
                or getattr(feature_spec, "cache_entries", 65) > 64):
            raise ProducerBoundaryViolation("frozen-source feature contract differs")
        if getattr(provider, "adapter", None) is not producer.adapter:
            raise ProducerBoundaryViolation("feature provider is bound to another adapter")
        mapping, groups, names = physical_groups(producer.reader)
        wanted = len(spec.sites) if comparison == "fixed_per_site" else 2
        if expected_groups is not None and expected_groups != wanted:
            raise ProducerBoundaryViolation("expected group count cannot weaken comparison contract")
        if len(groups) != wanted:
            raise ProducerBoundaryViolation(
                f"{comparison} requires {wanted} physical groups, found {len(groups)}")
        if comparison == "matched_unique_parameters":
            self._validate_matched_aliases(producer.reader, mapping, groups, names)
        self.audit = parameter_audit(producer)
        if not self.audit["ownership_exact"]:
            raise ProducerBoundaryViolation("exact single optimizer ownership failed")
        self.producer = producer
        self.comparison = comparison
        self.spec = spec
        self.site_to_group = mapping
        self.group_names = {gid: names.get(gid) for gid in groups}
        self.feature_contract = feature

    @staticmethod
    def _validate_matched_aliases(reader, mapping, groups, names):
        canonical = getattr(reader, "groups", None)
        if canonical is None or len(canonical) != 2:
            raise ProducerBoundaryViolation("matched reader must register exactly two canonical groups")
        index = {}
        for name, module in canonical.items():
            index[id(module)] = int(name)
        if set(index) != set(groups):
            raise ProducerBoundaryViolation("matched aliases do not resolve to the canonical groups")
        for site, group_id in mapping.items():
            if index[group_id] != (site // 4) % 2:
                raise ProducerBoundaryViolation("matched site-to-group mapping differs")
        if len(reader.state_dict()) != sum(len(module.state_dict())
                                           for module in canonical.values()):
            raise ProducerBoundaryViolation("site aliases duplicated reader state tensors")

    def identity(self):
        provider = self.producer.encoder.provider
        release = getattr(self.producer.encoder, "release", None)
        producer_release = release() if callable(release) else None
        return {
            "format": PRODUCER_BOUNDARY_FORMAT,
            "api_version": PRODUCER_API_VERSION,
            "comparison": self.comparison,
            "spec": self.spec.identity(),
            "source_identity": self.producer.source_identity,
            "execution_identity": getattr(self.producer, "execution_identity", None),
            "objective_identity": getattr(self.producer, "objective_identity", None),
            "feature_contract": self.feature_contract,
            "provider_provenance": callable(getattr(provider, "provenance", None)),
            "producer_release": producer_release,
            "optimizer_ownership": self.audit,
            "site_to_group": {str(site): group for site, group in self.site_to_group.items()},
            "group_names": {str(gid): name for gid, name in self.group_names.items()},
        }

    # Explicit delegation only: no attribute proxying over private internals.
    def forward(self, *args, **kwargs):
        return self.producer.forward(*args, **kwargs)

    def training_values(self, *args, **kwargs):
        return self.producer.training_values(*args, **kwargs)

    def training_from_stored(self, *args, **kwargs):
        return self.producer.training_from_stored(*args, **kwargs)

    def snapshot(self, *args, **kwargs):
        return self.producer.snapshot(*args, **kwargs)

    def validate(self, *args, **kwargs):
        return self.producer.validate(*args, **kwargs)

    def admit(self, *args, **kwargs):
        return self.producer.admit(*args, **kwargs)

    def read(self, *args, **kwargs):
        return self.producer.read(*args, **kwargs)

    def correct(self, *args, **kwargs):
        return self.producer.correct(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self.producer.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        return self.producer.load(*args, **kwargs)

    def close(self):
        return self.producer.close()

    def profiler(self, **kwargs):
        return SiteInvocationProfiler(self.producer, **kwargs)

    def cost_snapshot(self):
        return cost_snapshot(self.producer)

    def __repr__(self):
        return (f"DensityProducerBoundary(comparison={self.comparison!r}, "
                f"groups={len(self.group_names)}, sites={len(self.spec.sites)})")
