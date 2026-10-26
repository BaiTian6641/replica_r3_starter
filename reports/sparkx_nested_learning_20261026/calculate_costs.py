"""Analytical payload ledger only; no torch, model imports, or native execution."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/sparkx_2_6_4b_nested_learning_v1.json"
OUT = Path(__file__).with_name("analytical_costs.json")


def calculate():
    config = json.loads(CONFIG.read_text())
    shape = config["shared_shapes"]
    plastic = config["plastic_bank"]
    fp32 = shape["dynamic_parameter_bytes"]
    expert_params = plastic["matrices_per_expert"] * plastic["width"] * plastic["hidden_width"]
    all_params = expert_params * plastic["experts"]
    active_params = expert_params * plastic["active_write_experts_max"]
    out = {"kind": "analytical_payload_lower_bounds_not_measured_fit", "profiles": {}}
    for name, profile in config["profiles"].items():
        record = shape["record_elements"] * shape["record_value_bytes"] + shape["record_metadata_max_bytes"]
        live = {
            "E_tensors_and_metadata": record * profile["episodic_records"],
            "F": profile["fast_groups"] * profile["fast_experts_per_group"] * profile["fast_heads"] * profile["fast_head_width"] ** 2 * fp32,
            "C": (shape["workspace_slots"] + shape["event_slots"]) * shape["workspace_width"] * shape["workspace_element_bytes"],
            "P": all_params * fp32,
            "P_two_Adam_moments": 2 * all_params * fp32,
        }
        address = profile["address_matrices"] * shape["address_width"] ** 2 * fp32
        optional_live = {"A": address, "A_two_Adam_moments": 2 * address}
        # The previous active revision is already included in 'live'. Candidate
        # weights/moments and gradients are additional; no second whole-bank
        # snapshot is assumed. A second full snapshot must be added if used.
        staging = {
            "F_candidate": live["F"],
            "P_selected_candidate_weights": active_params * fp32,
            "P_selected_candidate_moments": 2 * active_params * fp32,
            "P_selected_gradients": active_params * fp32,
            "C_candidate": live["C"],
            "E_one_record_candidate": record,
        }
        optional_staging = {"A_candidate": address, "A_candidate_moments": 2 * address, "A_gradients": address}
        base = sum(live.values())
        adaptive = base + sum(optional_live.values())
        peak = adaptive + sum(staging.values()) + sum(optional_staging.values())
        replay = config["resource_candidate"]["replay_envelope_MiB"] * 2**20
        out["profiles"][name] = {
            "plastic_parameters": all_params,
            "active_update_parameters_max": active_params,
            "live_bytes": live,
            "optional_adaptive_address_live_bytes": optional_live,
            "staging_bytes": staging,
            "optional_adaptive_address_staging_bytes": optional_staging,
            "base_live_MiB": base / 2**20,
            "with_adaptive_address_live_MiB": adaptive / 2**20,
            "illustrated_peak_with_adaptive_address_MiB": peak / 2**20,
            "illustrated_peak_plus_replay_MiB": (peak + replay) / 2**20,
            "payload_below_256_MiB_envelope": peak < 256 * 2**20,
            "128_MiB_envelope_fits_live_Adam_state": adaptive <= 128 * 2**20,
        }
    # One scalar first-step AdamW example: a math illustration of the missing
    # adaptation Jacobian. Deliberately large epsilon, not a proposed model LR
    # or epsilon. Initial moments zero, weight decay zero, no clipping active.
    p0, target, future, lr, eps = 0.3, 0.8, 1.4, 0.4, 0.3
    def adapted(p):
        g = p - target
        return p - lr * g / (abs(g) + eps)
    def outer(p):
        return 0.5 * (adapted(p) - future) ** 2
    jac = 1 - lr * eps / (abs(p0 - target) + eps) ** 2
    exact = (adapted(p0) - future) * jac
    first_order = adapted(p0) - future
    h = 1e-6
    finite_difference = (outer(p0 + h) - outer(p0 - h)) / (2 * h)
    out["scalar_math_only"] = {
        "description": "one-step AdamW, zero moments/decay, epsilon .3 solely for illustration; no model or optimizer execution",
        "adapted_parameter": adapted(p0),
        "exact_gradient": exact,
        "finite_difference": finite_difference,
        "first_order_gradient": first_order,
        "exact_matches_finite_difference": abs(exact - finite_difference) < 1e-8,
        "first_order_is_not_exact": abs(exact - first_order) > 1e-3,
    }
    out["excluded"] = config["resource_candidate"]["omissions"] + [
        "optimizer scalar counters and metadata", "stable release weights and training state",
        "extra full snapshots or reference-model copies", "raw evidence and expanded journals",
    ]
    out["native_execution"] = False
    out["training_admitted"] = False
    return out


if __name__ == "__main__":
    result = calculate()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {k: v for k, v in p.items() if k.endswith("MiB")} for name, p in result["profiles"].items()}, indent=2))
