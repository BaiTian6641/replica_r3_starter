"""Analytical proposal accounting; does not instantiate or benchmark a model."""
from pathlib import Path
import json

MIB = 2**20
D, M, H, K, V = 2560, 256, 4, 64, 64
PORTS = 2

def profile(rows, row_width, slots, experts):
    table_parameters = 2 * 4 * rows * row_width
    values = {
        'static_table_parameters': table_parameters,
        'static_table_bf16_bytes': table_parameters * 2,
        'static_table_fp32_weights_gradients_adam_moments_bytes': table_parameters * 16,
        'episodic_keys_and_values_bf16_bytes': slots * (128 + 256) * 2,
        'episodic_token_payload_int32_bytes': slots * 16 * 4,
        'episodic_metadata_allowance_bytes': slots * 64,
        'shared_fast_state_fp32_bytes': experts * H * K * V * 4,
        'generated_rank8_two_port_state_fp32_bytes': PORTS * 8 * (D + D) * 4,
        'workspace_fp32_bytes': 16 * 256 * 4,
        'temporal_cache_bf16_bytes': 64 * 256 * 2,
        'appraisal_fp32_bytes': 8 * 4,
    }
    state_fields = [k for k in values if k.endswith('_bytes') and not k.startswith('static_')]
    values['live_state_payload_bytes'] = sum(values[k] for k in state_fields)
    values['live_state_payload_mib'] = values['live_state_payload_bytes'] / MIB
    return values

block_macs = D*(16+2*4)*256 + D*16 + 16*256*D + 3*D*10240
baseline_linear_flops = 2*(36*block_macs + 131072*D)
one_port_weights = 2*D*M + 6*M*M
read_macs_per_port = 2*D*M + M*M + H*K*V
result = {
    'format': 'replica.dynamic_engram.analytical_costs.v1',
    'status': 'proposal_estimates_not_measured_fit_or_latency',
    'profiles': {
        'compact_pilot': profile(16384, 32, 4096, 1),
        'expanded_candidate': profile(65536, 64, 16384, 4),
    },
    'one_expert_factorized_fast_branch': {
        'definition': 'Per port: D-to-M stem, six M-to-M projections for q/k/v/erase/write/decay, M-to-D output; excludes biases, normalization, router and output-gate parameters',
        'parameters_per_port': one_port_weights,
        'parameters_two_private_ports_shared_state': PORTS * one_port_weights,
        'decode_flops_two_ports': 2 * PORTS * read_macs_per_port,
        'four_expert_dense_reads_shared_query_flops_two_ports': 2*PORTS*(2*D*M+M*M+4*H*K*V),
        'separate_expert_projections_not_in_four_expert_read_estimate': True,
    },
    'spark_linear_baseline': {
        'flops_per_token': baseline_linear_flops,
        'excludes': ['attention score/value sequence-length terms', 'normalization', 'nonlinearities', 'memory traffic', 'kernel launches'],
    },
    'one_expert_fast_read_fraction_of_linear_baseline': 2*PORTS*read_macs_per_port / baseline_linear_flops,
    'episodic_query_boundary_search': {
        '4096_slots_128d_dot_products_macs': 4096*128,
        '16384_slots_128d_dot_products_macs': 16384*128,
        'topk_selection_and_normalization_not_counted': True,
        'decode_selection': 'Fixed shortlist; optional additional model query at declared inspection boundary',
    },
    'active_static_payload_bytes_per_token': {'compact_pilot_bf16': 2*4*32*2, 'expanded_candidate_bf16': 2*4*64*2},
    'write_and_training_limits': [
        'State update is linear in accepted observation tokens; no constant-cost lifetime-learning claim.',
        'Live state excludes indices, alignment, allocator overhead, copies, retained autograd states and replay data.',
        'Training differentiable memory through a frozen backbone still requires activation/backward work.',
        'A second backbone observation pass can dominate arithmetic savings.',
        'No host or GPU throughput or fit claim is made by these formulas.'
    ],
}
assert result['profiles']['compact_pilot']['static_table_bf16_bytes'] == 8*MIB
assert result['profiles']['expanded_candidate']['static_table_bf16_bytes'] == 64*MIB
assert result['profiles']['compact_pilot']['shared_fast_state_fp32_bytes'] == 65536
assert result['profiles']['expanded_candidate']['shared_fast_state_fp32_bytes'] == 262144
if __name__ == '__main__':
    output = Path(__file__).with_name('analytical_costs.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
