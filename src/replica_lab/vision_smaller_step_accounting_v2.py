"""Versioned smaller-step accounting derivation with enforced identities.

The v1 declaration fixed the campaign at 4,824 decoder forwards while its own
component identity was 4,804, capped a wider readout at 2,400 while its
components summed to 2,390, declared 78 visual observations beside 78 cache
hits plus 78 cache misses, and charged a projector pass to every decoder pass.
Those constants are preserved in ``configs/vision_smaller_step_lr3e5_v1.json``
and in the v1 launch packet; this module is the v2 amendment that derives the
cap from the declared schedule and validates the identities rather than
trusting stale literals.

Units follow ``replica_lab.vision_forward_accounting`` exactly.  A wider
readout has 98 available logical cells out of 104 records (six unmatched
sibling placeholders).  One no-image control exists per training row, so 26 of
the 98 available cells carry no visual patches and the remaining 72
observations run the encoder.  Every available cell contributes three
prepare/sense passes and two explicit-logits passes.  Generation contributes one
decoder pass per generated token, including the prefill.  The only visual-less
generation passes are those of the 26 no-image cells.

The module is CPU-only: it validates declarations, never imports a native
runtime and never loads a model.
"""
from __future__ import annotations

from typing import Any, Mapping

FORMAT = "replica.vision_smaller_step.accounting.v2"
PROTOCOL_FORMAT = "replica.vision_smaller_step_config.v2"
PROTOCOL_VERSION = "replica.vision_smaller_step.v2"
NAME = "vision_smaller_step_lr3e5_v2"

# Derived schedule inputs.  They are the same values used to build the schedule
# in the v1/v2 protocol and the preserved completed-arm evidence.
WIDER_CELLS = 98
WIDER_RECORDS = 104
WIDER_PLACEHOLDERS = 6
WIDER_NOIMAGE_CELLS = 26
WIDER_IMAGE_CELLS = WIDER_CELLS - WIDER_NOIMAGE_CELLS  # 72
WIDER_GENERATED_TOKENS = 1900
WIDER_LOGICAL_CELLS = 98
WIDER_LOGICAL_OPERATIONS = 3 * WIDER_LOGICAL_CELLS
UPDATE_BATCH_EVALUATIONS = 2
UPDATE_BATCH_ROWS = 4
UPDATE_ROWS = 4
UPDATE_BACKWARD_CALLS = 4
UPDATE_OPTIMIZER_STEPS = 1
MAX_SEQUENCE_TOKENS = 288
UPDATE_TEXT_TOKENS_PER_PASS = 150
VISUAL_TOKENS_PER_PASS = 196

# The v1 caps that are superseded.  They are carried in the amendment and must
# never be silently reused as if they were derived.
V1_SUPERSEDED = {
    "wider_readout_template.per_readout.decoder_forward_calls": 2400,
    "wider_readout_template.per_readout.visual_forward_calls": 2400,
    "wider_readout_template.per_readout.projector_passes": 2400,
    "wider_readout_template.per_readout.visual_observation_calls": 78,
    "wider_readout_template.per_readout.visual_encoder_calls": 78,
    "wider_readout_template.per_readout.visual_encoder_cache_hits": 78,
    "wider_readout_template.per_readout.visual_encoder_cache_misses": 78,
    "wider_readout_template.cumulative.decoder_forward_calls": 4800,
    "wider_readout_template.cumulative.visual_forward_calls": 4800,
    "wider_readout_template.cumulative.projector_passes": 4800,
    "wider_readout_template.cumulative.visual_observation_calls": 156,
    "wider_readout_template.cumulative.visual_encoder_calls": 156,
    "wider_readout_template.cumulative.visual_encoder_cache_hits": 156,
    "wider_readout_template.cumulative.visual_encoder_cache_misses": 156,
    "campaign_cumulative.decoder_forward_calls": 4824,
    "campaign_cumulative.visual_forward_calls": 4824,
    "campaign_cumulative.projector_passes": 4824,
    "campaign_cumulative.visual_observation_calls": 168,
    "campaign_cumulative.visual_encoder_calls": 168,
    "campaign_cumulative.visual_encoder_cache_hits": 168,
    "campaign_cumulative.visual_encoder_cache_misses": 168,
    "limits.max_decoder_forwards_per_readout": 2400,
    "limits.max_visual_encoder_calls_per_readout": 78,
    "limits.max_visual_encoder_calls": 168,
}

COUNTERS = (
    "decoder_forward_calls",
    "prepare_sense_passes",
    "explicit_logits_forwards",
    "generation_prefill_calls",
    "generation_decode_calls",
    "generated_tokens",
    "processed_text_positions",
    "processed_visual_positions",
    "visual_forward_calls",
    "projector_passes",
    "visual_observation_calls",
    "visual_encoder_calls",
    "visual_encoder_cache_hits",
    "visual_encoder_cache_misses",
    "backward_calls",
    "optimizer_steps",
    "logical_scoring_cells",
    "logical_scoring_operations",
)


class AccountingDeclarationError(ValueError):
    """A versioned accounting declaration is internally inconsistent."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise AccountingDeclarationError(message)


def _nonnegative_int(value: Any, name: str) -> int:
    _require(type(value) is int and value >= 0,
             f"{name} must be a nonnegative integer")
    return int(value)


def _zero() -> dict[str, int]:
    return {key: 0 for key in COUNTERS}


def derivation_inputs() -> dict[str, Any]:
    """Return the exact inputs the v2 caps are derived from."""
    return {
        "wider": {
            "records_per_readout": WIDER_RECORDS,
            "available_per_readout": WIDER_CELLS,
            "placeholders_per_readout": WIDER_PLACEHOLDERS,
            "noimage_cells_per_readout": WIDER_NOIMAGE_CELLS,
            "image_cells_per_readout": WIDER_IMAGE_CELLS,
            "generated_tokens_per_readout": WIDER_GENERATED_TOKENS,
            "prepares_per_cell": 3,
            "explicit_logits_per_cell": 2,
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "visual_tokens_per_pass": VISUAL_TOKENS_PER_PASS,
            "expected_cache_hits_per_readout": 0,
            "expected_cache_misses_per_readout": WIDER_IMAGE_CELLS,
        },
        "update_batch": {
            "evaluations": UPDATE_BATCH_EVALUATIONS,
            "rows_per_evaluation": UPDATE_BATCH_ROWS,
        },
        "update": {
            "rows": UPDATE_ROWS,
            "backward_calls": UPDATE_BACKWARD_CALLS,
            "optimizer_steps": UPDATE_OPTIMIZER_STEPS,
        },
    }


def _wider_readout() -> dict[str, int]:
    prepare = 3 * WIDER_CELLS
    explicit = 2 * WIDER_CELLS
    prefill = WIDER_CELLS
    generated = WIDER_GENERATED_TOKENS
    decode = generated - prefill
    decoder = prepare + explicit + prefill + decode
    observations = WIDER_IMAGE_CELLS
    # Every observation is budgeted as an encoder miss (full encoder pass).
    # Cache hits may occur but can only turn a miss into a hit inside the same
    # declared observation budget.
    encoder = observations
    misses = observations
    hits_cap = observations
    # A no-image cell contributes three prepares and two explicit-logits
    # passes but no visual patches.  At the declared generation cap the
    # visual-bearing passes are therefore the decoder cap minus those 5x26
    # visual-less passes; no-image generation can only lower this number.
    visual_forward = decoder - 5 * WIDER_NOIMAGE_CELLS
    processed_visual = VISUAL_TOKENS_PER_PASS * visual_forward
    processed_text = MAX_SEQUENCE_TOKENS * decoder
    counters = _zero()
    counters.update({
        "decoder_forward_calls": decoder,
        "prepare_sense_passes": prepare,
        "explicit_logits_forwards": explicit,
        "generation_prefill_calls": prefill,
        "generation_decode_calls": decode,
        "generated_tokens": generated,
        "processed_text_positions": processed_text,
        "processed_visual_positions": processed_visual,
        "visual_forward_calls": visual_forward,
        "projector_passes": visual_forward,
        "visual_observation_calls": observations,
        "visual_encoder_calls": encoder,
        "visual_encoder_cache_hits": hits_cap,
        "visual_encoder_cache_misses": misses,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "logical_scoring_cells": WIDER_LOGICAL_CELLS,
        "logical_scoring_operations": WIDER_LOGICAL_OPERATIONS,
    })
    return counters


def _update_batch_evaluation() -> dict[str, int]:
    prepare = UPDATE_BATCH_ROWS
    explicit = UPDATE_BATCH_ROWS
    decoder = prepare + explicit
    observations = UPDATE_BATCH_ROWS
    visual_forward = decoder
    counters = _zero()
    counters.update({
        "decoder_forward_calls": decoder,
        "prepare_sense_passes": prepare,
        "explicit_logits_forwards": explicit,
        "processed_text_positions": (
            2 * UPDATE_BATCH_ROWS * UPDATE_TEXT_TOKENS_PER_PASS),
        "processed_visual_positions": VISUAL_TOKENS_PER_PASS * visual_forward,
        "visual_forward_calls": visual_forward,
        "projector_passes": visual_forward,
        "visual_observation_calls": observations,
        "visual_encoder_calls": observations,
        "visual_encoder_cache_hits": observations,
        "visual_encoder_cache_misses": observations,
    })
    return counters


def _update() -> dict[str, int]:
    decoder = 2 * UPDATE_ROWS
    visual_forward = decoder
    counters = _zero()
    counters.update({
        "decoder_forward_calls": decoder,
        "processed_text_positions": 2 * UPDATE_ROWS * UPDATE_TEXT_TOKENS_PER_PASS,
        "processed_visual_positions": VISUAL_TOKENS_PER_PASS * visual_forward,
        "visual_forward_calls": visual_forward,
        "projector_passes": visual_forward,
        "visual_observation_calls": UPDATE_ROWS,
        "visual_encoder_calls": UPDATE_ROWS,
        "visual_encoder_cache_hits": UPDATE_ROWS,
        "visual_encoder_cache_misses": UPDATE_ROWS,
        "backward_calls": UPDATE_BACKWARD_CALLS,
        "optimizer_steps": UPDATE_OPTIMIZER_STEPS,
    })
    return counters


def derive_accounting(*, wall_limit_seconds: int = 1200,
                      artifact_bytes: int = 33554432,
                      wall_check_every_calls: int = 64) -> dict[str, Any]:
    """Derive the complete v2 accounting block from the schedule inputs."""
    wider = _wider_readout()
    update_batch = _update_batch_evaluation()
    update = _update()
    wider_cumulative = {key: 2 * value for key, value in wider.items()}
    update_batch_cumulative = {
        key: UPDATE_BATCH_EVALUATIONS * value
        for key, value in update_batch.items()}
    campaign = {
        key: (wider_cumulative[key] + update_batch_cumulative[key]
              + update[key])
        for key in COUNTERS}
    return {
        "format": FORMAT,
        "derivation": derivation_inputs(),
        "supersedes_v1": {
            "notice": (
                "The v1 declaration and launch packet remain preserved and "
                "unmodified. These caps are superseded only inside this v2 "
                "amendment, which derives each value from the schedule and "
                "enforces the identities below."),
            "old_caps": dict(V1_SUPERSEDED),
            "corrections": [
                {
                    "counter": "wider_readout_template.per_readout."
                               "decoder_forward_calls",
                    "old": 2400,
                    "new": wider["decoder_forward_calls"],
                    "reason": (
                        "294 prepares + 196 explicit logits + 1,900 generated "
                        "tokens = 2,390, not 2,400."),
                },
                {
                    "counter": "campaign_cumulative.decoder_forward_calls",
                    "old": 4824,
                    "new": campaign["decoder_forward_calls"],
                    "reason": (
                        "2x2,390 wider passes + 2x8 update-batch passes + 8 "
                        "update forwards = 4,804, not 4,824."),
                },
                {
                    "counter": "visual_observation_calls per wider readout",
                    "old": 78,
                    "new": wider["visual_observation_calls"],
                    "reason": (
                        "98 available cells minus the 26 no-image controls "
                        "equals 72 observations; 78 counted six unmatched "
                        "sibling placeholders as observations."),
                },
                {
                    "counter": "visual_encoder_cache_hits plus "
                               "visual_encoder_cache_misses per wider readout",
                    "old": "78 + 78 = 156 against 78 observations",
                    "new": (
                        f"{wider['visual_encoder_cache_hits']} hit cap plus "
                        f"{wider['visual_encoder_cache_misses']} misses "
                        "against 72 observations; runtime enforces "
                        "observations = hits + misses and encoder calls = "
                        "misses."),
                    "reason": (
                        "An observation is exactly one cache hit or one "
                        "cache miss; it cannot be both."),
                },
                {
                    "counter": "visual_forward_calls and projector_passes",
                    "old": 2400,
                    "new": wider["visual_forward_calls"],
                    "reason": (
                        "Only visual-bearing decoder passes advance the "
                        "projector; the 26 no-image cells contribute five "
                        "visual-less passes each. Projector passes equal "
                        "visual-bearing passes, never all decoder passes."),
                },
                {
                    "counter": "update_batch_evaluation prepare/logits",
                    "old": "0 prepares and 0 explicit logits for 8 decoder "
                           "passes",
                    "new": (
                        f"{update_batch['prepare_sense_passes']} prepares and "
                        f"{update_batch['explicit_logits_forwards']} explicit "
                        "logits"),
                    "reason": (
                        "Each of the four rows performs one prepare/sense "
                        "pass and one teacher-forced logits pass; the v1 "
                        "zero caps contradicted its own eight decoder passes."),
                },
            ],
        },
        "units": {
            "decoder_forward_calls": "decoder_forward_call",
            "prepare_sense_passes": "prepare_sense_pass",
            "explicit_logits_forwards": "explicit_logits_forward",
            "generation_prefill_calls": "generation_prefill_call",
            "generation_decode_calls": "generation_decode_call",
            "generated_tokens": "generated_token",
            "processed_text_positions": "processed_text_position",
            "processed_visual_positions": "processed_visual_position",
            "visual_forward_calls": "visual_forward_call",
            "projector_passes": "projector_pass",
            "visual_observation_calls": "visual_observation_call",
            "visual_encoder_calls": "visual_encoder_call",
            "visual_encoder_cache_hits": "visual_encoder_cache_hit",
            "visual_encoder_cache_misses": "visual_encoder_cache_miss",
            "backward_calls": "backward_call",
            "optimizer_steps": "optimizer_step",
            "logical_scoring_cells": "logical_scoring_cell",
            "logical_scoring_operations": "logical_scoring_operation",
        },
        "wider_readout_template": {
            "per_readout": wider,
            "cumulative": wider_cumulative,
            "wall_limit_seconds": wall_limit_seconds,
            "wall_check_every_calls": wall_check_every_calls,
        },
        "update_batch_evaluation": {
            "per_evaluation": update_batch,
            "cumulative": update_batch_cumulative,
        },
        "update": update,
        "campaign_cumulative": campaign,
        "expected_counts": {
            "wider_per_readout": {
                "visual_observation_calls":
                    wider["visual_observation_calls"],
                "visual_encoder_cache_hits": 0,
                "visual_encoder_cache_misses":
                    wider["visual_encoder_cache_misses"],
                "identity": (
                    f'{wider["visual_observation_calls"]} observations = '
                    f'0 expected hits + '
                    f'{wider["visual_encoder_cache_misses"]} misses'),
            },
            "campaign": {
                "visual_observation_calls":
                    campaign["visual_observation_calls"],
                "visual_encoder_cache_hits": 0,
                "visual_encoder_cache_misses":
                    campaign["visual_encoder_cache_misses"],
                "identity": (
                    f'{campaign["visual_observation_calls"]} observations = '
                    f'0 expected hits + '
                    f'{campaign["visual_encoder_cache_misses"]} misses'),
            },
        },
        "wall_limit_seconds": wall_limit_seconds,
        "artifact_bytes": artifact_bytes,
        "wall_check_every_calls": wall_check_every_calls,
        "identities": _identities(wider, wider_cumulative, update_batch,
                                  update_batch_cumulative, update, campaign),
    }


def _identities(wider: Mapping[str, int], wider_cumulative: Mapping[str, int],
                update_batch: Mapping[str, int],
                update_batch_cumulative: Mapping[str, int],
                update: Mapping[str, int],
                campaign: Mapping[str, int]) -> list[str]:
    return [
        "wider.decoder = wider.prepares + wider.explicit_logits + "
        "wider.generated_tokens",
        "wider.generated = wider.prefill + wider.decode",
        "wider.visual_observations = wider.cache_hits_recorded + "
        "wider.cache_misses_recorded",
        "wider.visual_encoder_calls = wider.cache_misses_recorded",
        "wider.visual_forward = wider.decoder - 5 * wider.noimage_cells",
        "wider.processed_visual = 196 * wider.visual_forward",
        "wider.projector_passes = wider.visual_forward",
        "update_batch.decoder = update_batch.prepares + "
        "update_batch.explicit_logits",
        "update_batch.visual_forward = update_batch.decoder",
        "update.decoder = 8 (four rows, one sense and one logits pass)",
        "update.visual_forward = update.decoder",
        "campaign.decoder = campaign.prepares + campaign.explicit_logits + "
        "campaign.generated_tokens + 8 update forwards",
        "campaign = wider.cumulative + update_batch.cumulative + update",
        "campaign.visual_observations = campaign.cache_hits_recorded + "
        "campaign.cache_misses_recorded",
        "campaign.visual_encoder_calls = campaign.cache_misses_recorded",
        "campaign.processed_visual = 196 * campaign.visual_forward",
    ]


def _check_identity_block(wider: Mapping[str, int],
                          wider_cumulative: Mapping[str, int],
                          update_batch: Mapping[str, int],
                          update_batch_cumulative: Mapping[str, int],
                          update: Mapping[str, int],
                          campaign: Mapping[str, int]) -> None:
    _require(wider["decoder_forward_calls"]
             == wider["prepare_sense_passes"]
             + wider["explicit_logits_forwards"]
             + wider["generated_tokens"],
             "Wider decoder passes do not equal prepares + logits + tokens")
    _require(wider["generated_tokens"]
             == wider["generation_prefill_calls"]
             + wider["generation_decode_calls"],
             "Wider generated tokens do not equal prefill + decode")
    _require(wider["visual_forward_calls"]
             == wider["decoder_forward_calls"]
             - 5 * WIDER_NOIMAGE_CELLS,
             "Wider visual-bearing passes must exclude the no-image cells")
    _require(wider["visual_forward_calls"] < wider["decoder_forward_calls"],
             "Visual/projector passes must not equal all decoder passes")
    _require(wider["visual_encoder_calls"]
             == wider["visual_encoder_cache_misses"],
             "Wider encoder calls must equal cache misses")
    _require(wider["visual_encoder_cache_hits"] <= wider["visual_observation_calls"]
             and wider["visual_encoder_cache_misses"]
             == wider["visual_observation_calls"],
             "Wider observation budget must be one hit or one miss each")
    _require(wider["processed_visual_positions"]
             == VISUAL_TOKENS_PER_PASS * wider["visual_forward_calls"],
             "Wider visual positions must equal 196 per visual-bearing pass")
    _require(wider["projector_passes"] == wider["visual_forward_calls"],
             "Wider projector passes must equal visual-bearing passes")
    _require(wider["processed_text_positions"]
             == MAX_SEQUENCE_TOKENS * wider["decoder_forward_calls"],
             "Wider text positions must be bounded by the per-pass sequence")
    _require(update_batch["decoder_forward_calls"]
             == update_batch["prepare_sense_passes"]
             + update_batch["explicit_logits_forwards"],
             "Update-batch decoder passes do not equal prepares + logits")
    _require(update_batch["visual_forward_calls"]
             == update_batch["decoder_forward_calls"],
             "Update-batch evaluations are fully visual-bearing")
    _require(update_batch["visual_encoder_calls"]
             == update_batch["visual_encoder_cache_misses"],
             "Update-batch encoder calls must equal cache misses")
    _require(update_batch["visual_encoder_cache_misses"]
             == update_batch["visual_observation_calls"],
             "Update-batch observations must be budgeted as misses")
    _require(update_batch["processed_visual_positions"]
             == VISUAL_TOKENS_PER_PASS
             * update_batch["visual_forward_calls"],
             "Update-batch visual positions must equal 196 per pass")
    _require(update["decoder_forward_calls"] == 2 * UPDATE_ROWS,
             "Update decoder passes must be two per row")
    _require(update["visual_forward_calls"]
             == update["decoder_forward_calls"],
             "Update forwards are fully visual-bearing")
    _require(update["visual_encoder_calls"]
             == update["visual_encoder_cache_misses"]
             == update["visual_observation_calls"],
             "Update observations must be budgeted as misses")
    _require(update["backward_calls"] == UPDATE_BACKWARD_CALLS
             and update["optimizer_steps"] == UPDATE_OPTIMIZER_STEPS,
             "Update backward/step counts differ")
    for key in COUNTERS:
        _require(wider_cumulative[key] == 2 * wider[key],
                 f"Wider cumulative is not two readouts for {key}")
        _require(update_batch_cumulative[key]
                 == UPDATE_BATCH_EVALUATIONS * update_batch[key],
                 f"Update-batch cumulative differs for {key}")
        _require(campaign[key]
                 == wider_cumulative[key]
                 + update_batch_cumulative[key] + update[key],
                 f"Campaign is not the sum of its stages for {key}")
    _require(campaign["decoder_forward_calls"]
             == campaign["prepare_sense_passes"]
             + campaign["explicit_logits_forwards"]
             + campaign["generated_tokens"]
             + update["decoder_forward_calls"],
             "Campaign decoder identity differs")
    _require(campaign["visual_encoder_calls"]
             == campaign["visual_encoder_cache_misses"],
             "Campaign encoder calls must equal cache misses")
    _require(campaign["visual_encoder_cache_hits"]
             <= campaign["visual_observation_calls"],
             "Campaign hit cap exceeds observations")
    _require(campaign["visual_encoder_cache_misses"]
             == campaign["visual_observation_calls"],
             "Campaign observations must be budgeted as misses")
    _require(campaign["processed_visual_positions"]
             == VISUAL_TOKENS_PER_PASS
             * campaign["visual_forward_calls"],
             "Campaign visual positions must equal 196 per visual pass")
    _require(campaign["projector_passes"]
             == campaign["visual_forward_calls"],
             "Campaign projector passes must equal visual-bearing passes")
    _require(campaign["visual_forward_calls"]
             < campaign["decoder_forward_calls"],
             "Campaign visual passes must not equal all decoder passes")


def validate_accounting(accounting: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one v2 accounting block against derivation and identities."""
    _require(isinstance(accounting, Mapping),
             "Accounting must be a mapping")
    _require(accounting.get("format") == FORMAT,
             "Accounting format is not the v2 amendment")
    required = {
        "format", "derivation", "supersedes_v1", "units",
        "wider_readout_template", "update_batch_evaluation", "update",
        "campaign_cumulative", "wall_limit_seconds", "artifact_bytes",
        "wall_check_every_calls", "identities", "expected_counts",
    }
    allowed = required | {"amends"}
    _require(set(accounting) <= allowed,
             f"Accounting fields differ: {sorted(accounting)}")
    _require(required <= set(accounting),
             f"Accounting fields are missing: {sorted(required - set(accounting))}")
    _require(accounting["derivation"] == derivation_inputs(),
             "Accounting derivation inputs differ from the declared schedule")
    expected = derive_accounting(
        wall_limit_seconds=accounting["wall_limit_seconds"],
        artifact_bytes=accounting["artifact_bytes"],
        wall_check_every_calls=accounting["wall_check_every_calls"])
    for section in ("wider_readout_template", "update_batch_evaluation",
                    "update", "campaign_cumulative"):
        _require(accounting[section] == expected[section],
                 f"Accounting {section} is not derived from the schedule")
    _require(accounting["units"] == expected["units"],
             "Accounting units differ")
    _require(accounting["identities"] == expected["identities"],
             "Accounting identity list differs")
    _require(accounting["expected_counts"] == expected["expected_counts"],
             "Expected exact hit/miss counts differ")
    for section in ("wider_per_readout", "campaign"):
        counts = accounting["expected_counts"][section]
        _require(counts["visual_encoder_cache_hits"]
                 + counts["visual_encoder_cache_misses"]
                 == counts["visual_observation_calls"],
                 f"{section}: expected observations != hits + misses")
    _check_identity_block(
        accounting["wider_readout_template"]["per_readout"],
        accounting["wider_readout_template"]["cumulative"],
        accounting["update_batch_evaluation"]["per_evaluation"],
        accounting["update_batch_evaluation"]["cumulative"],
        accounting["update"],
        accounting["campaign_cumulative"])
    return dict(accounting)


def expected_measured_work() -> dict[str, int]:
    """Return the campaign-shape counters used by readout validation."""
    campaign = derive_accounting()["campaign_cumulative"]
    return {
        "wider_readouts": 2,
        "update_batch_evaluations": UPDATE_BATCH_EVALUATIONS,
        "updates": 1,
        "readout_stages": 5,
        "decoder_forward_calls": campaign["decoder_forward_calls"],
        "prepare_sense_passes": campaign["prepare_sense_passes"],
        "explicit_logits_forwards": campaign["explicit_logits_forwards"],
        "generation_prefill_calls": campaign["generation_prefill_calls"],
        "generation_decode_calls": campaign["generation_decode_calls"],
        "generated_tokens": campaign["generated_tokens"],
        "visual_observation_calls": campaign["visual_observation_calls"],
        "visual_encoder_calls_max": campaign["visual_encoder_calls"],
        "visual_encoder_cache_misses_max":
            campaign["visual_encoder_cache_misses"],
        "visual_forward_calls": campaign["visual_forward_calls"],
        "projector_passes": campaign["projector_passes"],
        "backward_calls": campaign["backward_calls"],
        "optimizer_steps": campaign["optimizer_steps"],
    }


def validate_live_counters(counters: Mapping[str, int], *,
                           label: str = "campaign",
                           training_forwards: int = 0) -> dict[str, int]:
    """Validate measured native counters against the exact identities.

    This is intentionally stricter than a cap check: stale literals that only
    bound a count cannot satisfy these equalities.  ``training_forwards`` is the
    number of phase-labelled training forwards, which the wave decodes as
    neither prepare/sense nor explicit-logits passes in the tap.
    """
    values = {key: _nonnegative_int(counters[key], f"{label}.{key}")
              for key in COUNTERS}
    _nonnegative_int(training_forwards, f"{label}.training_forwards")
    _require(values["decoder_forward_calls"]
             == values["prepare_sense_passes"]
             + values["explicit_logits_forwards"]
             + values["generated_tokens"]
             + training_forwards,
             f"{label}: decoder passes != prepares + logits + generated "
             f"tokens + {training_forwards} training forwards")
    update_batch_forwards = (
        UPDATE_BATCH_EVALUATIONS * 2 * UPDATE_BATCH_ROWS)
    teacher_forwards = update_batch_forwards + training_forwards
    wider_forwards = values["decoder_forward_calls"] - teacher_forwards
    _require(wider_forwards >= 0,
             f"{label}: decoder passes are shorter than the update stages")
    _require(values["processed_text_positions"]
             == MAX_SEQUENCE_TOKENS * wider_forwards
             + UPDATE_TEXT_TOKENS_PER_PASS * teacher_forwards,
             f"{label}: text positions != 288 per wider pass plus 150 per "
             "update/update-batch pass")
    _require(values["generated_tokens"]
             == values["generation_prefill_calls"]
             + values["generation_decode_calls"],
             f"{label}: generated tokens != prefill + decode")
    _require(values["visual_encoder_calls"]
             == values["visual_encoder_cache_misses"],
             f"{label}: encoder calls != cache misses")
    _require(values["visual_observation_calls"]
             == values["visual_encoder_cache_hits"]
             + values["visual_encoder_cache_misses"],
             f"{label}: observations != cache hits + misses")
    _require(values["processed_visual_positions"]
             == VISUAL_TOKENS_PER_PASS * values["visual_forward_calls"],
             f"{label}: visual positions != 196 per visual pass")
    _require(values["projector_passes"] == values["visual_forward_calls"],
             f"{label}: projector passes != visual-bearing passes")
    _require(values["visual_forward_calls"]
             < values["decoder_forward_calls"],
             f"{label}: visual passes must be fewer than all decoder passes")
    return values
