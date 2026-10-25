# Independent V1 wide-connector evidence review

**Reviewer:** `V1EvidenceReview`, read-only, 25 October 2026.

## Verdict

**Engineering: PASS. Learned quality: FAIL. Overall `accepted=false`.** The
attempt07/restart07/score08 chain shows no integrity discrepancies. Declared
gates were not relaxed; V2 and learned V3 visual memory remain blocked.

## Quality evidence

Correct image is 21/64 (32.8125%), below the 65% floor. Correct-minus-control
margins are wrong +14.0625 pp, shuffled +9.375 pp, no-image −1.5625 pp, oracle
caption −60.9375 pp and empty OCR +9.375 pp. Every margin gate fails. No-image
exceeds correct-image accuracy. All 32 families are represented and unknown rate
is zero, but those passing values do not establish visual capability.

Baseline and post-training retention are both 22/32 (68.75%), below the 75%
absolute floor. Retention drop is zero and baseline, primary and restart
retention generations are byte-identical.

## Integrity checks

- Fresh seed-17 initialization; 128/128 updates and 512 backward calls; no
  budget stop. Exactly six projector tensors changed, 43 froze, totaling 49
  extension tensors. Trainable parameters: 11,542,016.
- Fresh-process reload performed zero updates and reproduced all 384 evaluation
  records and 32 retention records bytewise. Evaluation SHA-256:
  `e4f745a52c2613f628fc01a40e2e0fd43f6dffebf28e8fbd30bc06f1dc9099bc`;
  retention SHA-256:
  `3d788bdb971059bb954de6677accabf035c4fe4068163ce86e7879ba980ef4ed`.
- Native and replay implementation maps each contain 85 entries. The sole
  difference is `scripts/score_vision_qualification.py`; runner, data and gate
  bytes are unchanged. Protocol, primary, restart and replay parent hashes match.
- Primary/reload peak sampled driver bytes are 11,956,273,152/10,768,056,320,
  swap growth is 0/0 and minimum host free is 74%/76%. Both guards exited within
  policy with `stop_reason=null`. These are sampled limits, not proof against
  unsampled transients.
- No obvious leakage was found: visual controls carry no text context, text-only
  controls carry no image, caption-only is an explicit oracle upper bound and
  OCR-only uses declared independently verified empty text.

The evidence supports only a bounded procedural/projector pilot, not broad
perception or deployment capability.
