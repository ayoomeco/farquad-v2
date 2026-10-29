---
name: likeness-lora
description: >-
  Reference only: use when training or testing a person likeness LoRA to stack
  with H3 reference-to-video refs. The default Bill render loads NO likeness LoRA.
---
# Likeness LoRA (reference, not default)

The default render loads no likeness LoRA; the refs carry the likeness. Only stack one as a separate, quoted test with the owner's yes.

1. Dataset: 25-40 real photos of a consenting person (varied angles, lighting, expressions). Un-mirror front-camera and mirror selfies (check for backwards text). Caption with a rare trigger token; describe pose, outfit, setting and light only (no face/skin/hair/tattoo words).
2. Train on the same base the swap uses (H3 ref2va int8 via musubi-tuner minimax_h3, task ref2va --one_frame --video_only): rank 16 / alpha 16, adamw8bit, LR 2e-4 constant_with_warmup (50), batch 1, 1024 buckets; ~1250-2000 steps, checkpoint every 250. About 17 s/it on an A6000. Quote time and cost first; get a yes.
3. Eval on one fixed clip/seed/refs: no-LoRA baseline vs several checkpoints at low strengths. Score with ArcFace (insightface buffalo_l) against the mean of the real photos (median + 10th percentile), background drift vs the source, identity pops.
4. Deliver a labelled side-by-side grid and a scores table; pick the best, name a backup.

## Known findings
- Gains over refs alone were small. Still-photo LoRAs learn studio lighting; high strength or late checkpoints darken or repaint backgrounds.
- On a swap, strengths of 0.35-0.68 broke identity on one test (the face flipped back to the source person). A low 0.15-0.25 range is the untested next step.
