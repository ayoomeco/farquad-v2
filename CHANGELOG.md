# Changelog

## GJ V2

Approved A08 is the default MiniMax H3 reference-to-video render. Every item below is on by default.

1. ComfyUI is 0.42.5 or newer, and every render goes through the vendored `h3_exact_lora` node (`H3ExactLoraStack` + `H3LoadReport`). The load check aborts unless lossy merges are 0, every hook it asks for is present, and each LoRA is at its set strength. Stock chained loaders kept only the last bypass LoRA and merged adaLN weights lossily.
2. The duplicate speed LoRA is gone. The old turbo LoRA (`minimax_h3_turbo_v4_step600_ema.safetensors`, and any second speed LoRA) does not load next to lightx2v. The only speed LoRA is lightx2v Minimax-h3-Turbo 8-step v1.1 at 0.8 (`minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors`).
3. The personal likeness (Meek) LoRA is off. `m33k_lme_s0500.safetensors` is not in the default download set. Set `GJ_LIKENESS_LORA` only to opt in, and a strength of 0.35 or above is refused because it broke the swap.
4. Steps are 8 (was 6). Video shift is 6 and audio shift is 3 (was unset on the GJ graph, and 12/3 on the genjutsu distill path). Sampler is euler and the scheduler is simple (was er_sde / normal). Seed is 42. `ref_image_size` is `match`. Sage Attention stays on, after the shift node.
5. The prompt names each reference (image 1 face, image 2 body, image 3 outfit), says "Match the source's mouth and expression exactly.", and says "keep any objects already in the scene; add nothing." It does not list example props.
6. References are separate images: a neutral closed-mouth head close-up, then the body only on a full-body shot, then the outfit. Crops are upscaled with SeedVR2 in image mode to a 2048 short side, with no face restore. A combined character sheet is refused. M35, M50, and A10 graphs are not defaults.
7. Post is upscale only, on the pod, at the source frame rate. No frame interpolation, no colour grade, no grain, no face restore, no paste-back. Encode is H.264 High 4.2, yuv420p, bt709, AAC, +faststart. Stitch, score, and encode run before the pod is terminated. Only finished files are the deliverable.

### Old to new

| | Old default | GJ V2 |
|---|---|---|
| Version | Farquad V2 / app 0.3.0 | GJ V2 |
| ComfyUI | v0.39.1 | 0.42.5 or newer |
| LoRA loader | chained `LoraLoaderModelOnly` | `H3ExactLoraStack` + `H3LoadReport` (abort if lossy ≠ 0 or strength/hooks mismatch) |
| Speed LoRA | `minimax_h3_turbo_v4_step600_ema.safetensors` @ 0.8, plus genjutsu draft turbo4 / final base20 | only `minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors` @ 0.8 |
| Extra swap LoRA | `h3_character_swap_pro4500_1000.safetensors` @ 1.0 | removed |
| Likeness | A08 variants used Meek `m33k_lme_s0500.safetensors` @ 0.35 or 0.5 | off; refuse ≥ 0.35 |
| Steps | 6 | 8 |
| Shift | none on the GJ graph; distill 12/3 | video 6 / audio 3 |
| Sampler | er_sde / normal | euler / simple |
| Seed | base graph 44 (builder already defaulted 42) | 42 |
| ref_image_size | unset in the recipe | match |
| Prompt | one sheet per subject | role lines, mouth/expression line, keep existing objects and add nothing |
| Refs | `v5b_<key>.png` character sheet | `v5b_<key>_face.png`, `_body.png` only when full-body, `_outfit.png`; SeedVR2 image mode, short side 2048 |
| Post | SeedVR2, then RIFE to 30 fps, then LAB grade + grain; box stitch after the pod was killed, with Lab match and a B&W grade | on-pod upscale only, source fps, no interpolation / grade / grain / face-restore / paste-back; High 4.2, yuv420p, bt709, AAC, +faststart |
| Install root | `FARQUAD_HOME`, default `~/farquad` | `GJ_HOME`, else `FARQUAD_HOME`, else the clone directory |
