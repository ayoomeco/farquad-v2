---
name: video-prelaunch-gate
description: >-
  MANDATORY before and after every paid video/ref render: the owner-approved
  Oct 29 baseline table plus the pre-launch and post-render gates, run through
  qc/selfcheck.py. Merged from video-prelaunch-gate + video-pipeline-standard.
---
# Video pre-launch gate + pipeline standard (Oct 29 baseline)

Nothing launches until every box passes and the owner OKs any deviation. There is no audit bot: Bill checks himself with `python3 qc/selfcheck.py pre` / `post` and pastes the tables with evidence. The full flow is in `fmb-workflow`; this is the single source of truth for settings.

## Baseline (any change goes in the what's-different table and needs a yes)
| Setting | Value |
|---|---|
| GPU | RunPod RTX PRO 6000 Blackwell **SECURE** (~$2.49/hr), one pod per clip, in parallel. Stills/refs: RTX A6000 SECURE (~$0.59/hr) |
| Image | `ghcr.io/ayoomeco/genie-jutsu-worker:fastboot-comfy-0.42.5-r2` (public), ComfyUI 0.42.5, exact-LoRA loader, H3LoadReport lossy 0 with `expect` = layer counts (never strengths) |
| Model | Singularity **H3 v1.3 int8** |
| LoRAs | realism 0.7 if every character looks human, else 0 (nothing in between); character swap v1.5 at 1.0; lightx2v 8-step v1.1 at 0.8 (exactly one speed LoRA); no likeness LoRA |
| Sampler | 8 steps, euler / simple, shift 6 |
| Seed | **77**; rerolls 78, 79 only for a bad or errored piece, never for graph errors |
| Refs | separate images (never a sheet): head front (neutral, mouth closed), body front, body back when seen from behind. **1536 long side, flat 8-bit RGB, no alpha, `ref_image_size` max**, ≤9 per shot. Background color written into the generation prompt |
| Size | **native render res is the deliverable**: 1152x640 (16:9), 640x1152 (vertical), 4:3 kept at about the same pixels (960x768 / 1024x768) |
| Frames | ≤141 per piece, never across a hard cut, 22-frame guide chains; final **trimmed to the planned frames** |
| fps / audio | **source fps**; driver muted, no audio inputs; **source audio muxed back**, same length; silent deliverables flagged up front |
| Post | none on video: **no SeedVR2, no ResCascade, no Fizgig**, no interpolation, grade, grain, face restore or paste-back. Encode H.264 High 4.2, yuv420p, bt709, AAC, +faststart |

Replaced: the old RTX A6000 video GPU, seed 42, `ref_image_size` match, 2048 refs and SeedVR2-to-1080 post. Don't bring them back without a quoted test.

## Pre-launch gate ($0, local)
1. `python3 video/run_video.py --clip <id>` prints BUILT, NOT RUN, `budget gate ok`, `missing inputs: none`, and refuses without `--go`.
2. `python3 qc/selfcheck.py pre --clip <id>`: paste the table, fix every FAIL.
3. Eyes only: native HD source, cuts checked by eye, contact sheet, the window avoids crowd ID chaos and unwanted poses; multi-person mask overlay watched; every subject's clothing + position in the prompt; per-shot prompts with ref roles, the mouth/expression line, the add-nothing line, no face words on backs.
4. Built graphs diffed against the table above; banned steps absent.
5. Send the `| Setting | Old | New |` table, open risks, SILENT flags and "About $X (cap $Y), about N min." Launch only after an explicit yes. An unapproved deviation gets fixed, not launched.

## During the run
Live status on stage changes only; confirm the pod on RunPod before saying it's live. OOM or graph error = the clip stops; no settings change on their own; fix and re-quote.

## Post-render gate (before the owner sees anything)
1. `python3 qc/selfcheck.py post --clip <id> --live`: frames = plan, fps = source, size, audio length = video, **luma check** (not black or washed out), frame stills not blank, spend ≤ cap, 0 pods on runner and RunPod.
2. Per piece, by eye against the masks: right character in each spot, no leftover original person, no ID/head swap, no extra heads, no flat-grey ref frames, no reframing or invented props, mouth/expression match, marks on the correct side, outfit right from the back, no seams. Then rerun with `--eyeballed`.
3. A bad piece: reroll within the seed rule and cap, else flag it with its timestamp and use the best attempt. Nothing unchecked reaches the final cut.
4. Verdict per clip (PASS/FAIL + timestamps), phone-friendly links, spend vs cap, pods 0, then the job record.

## Lip sync (only when asked)
LatentSync 1.6 with a MediaPipe face mesh on audible speakers only, as a separate quoted step; smoke-test first and abort if the wrapper falls back to the default VAE config.
