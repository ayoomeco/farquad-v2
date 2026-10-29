---
name: fmb-knowledge-base
description: >-
  Use before any Fire Marshal Bill job, quote or troubleshooting, and whenever a
  setting, cost, failure or rule is in question: the full working recipe v1.1
  (Oct 29, 2026), standing rules and self-check lists, so the knowledge base
  works without the repo.
---
# Fire Marshal Bill knowledge base (recipe v1.1, Oct 29, 2026)

Paths like `code/lib/graphs.py` refer to the fmb-v1 branch (see `fmb-install`). If this skill and the repo docs ever disagree, the newer patch notes entry wins; tell the owner.

# Part A: Working recipe

The code builds every graph below itself (`code/lib/graphs.py`) and asserts these values. The model files, Hugging Face repo@revision and sha256 are in `code/image/models.manifest.json`, and the node packs at their pinned commits are in `code/image/custom_nodes.txt` (KJNodes 2371136, VideoHelperSuite 4d907be, plus the vendored h3_exact_lora).

## 1. Video: H3 reference-to-video
| Item | Value |
|---|---|
| ComfyUI | 0.42.5 (GJ fastboot image), exact-LoRA loader (H3ExactLoraStack, node 80) + H3LoadReport (node 81), lossy must be 0 |
| Model | `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors` |
| Realism LoRA | `h3-realism-people-t2v-i2v-r2v.safetensors`: **0.7 if the character looks human, 0 if not** (cartoon, game, Muppet). Nothing in between. |
| Swap LoRA | `h3_character_swap_v1.5.safetensors` at 1.0 |
| Speed LoRA | `minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors` (lightx2v) at 0.8. Only one speed LoRA. |
| Sampler | 8 steps, euler / simple, shift 6 |
| Seed | 77; rerolls 78, 79 only when a piece errors or is bad |
| Size | 1152x640 for 16:9 (640x1152 vertical); 4:3 sources keep 4:3 at about the same pixel count (960x768, 1024x768); 1120x640 also used |
| Refs | 1536 long side, `ref_image_size` = max, each one its own LoadImage (nodes 300+), never a sheet |
| Frames | ≤141 per piece; long shots are chained with a 22-frame guide (MiniMaxH3AddGuide, node 16); a seam is a 10-frame hold + 12-frame crossfade |
| Audio | no audio inputs in the graph; the driver is muted |
| H3LoadReport `expect` | **layer counts [n, lossy]**, e.g. realism [104, 0]. Never strengths. When realism is off, drop its entry. |
| Banned | ResCascade (H3HiresFix), Fizgig (FizgigH3Tweaks), SeedVR2 on video |

### GPU and cost
- Video: **RunPod RTX PRO 6000 Blackwell SECURE, $2.49/hr**, one pod per clip, run in parallel. Boot takes about 12 min including model download (sets `h3_a08`, sha-verified, 3 tries).
- Stills: **RTX A6000 SECURE, $0.59/hr** (sets `qwen21`, `qwen2511_lite`, `seedvr2`).
- Piece times (PRO 6000): 1 person 141 f at 960x768 ≈ 340 s (≈ $0.24); 2 people with 4 refs at 1120x640 ≈ 450–480 s (≈ $0.32); short group shots ≈ 200–230 s, 90 f ≈ 400 s.
- Real clip costs Oct 29 (~15 s each, crashed starts included): one-person 4 pieces $1.32, two-person 3 pieces $1.74, group 9 pieces $1.12, for **$4.17 total**. The crashed starts ($0.50) are inside those numbers.
- Ref job: about 150 s per white, 120 s per edit and 60 s per upscale on an A6000, so a 5-character set (8 whites + 15 edits + 15 upscales) is about $1.20.

### Memory
- 90-frame max-size refs peak at about **45 GiB**. 3-character shots with 9 refs at 2048 went OOM (Oct 27). 6 refs at 1536 fit on the PRO 6000.
- On OOM, the clip stops. No settings get changed on their own.

## 2. Stills: character refs
| Stage | Recipe |
|---|---|
| Fresh whites (T) | Qwen-Image 2.1 `qwen_image_2.1_bf16.safetensors`, text encoder `qwen_2.5_vl_7b_fp8_scaled.safetensors`; 25 steps, cfg 1, euler/simple, seed 77, no LoRA; **decoded with the texture-fix VAE `texture_fix_vae_for_qwen_image_2.1_bf16.safetensors`**. A blank white canvas goes in as image 1 to set the shape (heads 896x1152, bodies 768x1376 ≈ 1 MP). Real photos go in as image 2+ for identity. |
| Edits to black (R) | Qwen-Image-Edit-2511 fp8 `qwen_image_edit_2511_fp8mixed.safetensors`, VAE `qwen_image_vae.safetensors` (stock: it partly undoes texfix); 1 MP, 40 steps, cfg 4, euler/simple, AuraFlow shift 3.1, CFGNorm 1. Front shots: LoRA off. Angles/back: `qwen-image-edit-2511-multiple-angles-lora.safetensors` at 0.9, no face words. |
| Upscale (U) | SeedVR2 7B `seedvr2_7b_fp16.safetensors` + `seedvr2_ema_vae_fp16.safetensors`, image mode, to 1536, 1 step, lab color, tiled VAE 512/128. No face restore. (Optional on refs; never on video.) |
| Checks | output size == canvas, pixel std > 10 (not washed out), flat 8-bit RGB, no alpha. The first miss stops the stage. |

**Ref spec per character:** (1) head front: neutral, mouth closed, face filling the frame; (2) full body front in the outfit; (3) **full body back in the same outfit** whenever they're ever seen from behind. Optional 3/4 head. The background color goes in the prompt ("solid pure black #000000 background, evenly black edge to edge"). Vault names: `H1_front, H2_34, H3_left, H4_right, H5_back, B1_front, B2_back, B3_side, O1_outfit`.

**Plastic skin causes:** edits stacked on AI images, fp8 + the angle LoRA, 1 MP with no detail pass, flat white light. Fix direction: one pass from real photos; texfix on decode (small gain); a realism LoRA test is parked.

## 3. Masks (who's-who)
- **YOLO11m-seg + ByteTrack** person tracks → a role per track (by side/size) → **SAM2.1-tiny** box prompt (box +12 px) on every frame. Files: `yolo11m-seg.pt`, `sam2.1_t.pt`. Runs on CPU for $0.
- Code: `video/build_masks.py <clip>`. Roles come from each subject's `side` (left/center/right) among the largest tracks. Outputs: a per-character gray mkv (ffv1), `overlay_<clip>_<piece>.mp4` with names on screen, and `masks_report.json` (missing frames, extra-person frames). Tested on a real two-person driver: 141 frames, both tracked, 0 missing.
- **H3 has no mask input.** Masks are used to (a) write each subject's position and clothing into the prompt and (b) check each piece afterward. Real mask-driven casting needs a different model path (SCAIL-2, 20 steps, less photoreal) and needs the owner's OK.
- Failure signs: an edge person with only an arm showing gets no track; crowds make IDs jump; 3 characters → H3 can swap heads.

## 4. Audio mux
- Cut the source audio for the exact window: `start = first_frame / fps`, `dur = frames / fps` (use 30000/1001 for 29.97).
- `video/prep_audio.py` cuts each clip's source window to a 48 kHz wav. With no audio block, the clip is flagged SILENT up front.
- Assembly runs on the pod (`lib/assemble.py`): chain seams (10-frame hold + 12-frame crossfade), side-by-side and a QC sheet.
- Then, locally, `checks.trim_remux`: keep exactly `frames_out` frames, mux the source audio cut to the same length, and encode H.264 High 4.2 yuv420p bt709 + AAC + faststart. This kills the frozen padded tail.
- Phone-safe encode: H.264 High 4.2, yuv420p, bt709, AAC, +faststart.

## 5. Dropbox big-file upload
`dbx/upall.py`: for each file, `files/get_temporary_upload_link` → **one curl POST of the whole file** → `sharing/create_shared_link_with_settings` (or reuse the existing link). Up to 150 MB per file. **Never chunked connector uploads.** The token is read from the environment only. Deliver the shared view links.

## 6. Live status
`$FMB_DATA/<job>/status_<clip>.txt` lines: STARTING → LAUNCHED (pod, rate, cap) → DOWNLOADING (the pod's proxy answers) → MODELS VERIFIED → FIRST OUTPUT → PIECE done n/N → ASSEMBLY ok → TRIM + AUDIO ok → DONE → TERMINATED (spend, pods left). FAILED <reason> and CAP GUARD lines can come at any point. Relay about every 5 min, and only on a stage change. Check RunPod directly before saying a run is live.

## 7. Known failure modes → fixes
| Symptom | Cause | Fix |
|---|---|---|
| Dies in 1–7 s on every seed, "H3LoadReport" | strength written into `expect` | keep the template's layer counts; the build asserts |
| Same crash 3x, paying each time | rerolling a graph error | graph/validation errors stop the clip, no reroll |
| Unclear crash | truncated error | `fail_<clip>_<piece>_s<seed>.log` with the full error + ComfyUI log tail |
| Output black or washed out but reported ok | no output check | `checks.luma` (most frames YMAX ≤ 16 or YMIN ≥ 235 = FAIL); SeedVR2 video is removed |
| 1080 looks painted/plastic | SeedVR2 over-smooths video | deliver native |
| Frozen tail, audio runs on | padded frames | trim to planned frames, match the audio |
| Odd color behind a ref | alpha in the PNG | flatten to RGB |
| Front logo on someone's back | no back ref | make a back-view ref of the outfit |
| Arms crossed despite the prompt | the source pose wins | pick another window |
| Heads swapped between characters, extra heads | 3 characters, text-only placement | fewer characters per shot, tighter positions, or SCAIL with the OK |
| Washed-out or wrong-size still | blank-canvas trick | size + std check stops the stage |
| Spend over budget / wrong total | gate counted only today, missed roots, double-counted crashed pods | gate scans `ledger_roots`, prints every ledger, counts per-pod ledgers since budget.start, lists undatable pool ledgers as WARN |
| Back-facing shot shows the front logo | no back ref | `prompts.shot_refs` refuses to build without `refs.back` |
| No PRO 6000 stock | RunPod supply | runner retries ~20 min, then FAILED $0 |

## 8. Budget gate
- `ledger_roots` in the job JSON (supports `$FMB_DATA`, `~` and relative paths). `$FMB_DATA/run` is always included. Run `python3 lib/fmbcfg.py` to print every ledger counted.
- Per-pod ledgers (`{pod: {rate, t0, t_end}}`) are counted exactly for pods started on or after `budget.start`. Old pool ledgers (`{pods: {integ}}, total`) are lifetime totals with no start times, so they're listed as WARN and never counted.

## 9. Self-check
`qc/selfcheck.py pre` / `post --clip X [--live]` prints a markdown PASS/FAIL table with evidence and exits non-zero on any FAIL. The per-piece "eyeball vs masks" row stays FAIL until you watch the side-by-side and overlay and pass `--eyeballed`.

## 10. Image / boot
`code/image/` holds the worker image recipe (ComfyUI 0.42.5, pinned nodes, h3_exact_lora, boot.sh fetching manifest sets with sha256 checks). The default image tag is `ghcr.io/ayoomeco/genie-jutsu-worker:fastboot-comfy-0.42.5-r2`. If that image is private for you, build `code/image/Dockerfile`, push it, and set `GJ_IMAGE` (and `GJ_REGISTRY_AUTH_ID` if your registry is private).

## GPU image access
Both images are PUBLIC on GHCR, so anyone can pull them with no token (checked 2026-10-29: anonymous tag list and manifest both work).
- Current fastboot: `ghcr.io/ayoomeco/genie-jutsu-worker:fastboot-comfy-0.42.5-r2` (rollbacks `fastboot-comfy-0.42.5`, `fastboot-v0.39.1`).
- Older Genie Jutsu image: `ghcr.io/ayoomeco/chywgja` (made public Oct 20-21; Sephiroth pins it by digest).
- `GJ_REGISTRY_AUTH_ID` and `image/Dockerfile` are only for someone who wants their own private build.

## 11. Still true from the Genie Jutsu playbook
**Golden rules**
- No GPU for prep: masks, QC, prompts, quotes, uploads run on the free CPU box. GPU only renders.
- Full frame only; never paste a render back through a mask (halos, clipped silhouettes, luma seams).
- Native HD sources only (1080p+), one continuous take per piece.
- Second opinion (another AI or a fresh check) on any new method or any plan over about $5.
- Testing discipline: give every candidate the same clip/seed/refs, then a face-off of each one's best. Smoke-test unproven steps on a small cap. Tests never change the baseline until the owner picks a winner.

**SCAIL-2 (only with the owner's OK, 3+ people)**
- Wan 2.1 14B SCAIL-2 int8 + SCAIL-2 DPO LoRA 1.0, shift 5.0, 1280x704, euler/simple 20 steps with the speed LoRA unloaded, cfg 1.0, 81-frame chunks (step 76), replacement mode on.
- Inputs: driver, who's-who colour mask video (slot0 blue, slot1 red, slot2 green), refs in on-screen order on black incl. back views. A short "Left: X. Centre: Y. Right: Z." prompt works best.
- Trade-off: keeps people on their roles and follows pose, but less photoreal, can leave a faint halo, slower. Never 20 steps with the speed LoRA loaded (print and check settings before launch).

**Pods and guards**
- Each pod gets a watchdog (~97% of its cap), a dead-man timer and a 15-min idle watchdog; the job guard counts only its own pods (rate x time, never balance delta). Never touch another job's pods.
- 20-minute rule: no card in about 20 min = FAILED $0 or switch region.
- Env names only: RUNPOD_API_KEY, HF_TOKEN, DROPBOX_TOKEN.

**Other lessons**
- H3 has no pose lock: watch back views for faces popping onto the back of heads.
- A smiling head ref makes the swap smile: neutral, closed mouth.
- Listing example props in a prompt makes the model add them.
- Old loader (ComfyUI 0.39.1 + chained LoraLoaderModelOnly) silently dropped LoRA deltas; fixed by 0.42.5 + the exact loader. Keep the load check.
- Rejected for speed: w4a8, Sage3, EasyCache, RadialAttn, STA, torch.compile.
- Licence: recheck the H3 community licence for your region before commercial use or sharing.

# Part B: Standing rules and self-check

There's no audit bot. I check my own work with the two checklists below and show the evidence every time: real numbers, file names and frame checks, not "looks good".

## Money
1. **Nothing paid runs without your yes** to a quote with a cap. The quote line reads: "About $X (cap $Y), about N min."
2. **Quote first, with a "what's different from last time" table** showing every setting with its old and new values: GPU, model, size, steps, sampler, shift, seed, each LoRA and its strength, ref count/order/size, ref_image_size, frames per piece, prompt structure, and any extra pass.
3. **Never change a variable silently.** Anything off the baseline gets named and approved, or it gets fixed instead of launched.
4. **The budget gate** adds up every ledger since the budget started, plus the new caps. If the total goes over, nothing launches.
5. **Reroll only when needed:** for an error or a clearly bad piece, at most 2 new seeds, inside the cap. Graph or validation errors get fixed, not rerolled.
6. **Terminate every pod and confirm 0 are left.** Report the real spend against the cap.

## Live status
7. You get status lines at each stage: **LAUNCHED → DOWNLOADING → READY → FIRST OUTPUT → PIECE n/N → DONE**, or **FAILED + the reason** at any point.
8. I check the provider (RunPod) directly. I never say a run is going until the pod is confirmed live.

## Craft
9. **Per-shot prompts.** Every shot gets its own prompt. Back views have no face words.
10. **The background color goes in the generation prompt** ("solid pure black background"), never matted on afterward.
11. **Clips keep the source fps and the source audio.** If a deliverable will be silent, I flag it up front.
12. Refs are separate images (never a sheet), flat 8-bit RGB with no alpha, 1536 long side. Faces come from real photos whenever possible.
13. If a character is seen from behind, they get a back-view ref of that outfit.
14. Only consenting people or original characters.

## Delivery
15. **Phone-friendly share links only** (Dropbox view links). No folder paths.
16. **A job record for every job:** a README (what, why, recipe, spend, problems, results), a recipe/ folder with the scripts and configs, and one line in JOB_INDEX.md.
17. Results are private. I never post or share them for you.
18. Keys only through the secure secret card, read by name. Never printed or written down.

## Pre-launch self-check ($0, shown with evidence before asking for a yes)
Run `python3 qc/selfcheck.py pre` and paste its table; the boxes below are what it checks plus what only eyes can check.
- [ ] Source: native HD, cuts detected and checked by eye, contact sheet made, the window avoids crowd ID chaos.
- [ ] Pieces are ≤141 frames, never across a hard cut, source fps kept, driver muted.
- [ ] Refs: listed with size/mode (prep check: RGB, no alpha, 1536 long side). Back ref present if anyone's seen from behind.
- [ ] Masks (multi-person): who's-who overlay built and looked at; every subject has a position in the prompt.
- [ ] Prompts: every ref role named; "match the source's mouth and expression exactly"; "keep any objects already in the scene; add nothing"; no face words on back views. Source poses the swap must NOT copy are flagged.
- [ ] Built graphs printed and diffed against the baseline (steps 8, euler/simple, shift 6, seed 77, LoRAs, H3LoadReport expect = layer counts).
- [ ] Banned steps absent: ResCascade, Fizgig, SeedVR2 video.
- [ ] Dry run passes: budget gate ok, no missing inputs, refuses without --go.
- [ ] The what's-different table and quote are sent, and you said yes.

## Post-render self-check (before you see anything)
Run `python3 qc/selfcheck.py post --clip <id> --live` and paste its table; then do the eyeball rows.
- [ ] Specs: frame count = planned (trimmed), fps = source, audio length = video length, the file plays.
- [ ] Per piece, checked against the masks: right character in each spot, no leftover original person, no ID swap, no extra heads, no flat ref frames, no missing limbs, the outfit right from every side, mouth matches.
- [ ] Not black, not washed out (luma check).
- [ ] Contact sheet with timestamps; every flag named honestly with its time.
- [ ] Pods 0, spend vs cap reported.
- [ ] A PASS or FAIL verdict per clip with the reasons, then links, then the job record.
