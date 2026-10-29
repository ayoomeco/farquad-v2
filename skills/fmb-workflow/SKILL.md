---
name: fmb-workflow
description: >-
  Use for every Fire Marshal Bill job: putting a consenting person or original
  character into a clip with H3 reference-to-video, end to end — refs, quote,
  launch with live status, self-check, delivery and the job record.
---
# The job, end to end

Read RULES.md and KNOWLEDGE_BASE.md first. Code is in `$FMB_HOME` (see `fmb-install`). One job = one `jobs/<job>.json`: copy `example_job.json` and set the characters, clips, shots, caps, rate, `budget.start`/`total` and `ledger_roots`. Export `FMB_JOB` and `FMB_DATA`.

## Baseline (owner-approved; any change goes in the what's-different table and needs a yes)
| Setting | Value |
|---|---|
| GPU | RunPod RTX PRO 6000 Blackwell **SECURE**, ~$2.49/hr, one pod per clip (stills: RTX A6000 SECURE ~$0.59/hr) |
| Model | Singularity **H3 v1.3 int8**, ComfyUI 0.42.5, exact-LoRA loader, load check lossy 0 with `expect` = layer counts |
| LoRAs | realism 0.7 if every character in the shot looks human, else 0; character swap v1.5 1.0; lightx2v 8-step v1.1 0.8 (one speed LoRA) |
| Sampler | 8 steps, euler / simple, shift 6 |
| Seed | **77**; rerolls 78, 79 only for a bad or errored piece (never for graph errors) |
| Refs | separate images, head front + body front (+ body back when seen from behind), **1536 long side, flat 8-bit RGB, ref_image_size max**, ≤9 per shot |
| Size | 1152x640 for 16:9 (640x1152 vertical); 4:3 sources keep 4:3 at about the same pixels (960x768, 1024x768) |
| Frames | ≤141 per piece, never across a hard cut, 22-frame guide chains |
| Video | **source fps**, driver muted, **native res is the deliverable** (no SeedVR2/ResCascade/Fizgig) |
| Audio | **source audio muxed back**, trimmed to the planned frames; silent deliverables are flagged up front |

## 1. Intake ($0)
- Get the clip and who becomes who. Use the best native HD copy. Split on the original cuts, check them by eye, make a contact sheet. Note per shot: people count, framing and facing (front/back), and each subject's screen side.
- Pick a window **without crowd ID chaos**, and with source poses you want copied. The swap copies the source arms, whatever the prompt says.
- Cut per-piece drivers (muted, source fps). Write the clip and shots into the job JSON with an audio block `{src, start}`.

## 2. Refs
- Per character: head front (neutral, mouth closed), full body front in the outfit, and **full body back in the same outfit if they're ever seen from behind**. The builder refuses a back-facing shot without `refs.back`.
- Real photos first: `python3 video/prep_refs.py` (Lanczos to 1536 long side, flat RGB, no AI; flags anything smaller).
- Missing views → the ref job (`ref_job` in the JSON): `refs/graph_refs.py` then `refs/run_refs.py` (dry run). Whites from real photos with the texfix VAE, edits to black with the background in the prompt, and the angle LoRA 0.9 with no face words on backs. It's paid (A6000), so quote it separately. Size + blank-canvas checks stop a stage on the first miss.

## 3. Masks (multi-person only, $0)
`python3 video/build_masks.py <clip>`, then watch `overlay_*.mp4` and read `masks_report.json` (missing frames, extra people). H3 has no mask input: use the maps to write each subject's clothing + position into the shot, and to check pieces afterward. With 3+ characters in one shot, warn about head swaps, and offer SCAIL-2 only with the owner's OK.

## 4. Build, self-check, quote ($0)
```bash
python3 video/prep_audio.py && python3 video/graph_video.py
python3 video/run_video.py --clip <id>          # BUILT, NOT RUN + gate + missing inputs
python3 qc/selfcheck.py pre --clip <id>          # paste this table
```
- Write `$FMB_DATA/<job>/QUOTE.md`: the line "About $X (cap $Y), about N min", a `| Setting | Old | New |` table vs the last run the owner saw, the open risks, and SILENT flags. The quote cap must equal the clip caps.
- Paste the pre-launch table. Fix every FAIL first. **Wait for the owner's yes.**

## 5. Launch with live status (paid)
```bash
(python3 video/status_video.py <id> >> status_relay_<id>.out 2>&1 &) && (python3 video/run_video.py --clip <id> --go > run_<id>.out 2>&1 &)
```
- Confirm the pod on RunPod before saying it's live. Relay status lines on stage changes only: LAUNCHED / DOWNLOADING / READY / FIRST OUTPUT / PIECE n/N / ASSEMBLY / TRIM + AUDIO / DONE / FAILED + reason / TERMINATED.
- On FAILED: read `fail_*.log`. Graph errors get fixed (a pure bug fix inside the same cap, told to the owner), never rerolled. Any setting change → re-quote.

## 6. Post-render self-check ($0)
`python3 qc/selfcheck.py post --clip <id> --live` checks frames = plan, fps, size, audio synced, luma, frame-still size/aspect/blank, spend ≤ cap, and 0 pods (runner + RunPod). Then watch the side-by-side + mask overlay per piece (right person, no leftover, no ID swap, limbs, outfit from the back) and rerun with `--eyeballed`. Give a verdict per clip: PASS or FAIL with timestamps.

## 7. Deliver
`DROPBOX_TOKEN=… python3 dbx/upall.py /<owner folder>/<job>_<date> <files>` (a temporary link + one curl, never chunked), or the owner's connector. Send **phone-friendly share links**: the final, the side-by-side (say which side is which), the QC sheet, the self-check tables, spend vs cap, and flags.

## 8. Job record
In the job folder: README.md (what and why in the owner's words, recipe, what's different, a spend table from the ledgers, problems + fixes, results, file map, open decisions), recipe/ (job JSON, built graphs, plan/end/times/post JSON, status + fail logs, self-check tables; no keys), and one line in JOB_INDEX.md. Use the shared `job-record` skill.
