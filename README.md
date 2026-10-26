# Farquad V2

Video swap pipeline code for the Farquad V2 Grok Bot template. No keys, no model weights: keys come from env vars only. Install: `bash setup.sh` (set FARQUAD_HOME, default ~/farquad).

## What this is
Farquad swaps a person in a video clip for a new character with the A08 default:
Singularity H3 v1.3 trimmed int8, realism LoRA 0.7, character-swap v1.5 at 1.0, lightx2v Minimax-h3-Turbo 8-step v1.1 at 0.8.
No personal likeness LoRA unless `LIKENESS_LORA` is set and `LIKENESS_STRENGTH` is not 0.
ComfyUI 0.42.5 or newer, the vendored `h3_exact_lora` node (one exact stack, not chained stock loaders), and a load check that
aborts the render unless lossy merges are 0, every hook is present, and each LoRA is at its set strength.
Sage attention on, 8 steps, euler / simple, shift 6 (video) / 3 (audio), seed 42, `ref_image_size` match, canvas 1152x640.
Reference stills are separate images in order: head close-up, body shot only for full-body shots, outfit last.
Linked 141-frame pieces (step 119, 22-frame AddGuide), a 24-frame warm-up that is dropped at assembly,
natural mouth (no audio in the graph; the original audio is added back at the source frame rate).
Post is upscale only (SeedVR2, no colour correction), then H.264 High 4.2, yuv420p, bt709, AAC, +faststart.
No frame interpolation, no colour grade, no grain, no face restore, no paste-back.
Stitching, scoring, and the final encode run on the render pod before it is terminated.
The code lives in these skills as text. Extract it once, run setup, then follow the run order below.

## Parts
- `farquad-scripts-readme` (this): overview, env vars, run order, `extract.py`.
- `farquad-scripts-render`: card driver + graph builder + prep + launcher + keepalive + warm-up maker + pre-mask + face/pod helpers + the two base ComfyUI graphs.
- `farquad-scripts-checks-finish`: pop-in check, contact sheet, finisher (terminate + verify 0 pods + stitch), idle/cap watchdog, legacy guard, assembler.
- `farquad-scripts-setup-polish`: `setup.sh`, the polish-pod scripts (SeedVR2 upscale, then the A08 encode), the Dropbox uploader.
- `farquad-engine-01` ... `farquad-engine-NN`: the RunPod/ComfyUI runner the launcher drives (pod create, boot payload, queue, watchdog, terminate). Big files are split into `(chunk i/n)` blocks across these parts.

## Install
1. Save every Farquad script skill body (all `farquad-scripts-*` and `farquad-engine-*`) to `.md` files in one folder.
2. Write the `extract.py` block below to a file, then: `python3 extract.py "${FARQUAD_HOME:-$HOME/farquad}" *.md`
   (it joins chunks in order, keeps bytes exact, and makes `.sh` files executable).
3. `bash "${FARQUAD_HOME:-$HOME/farquad}/setup.sh"` -> must end with `FARQUAD_SETUP_OK`. Then `. $FARQUAD_HOME/venv/bin/activate` before every run.

## Environment variables
| Var | Needed | What |
|---|---|---|
| `FARQUAD_HOME` | optional | Install root, default `~/farquad`. Everything (plan/, lib/, engine/, models/, clip folders) lives under it. |
| `RUNPOD_API_KEY` | yes, for renders | Read from the environment only. Never print it, never write it to a file. |
| `DROPBOX_TOKEN` | optional | Only for `dbx/upall.py` token mode. Env only. Without it, deliver through your own storage connector. |
| `DEST_DIR` | optional | Where deliverables go (e.g. a Dropbox folder `/Farquad/<job>`); passed to `upall.py` as its first argument. |
| Higgsfield | via connector | Character sheets are made in the Higgsfield connector, not by these scripts. |
| `PSK` | polish pod | Shared secret for `polish/pod/agent.py` (pick a long random value per pod). |
Tuning knobs (all optional): `TRIES` (max 5; the rule is 2 tries total -> set `TRIES=2`), `ORIG_MAX` (3), `POP_MAX` (2), `POP_SPIKE` (3.0),
`POP_HOLD` (4), `IDLE_MIN` (15), `RIDTAG`, `GRAPHS` (graph module, default `graphs_v5b`), `BENCH_GPU` (any24), `BENCH_CLOUD` (COMMUNITY),
`BENCH_ORDER` (COMMUNITY_FIRST), `LINGER` (1800, so the pod stays up for the on-pod stitch and encode), `ATTACH` (reuse a running pod id), `POST_WAIT` (1500), `POST_LOCAL=1` (run that post on the box), `A08_UPSCALE` (`seedvr` default, `skip` for tests), `LIKENESS_LORA` / `LIKENESS_STRENGTH` (optional, off when unset or strength 0), `ASSEMBLE`, `MODE` (fit|fill), `ACTIVE` (y0,y1,x0,x1), `DRY=1` (tests).

## Layout after extract
```
$FARQUAD_HOME/
  setup.sh  extract.py  mkwu.sh  premask.py
  plan/        driver_v5b.py driver_v5b3.py graphs_v5b.py prep_v5b.py launch_v5b.py keepalive_v5b.py
               qc_v5b3.py qc.py test_qc_v5b3.py finish_v5b3.py idle_watchdog.py guard_v5b.py assemble_v5b.py
               base_graphs/ (2 ComfyUI API graphs)   graphs/ (generated)   run_<slug>/ (one per clip)
  lib/         fx.py (YuNet + SFace faces)   pods.py (RunPod list/terminate, no keys printed)
  engine/      scripts/phase1_bakeoff.py + app/ + worker/ + dsr/genie_dsr.toml + config/models.manifest.json
               data/bakeoff/clips/ (pieces + reference sheets uploaded to the pod)
  polish/pod/  setup.sh agent.py post.sh s1.sh s3.py s4.py
  dbx/upall.py
  models/      yunet, sface, yolo11l-seg (downloaded by setup.sh, sha256-pinned)
  <slug>/      <slug>_<tag>_native.mp4, <slug>_wu.mp4, premask_<slug>.npz, final/
```

## Per-clip edits (the recipe keeps clip specifics in code)
- `plan/graphs_v5b.py`: add the slug to `CLIPS` (`look`, `subj` = list of (who in the source, who they become, ref key), `keep`, `full_body`).
  Realism 0.7 is always on. The five entries are worked examples and are full-body, so each subject gets face, body, then outfit. Lead plus at most one other person.
- Reference stills: `engine/data/bakeoff/clips/v5b_<ref key>_face.png`, `_body.png` only when `full_body` is set, `_outfit.png` last
  (made in Higgsfield and approved by the owner). Face plus outfit is a valid pair when the shot is not full-body.
- `premask.py`: add the slug to `CAST` (role names, lead first). `plan/driver_v5b.py`: `NOMASK` for slugs whose checks run unmasked.
  Assembly keeps source letterbox bars and does not grade the picture.

## Run order (one clip = one slug = one card; run clips in parallel, one card each)
1. Source: native HD, one continuous take. Warm-up + native trim:
   `bash $FARQUAD_HOME/mkwu.sh <source.mp4> <start s> <dur s> $FARQUAD_HOME/<slug> <slug> <tag>`
2. Stills in Higgsfield -> owner approves -> save as `v5b_<key>_face.png`, optional `_body.png`, and `_outfit.png` (see above).
3. Pre-mask (check maps only, never pasted): `cd $FARQUAD_HOME && python premask.py <slug>` -> look at `<slug>/<slug>_numbered_board.jpg`.
4. Prep: `python $FARQUAD_HOME/plan/prep_v5b.py <slug>` (writes `plan/run_<slug>/fp_runs.json` + piece 0).
   Free quote: `cd $FARQUAD_HOME/engine && FP_RUN_DIR=$FARQUAD_HOME/plan/run_<slug> FP_GROUP=gjfast JUTSU_CONFIG=$PWD/dsr/genie_dsr.toml python scripts/phase1_bakeoff.py plan --refine --fullperson --gpu any24 --cloud COMMUNITY_FIRST --cap <cap>`
5. STOP: quote one price and a per-job cap to the owner and wait for an explicit yes. The launcher types the engine's `SPEND <cap>`
   confirmation for you, so it must only ever run after that yes. The engine refuses caps above `hard_cap_usd` in `engine/dsr/genie_dsr.toml` (6.00).
6. Launch + drive (run all three, in the background, per clip; the watchdog once per box):
   `python $FARQUAD_HOME/plan/launch_v5b.py <cap> <tag> $FARQUAD_HOME/plan/run_<slug>` (writes CAP.txt + B0.txt, creates the pod)
   `DSR_RUN=$FARQUAD_HOME/plan/run_<slug> TRIES=2 python $FARQUAD_HOME/plan/driver_v5b3.py <slug>`
   `python $FARQUAD_HOME/plan/keepalive_v5b.py $FARQUAD_HOME/plan/run_<slug>`
   `python $FARQUAD_HOME/plan/idle_watchdog.py` (kills pods idle 15+ min, enforces 95% of CAP.txt)
7. Automatic: per-piece checks (original-gone, bg drift, ref leak, pop-in) with targeted re-rolls, keep best. On DRIVER_DONE the finisher
   keeps the pod in linger, runs stitch + SeedVR2 upscale + the H.264 encode on the pod, downloads
   `<slug>/final/<slug>_v1_final.mp4` and `_v1_side_by_side.mp4` (original audio, source frame rate), then terminates this run's pod and verifies 0 remain.
   A failed load check or a failed post stops the run instead of falling back to a box stitch.
   Always confirm 0 pods yourself afterwards: `python $FARQUAD_HOME/lib/pods.py list`. Never terminate pods that belong to another job.
8. The render pod already upscaled and encoded. A second polish pod is not part of the default. `polish/pod/post.sh` is the same upscale-then-encode path if you drive a pod by hand.
9. Deliver: `python $FARQUAD_HOME/dbx/upall.py "$DEST_DIR" <files...>` (token mode) and send the share links + cost + any flags.
Stop a run by hand: `touch plan/run_<slug>/STOP` (driver/keepalive), `touch plan/WATCHDOG_OFF` (watchdog).

## Not included (and why)
- Model weights: the render pod pulls the prebuilt public image pinned in `plan/launch_v5b.py` (`FP_IMAGE`, about 100 GB, models baked in);
  `engine/config/models.manifest.json` lists the public Hugging Face URLs for the gjfast models. CPU models are fetched by `setup.sh`.
- The image itself is binary and lives in its original author's public GHCR namespace; to own it, rebuild from `engine/worker/` + the manifest.
- Polish-pod model weights (SeedVR2 7B) download on the pod in `polish/pod/setup.sh` / on first run. The render pod's post expects that CLI (`SEEDVR_CLI` or `/w/sv/inference_cli.py`) and fails if it is missing.
- No box-side polish launcher existed in V1 (the polish pod was driven by hand through agent.py), so the steps are written out instead of scripted.
- CodeFormer and RIFE are not installed by `polish/pod/setup.sh`. `s3.py` and `s4.py` exit unless `ALLOW_RIFE=1` or `ALLOW_GRADE=1`.

## Licensing
The MiniMax H3 models are under the MiniMax H3 Community License, and US use needs a separate written authorization from MiniMax. `engine/dsr/genie_dsr.toml` still has `license_authorized = true` from the original setup: get your own authorization and fill in the `license_*` fields before your first paid render. Check each LoRA's license on its Hugging Face page too.

## Post (default, on the render pod after the pieces are accepted)
Upscale with SeedVR2 only (no colour-correction flag), keep the source frame rate, then encode H.264 High profile level 4.2,
yuv420p, bt709, AAC, `+faststart`. The 22-frame overlap crossfade is stitch blending. Source letterbox bars are kept; the picture
is the full render. No frame interpolation, no colour grade, no grain, no face restore, no paste-back.
The prompt names each still (`image 1` face, `image 2` body on full-body shots, outfit last) and includes
"keep any objects already in the scene; add nothing" and "match the source's mouth and expression exactly".
1. The finisher uploads the native audio and the accepted pieces, then queues `pod_cmds/90_a08_post.sh`.
2. `engine/worker/post_a08.py` stitches, upscales, encodes, and prints `POST_DONE` only after ffprobe matches that encode.
3. To run the same steps by hand on a polish pod: `bash polish/pod/setup.sh`, then `bash polish/pod/post.sh` with `in.mp4` and `src.mp4` in `/w`.

## Delivery
`python $FARQUAD_HOME/dbx/upall.py "$DEST_DIR" <file> [...]` with `DROPBOX_TOKEN` in the env -> one JSON line per file with the share link.
`--links` mode is the original V1 loop for temporary upload links handed out by a Dropbox connector.

