# Farquad V2

Video swap pipeline code for the Farquad V2 Grok Bot template. No keys, no model weights: keys come from env vars only. Install: `bash setup.sh` (set FARQUAD_HOME, default ~/farquad).

## What this is
Farquad swaps a person in a video clip for a new character with the V1 "GJ-Fast linked" recipe:
Singularity H3 Ref2VA + realism LoRA 0.7 (live-action clips only) + character-swap v1.5 + pro4500 + turbo 0.8, Sage attention,
6 steps, 1152x640, linked 141-frame pieces (step 119, 22-frame AddGuide), a 24-frame warm-up that is dropped at assembly,
natural mouth (no audio in the graph; the original audio is added back), full-frame render, never a mask paste-back.
The code lives in these skills as text. Extract it once, run setup, then follow the run order below.

## Parts
- `farquad-scripts-readme` (this): overview, env vars, run order, `extract.py`.
- `farquad-scripts-render`: card driver + graph builder + prep + launcher + keepalive + warm-up maker + pre-mask + face/pod helpers + the two base ComfyUI graphs.
- `farquad-scripts-checks-finish`: pop-in check, contact sheet, finisher (terminate + verify 0 pods + stitch), idle/cap watchdog, legacy guard, assembler.
- `farquad-scripts-setup-polish`: `setup.sh`, the polish-pod scripts (SeedVR2 -> RIFE -> LAB match + grain), the Dropbox uploader.
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
`BENCH_ORDER` (COMMUNITY_FIRST), `LINGER` (600), `ATTACH` (reuse a running pod id), `ASSEMBLE`, `MODE` (fit|fill), `ACTIVE` (y0,y1,x0,x1), `DRY=1` (tests).

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
  polish/pod/  setup.sh agent.py s1.sh s3.py s4.py
  dbx/upall.py
  models/      yunet, sface, yolo11l-seg (downloaded by setup.sh, sha256-pinned)
  <slug>/      <slug>_<tag>_native.mp4, <slug>_wu.mp4, premask_<slug>.npz, final/
```

## Per-clip edits (the recipe keeps clip specifics in code)
- `plan/graphs_v5b.py`: add the slug to `CLIPS` (`look`, `subj` = list of (who in the source, who they become, ref key), `keep`) and to `REAL`
  (True = live-action clip -> realism LoRA 0.7). The five V1 entries are worked examples. Lead plus at most one other person.
- Reference sheets: `engine/data/bakeoff/clips/v5b_<ref key>.png` (one per subject, made in Higgsfield and approved by the owner).
- `premask.py`: add the slug to `CAST` (role names, lead first). `plan/driver_v5b.py`: `NOMASK` for slugs whose checks run unmasked.
  `plan/assemble_v5b.py`: B&W sources go in the `BW` tuple.

## Run order (one clip = one slug = one card; run clips in parallel, one card each)
1. Source: native HD, one continuous take. Warm-up + native trim:
   `bash $FARQUAD_HOME/mkwu.sh <source.mp4> <start s> <dur s> $FARQUAD_HOME/<slug> <slug> <tag>`
2. Sheets in Higgsfield -> owner approves -> save as `v5b_<key>.png` (see above).
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
7. Automatic: per-piece checks (original-gone, bg drift, ref leak, pop-in) with targeted re-rolls, keep best; on DRIVER_DONE the finisher
   terminates this run's pod, verifies 0 remain, then stitches -> `<slug>/final/<slug>_v1_final.mp4` + `_v1_side_by_side.mp4` (original audio).
   Always confirm 0 pods yourself afterwards: `python $FARQUAD_HOME/lib/pods.py list`. Never terminate pods that belong to another job.
8. Polish (default, needs its own cap OK): one GPU pod running `polish/pod/agent.py` (see farquad-scripts-setup-polish).
9. Deliver: `python $FARQUAD_HOME/dbx/upall.py "$DEST_DIR" <files...>` (token mode) and send the share links + cost + any flags.
Stop a run by hand: `touch plan/run_<slug>/STOP` (driver/keepalive), `touch plan/WATCHDOG_OFF` (watchdog).

## Not included (and why)
- Model weights: the render pod pulls the prebuilt public image pinned in `plan/launch_v5b.py` (`FP_IMAGE`, about 100 GB, models baked in);
  `engine/config/models.manifest.json` lists the public Hugging Face URLs for the gjfast models. CPU models are fetched by `setup.sh`.
- The image itself is binary and lives in its original author's public GHCR namespace; to own it, rebuild from `engine/worker/` + the manifest.
- Polish-pod model weights (SeedVR2 7B, RIFE 4.26) download on the pod in `polish/pod/setup.sh` / on first run.
- No box-side polish launcher existed in V1 (the polish pod was driven by hand through agent.py), so the steps are written out instead of scripted.
- CodeFormer stage `s2.py` is left out on purpose: no face restore.

## Licensing
The MiniMax H3 models are under the MiniMax H3 Community License, and US use needs a separate written authorization from MiniMax. `engine/dsr/genie_dsr.toml` still has `license_authorized = true` from the original setup: get your own authorization and fill in the `license_*` fields before your first paid render. Check each LoRA's license on its Hugging Face page too.

## Polish pass (default after the final stitch; one GPU pod inside the job's approved cap)
Order: SeedVR2 7B -> 1080p, then RIFE 4.26 (30 fps default, 60 fps option), then LAB colour match to the source (~0.8, drop toward 0.5
if skin goes too dark/red) + fine grain. No face restore. Inputs must be 1920x1080 24 fps for s3/s4 (mkwu.sh makes the native at 1920x1080).
1. Get the owner's OK + cap. Rent one GPU pod (48 GB class, e.g. A6000) on a PyTorch CUDA image with HTTP port 8000 exposed, then on the pod:
   `mkdir -p /w && PSK=<long random> python3 /w/agent.py &` (copy agent.py in with the pod's own file tools first).
2. From the box (header `X-P: $PSK`): `PUT /f/<name>` uploads to /w/<name>, `POST /` with a bash command body runs it in /w (returns a log id,
   log at /w/logs/<id>.log ending in `EXIT=<code>`), `GET /f/<name>` downloads. Upload setup.sh, s1.sh, s3.py, s4.py,
   `in.mp4` (the stitched final) and `src.mp4` (the native source).
3. Run in order: `bash setup.sh` -> `bash s1.sh` (writes s1_raw.mp4 and prints width,height,frames) -> trim extra leading frames so the frame
   count matches the final, e.g. `ffmpeg -y -i s1_raw.mp4 -vf "select=gte(n\,<extra>),setpts=PTS-STARTPTS" -c:v libx264 -crf 10 -pix_fmt yuv420p s1.mp4`
   -> `python3 s3.py /w/s1.mp4 /w/s3_30.mp4 30` -> `python3 s4.py /w/s3_30.mp4 /w/src.mp4 /w/s4_30.mp4 30`.
4. Download s4_30.mp4, add the original audio back: `ffmpeg -i s4_30.mp4 -i <slug>_v1_final.mp4 -map 0:v -map 1:a -c copy <slug>_polished.mp4`.
5. Terminate the polish pod and verify 0 of this job's pods remain (`python $FARQUAD_HOME/lib/pods.py list`).
polish/pod/setup.sh also clones CodeFormer (only used by the skipped s2 stage); it is kept as it was in V1.

## Delivery
`python $FARQUAD_HOME/dbx/upall.py "$DEST_DIR" <file> [...]` with `DROPBOX_TOKEN` in the env -> one JSON line per file with the share link.
`--links` mode is the original V1 loop for temporary upload links handed out by a Dropbox connector.

