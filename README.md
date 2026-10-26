# GJ V2

Genie Jutsu V2. Video swap on MiniMax H3 reference-to-video, ComfyUI, rented GPU pods. No keys and no model weights in this repo: keys come from the environment only.

Install from the `gj-v2` branch. See [INSTALL.md](INSTALL.md). The Farquad bot stays on `main`. Setup does not rent a GPU and does not spend.

```bash
git clone -b gj-v2 https://github.com/ayoomeco/farquad-v2
cd farquad-v2
bash setup.sh
```

A good setup ends with:

```text
GJ_V2_SETUP_OK FARQUAD_SETUP_OK  (activate with: . $GJ_HOME/venv/bin/activate)
```

The install root is `GJ_HOME`, else `FARQUAD_HOME`, else the directory that contains `setup.sh`.

## What changed from the old default

1. ComfyUI is 0.42.5 or newer, with the vendored `h3_exact_lora` loader. Every render must pass the load check (lossy merges 0, hooks present, each LoRA at its set strength) or it fails.
2. One speed LoRA only: lightx2v Minimax-h3-Turbo 8-step v1.1 at 0.8. The old turbo LoRA is not loaded beside it, and `pro4500` is not on the default stack.
3. The personal likeness (Meek) LoRA is off. It is not downloaded with the default set. A strength of 0.35 or above is refused.
4. 8 steps, video shift 6, audio shift 3, euler / simple, seed 42, `ref_image_size` match, Sage Attention on.
5. The prompt names image 1 face, image 2 body, image 3 outfit, and includes "Match the source's mouth and expression exactly." and "keep any objects already in the scene; add nothing."
6. References are separate images (neutral closed-mouth head, body only on a full-body shot, outfit last). SeedVR2 image mode upscales crops to a 2048 short side. No combined character sheet, no face restore.
7. Post is on the pod: upscale only, source frame rate, no interpolation, no colour grade, no grain, no face restore, no paste-back. Encode is H.264 High 4.2, yuv420p, bt709, AAC, +faststart. Stitch, score, and encode finish before the pod is terminated.

The full old-to-new list is in [CHANGELOG.md](CHANGELOG.md).

Default stack: Singularity H3 Ref2VA pruned v1.3 int8, realism 0.7, character-swap v1.5 at 1.0, lightx2v 8-step v1.1 at 0.8. Pieces stay 141 frames, step 119, 22-frame AddGuide, 24-frame warm-up dropped at assembly, 1152x640. Natural mouth: the graph has no audio drive; the source audio is muxed back at encode.

## Environment variables

| Var | Needed | What |
|---|---|---|
| `GJ_HOME` | optional | Install root. Wins over `FARQUAD_HOME`. |
| `FARQUAD_HOME` | optional | Install root when `GJ_HOME` is unset. |
| `RUNPOD_API_KEY` | yes, for a paid render | Environment only. Never print it, never write it to a file. Setup does not read it. |
| `HF_TOKEN` | optional | Hugging Face downloads on the pod. Public weights do not need it. |
| `DROPBOX_TOKEN` | optional | Only for `dbx/upall.py` token mode. |
| `DEST_DIR` | optional | Passed to `upall.py` as its first argument. |

Tuning knobs (all optional): `TRIES` (max 5; the rule is 2 tries total, so set `TRIES=2`), `ORIG_MAX` (3), `POP_MAX` (2), `POP_SPIKE` (3.0), `POP_HOLD` (4), `IDLE_MIN` (15), `RIDTAG`, `GRAPHS` (default `graphs_v5b`), `BENCH_GPU` (any24), `BENCH_CLOUD` (COMMUNITY), `BENCH_ORDER` (COMMUNITY_FIRST), `LINGER` (600), `ATTACH` (reuse a running pod id), `ASSEMBLE`, `MODE` (fit|fill), `ACTIVE` (y0,y1,x0,x1), `DRY=1` (tests), `GJ_LIKENESS_LORA` / `GJ_LIKENESS_STRENGTH` (off unless set; 0.35 and above is refused), `POD_STATUS_URL` and `JUTSU_FIX_NONCE` (or `plan/run_<slug>/pod.json`) for the on-pod post.

## Layout

```
$GJ_HOME/
  setup.sh  INSTALL.md  mkwu.sh  premask.py
  plan/        driver_v5b.py driver_v5b3.py graphs_v5b.py refs_gjv2.py prep_v5b.py launch_v5b.py
               qc_v5b3.py test_gj_v2.py finish_v5b3.py idle_watchdog.py assemble_v5b.py
               base_graphs/   graphs/   run_<slug>/
  lib/         home.py fx.py pods.py
  engine/      worker/install.sh (ComfyUI 0.42.5+)  worker/custom_nodes/h3_exact_lora/
               dsr/genie_dsr.toml  config/models.manifest.json
  polish/pod/  post_gjv2.py  (s3.py and s4.py refuse the old interpolation and grade)
  models/      yunet, sface, yolo11l-seg (setup.sh, sha256-pinned)
```

Reference files are `engine/data/bakeoff/clips/v5b_<key>_face.png`, `_body.png` (full-body shots), and `_outfit.png`.

## Free dry-runs

No GPU and no spend. Setup already runs the first two.

```bash
DRYCHECK=1 DSR_RUN=/tmp python plan/driver_v5b3.py
DRY=1 DRY_STATE='{"pods":[],"bal":10}' python plan/idle_watchdog.py
python polish/pod/post_gjv2.py --plan
python plan/test_gj_v2.py
```

A quote that does not rent a pod:

```bash
cd engine
FP_RUN_DIR=/tmp/gj-dry FP_GROUP=gjfast JUTSU_CONFIG=$PWD/dsr/genie_dsr.toml \
  python scripts/phase1_bakeoff.py plan --refine --fullperson --gpu any24 --cloud COMMUNITY_FIRST --cap 6
```

`launch` spends money. Run it only after an explicit per-job cap yes. The engine refuses caps above `hard_cap_usd` in `engine/dsr/genie_dsr.toml` (6.00).

## Run order

One clip is one slug and one card.

1. Warm-up and native trim: `bash $GJ_HOME/mkwu.sh <source.mp4> <start s> <dur s> $GJ_HOME/<slug> <slug> <tag>`
2. Save separate refs as `v5b_<key>_face.png`, `_body.png` when the shot is full-body, and `_outfit.png`.
3. Pre-mask (check maps only, never pasted): `cd $GJ_HOME && python premask.py <slug>`
4. Prep: `python $GJ_HOME/plan/prep_v5b.py <slug>`
5. Quote, then wait for an explicit yes before launch.
6. After that yes: `python $GJ_HOME/plan/launch_v5b.py <cap> <tag> $GJ_HOME/plan/run_<slug>`, then `DSR_RUN=$GJ_HOME/plan/run_<slug> TRIES=2 python $GJ_HOME/plan/driver_v5b3.py <slug>`, plus keepalive and `python $GJ_HOME/plan/idle_watchdog.py`.
7. On DRIVER_DONE the finisher runs `polish/pod/post_gjv2.py` on the pod (upscale, stitch, score, encode at the source fps), then terminates that pod. If the pod status channel is missing, it stitches pieces already on disk with the same encode flags and no grade. Confirm 0 pods afterwards: `python $GJ_HOME/lib/pods.py list`.
8. Deliver with `python $GJ_HOME/dbx/upall.py "$DEST_DIR" <files...>` when `DROPBOX_TOKEN` is set.

Stop a run by hand: `touch plan/run_<slug>/STOP`. Stop the watchdog: `touch plan/WATCHDOG_OFF`.

## Licensing

The MiniMax H3 models are under the MiniMax H3 Community License, and US use needs a separate written authorization from MiniMax. `engine/dsr/genie_dsr.toml` still has `license_authorized = true` from the original setup: get your own authorization and fill in the `license_*` fields before the first paid render. Check each LoRA's license on its Hugging Face page too.
