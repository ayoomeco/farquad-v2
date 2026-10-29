# Fire Marshal Bill — code (branch `fmb-v1` of ayoomeco/farquad-v2)

H3 reference-to-video, plus the character-ref job that feeds it, carrying every Oct 28–29 fix. **Self-contained:** nothing is read from outside this folder and `$FMB_DATA`. No keys in here; keys are read by name from the environment.

## Layout
| Path | What |
|---|---|
| `lib/fmbcfg.py` | Paths, job JSON, ledger roots + budget gate (`python3 lib/fmbcfg.py` prints every ledger counted). |
| `lib/graphs.py` | Builds every ComfyUI graph in code: H3 (+22-frame guide), Qwen 2.1 white with texfix decode, 2511 edit (+angle LoRA 0.9), SeedVR2 image. Asserts the recipe (node-81 `expect` = layer counts). |
| `lib/prompts.py` | Per-shot prompt templates from the shot's subjects + the job's characters. Refuses back-facing shots without a back ref and more than 9 refs. |
| `lib/checks.py` | Image size/aspect/blank/RGB check, video probe, luma check, frame stills, audio cut, trim-to-planned-frames + audio re-mux. |
| `lib/gj.py`, `lib/rpod.py`, `lib/assemble.py` | RunPod SECURE launch with cap guard, ledger, watchdog, run_graph (full error text), on-pod assembly. |
| `video/` | `prep_refs.py`, `prep_audio.py`, `graph_video.py`, `run_video.py` (dry run / `--go`), `status_video.py`, `build_masks.py` (YOLO11m-seg + ByteTrack + SAM2.1-tiny). |
| `refs/` | `graph_refs.py`, `run_refs.py` (T whites → R black edits → U upscale, with checks), `status_refs.py`. |
| `qc/selfcheck.py` | `pre` and `post` PASS/FAIL tables to paste to the owner. |
| `demo/` | `make_demo.py` (synthetic media so every dry run builds), `post_demo.py` (exercises trim + re-mux + post-check, no pod). |
| `dbx/upall.py` | Dropbox upload: temporary upload link + one curl POST per file, then a share link. |
| `image/` | Worker image recipe: Dockerfile, boot.sh, fetch_models.py, `models.manifest.json` (HF repo@revision + sha256 for sets h3_a08, seedvr2, qwen21, qwen2511_lite), `custom_nodes.txt` (repo@commit), vendored `nodes/h3_exact_lora`. |
| `jobs/example_job.json` | Neutral worked examples: `one_person` (chained 2 pieces), `two_person_contact` (one subject seen from behind), `group` (3 characters incl. a non-human → realism 0), plus a `ref_job`. |

## Quick start ($0)
```bash
bash setup.sh                                   # ends with FMB_SETUP_OK  (FMB_MASKS=0 skips YOLO/SAM)
. venv/bin/activate
export FMB_DATA=$HOME/fmb-data FMB_JOB=$PWD/jobs/example_job.json
python3 demo/make_demo.py && python3 video/prep_refs.py && python3 video/prep_audio.py && python3 video/graph_video.py
python3 video/run_video.py --clip one_person    # BUILT, NOT RUN ... missing inputs: none
python3 qc/selfcheck.py pre
python3 demo/post_demo.py && python3 qc/selfcheck.py post --clip one_person
```
Paid run, **only after the owner's yes to the quote + cap**: `python3 video/run_video.py --clip <id> --go` (with `status_video.py <id>` beside it).

## Job JSON essentials
`caps` per clip, `rate`, `gpu`, `budget {total, start}`, `ledger_roots`, `seed`, `rerolls`, `characters {Name: {human, look, outfit, refs {head, body, back?}}}`, and `clips {id: {W, H, fps, frames_out, assembly chain|shots, audio {src, start}, scene, shots [{piece, driver, frames, est, guide [prev, start]|null, trim_to, subjects [{source, becomes, facing, side}], action?, extras?}]}}`. Paths can use `$FMB_DATA`, `~`, or be relative to the JSON.

## Env (all optional except where noted)
`FMB_HOME`, `FMB_DATA` (default `~/fmb-data`), `FMB_JOB`, `FMB_CV` (YOLO/SAM weights), `GJ_IMAGE`, `GJ_REGISTRY_AUTH_ID` (private registry only), `GJ_GPU`. Keys: `RUNPOD_API_KEY` (paid runs), `DROPBOX_TOKEN` (uploads), `HF_TOKEN`.
