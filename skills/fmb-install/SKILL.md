---
name: fmb-install
description: >-
  Use when Fire Marshal Bill is first set up, when the fmb-v1 code is missing or
  broken on the box, or before the first render of a session: clone the fmb-v1
  branch, run setup, check keys by name only, and prove it works with free dry
  runs. Never spends GPU money.
---
# Install Fire Marshal Bill (free, no GPU)

Nothing here rents a GPU or spends money. **The money rule: no `--go`, no pod, no paid call during install. Ever.**

## 1. Clone
```bash
export FMB_HOME=/workspace/fmb-v1
if [ -d "$FMB_HOME/.git" ]; then git -C "$FMB_HOME" pull --ff-only; else git clone -b fmb-v1 https://github.com/ayoomeco/farquad-v2 "$FMB_HOME"; fi
git -C "$FMB_HOME" log -1 --format='%h %s'
```
Tell the owner the short commit id. On failure, report the exact error and stop. The patch notes, knowledge base and rules are in `$FMB_HOME/docs/`.

## 2. Setup
Needs Python 3.11+, ffmpeg, ffprobe and curl (`apt-get install -y ffmpeg curl python3-venv` if missing), plus about 2 GB free (check `df -h /` first; never delete the owner's files to make room).
```bash
FMB_DATA=/workspace/fmb-data bash $FMB_HOME/setup.sh
```
It must end with `FMB_SETUP_OK`. If it doesn't, report the last 20 lines and stop. `FMB_MASKS=0` skips the YOLO/SAM download. Before later commands: `. $FMB_HOME/venv/bin/activate`.

## 3. Keys (by name only)
- `RUNPOD_API_KEY`: needed only for paid runs. Get it through the secure secret card. Never print, echo, log or write it. Check: `[ -n "$RUNPOD_API_KEY" ] && echo set || echo missing`.
- Optional: `DROPBOX_TOKEN` (uploads via `dbx/upall.py`), `HF_TOKEN`. Same rules.

## 4. Free dry runs (synthetic demo media, no GPU)
```bash
. $FMB_HOME/venv/bin/activate && cd $FMB_HOME
export FMB_DATA=/workspace/fmb-data FMB_JOB=$FMB_HOME/jobs/example_job.json
python3 demo/make_demo.py && python3 video/prep_refs.py && python3 video/prep_audio.py
python3 video/graph_video.py
for c in one_person two_person_contact group; do python3 video/run_video.py --clip $c; done
python3 refs/graph_refs.py && python3 refs/run_refs.py
python3 qc/selfcheck.py pre
python3 demo/post_demo.py && python3 qc/selfcheck.py post --clip one_person
```
Every runner must print `BUILT, NOT RUN` with `budget gate ok` and `missing inputs: none`. Without `--go` they refuse to launch. On the demo, the self-check tables should pass every row except the key row (if `RUNPOD_API_KEY` isn't set yet) and the manual eyeball row. Report: commit, FMB_SETUP_OK, dry runs passed, $0 spent.

## 5. GPU image
The default image (`ghcr.io/ayoomeco/genie-jutsu-worker:fastboot-comfy-0.42.5-r2`) is public, so it pulls with no token. Only if you want your own build: build `image/Dockerfile`, push it, and set `GJ_IMAGE` (plus `GJ_REGISTRY_AUTH_ID` if your registry is private). Models are not baked in: boot.sh fetches the manifest sets and checks sha256.
