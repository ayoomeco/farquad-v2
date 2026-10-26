# Install GJ V2

Sephiroth installs this branch. The Farquad bot stays on `main`. Do not merge this branch to run it.

No GPU is rented during setup, and setup does not spend money. `RUNPOD_API_KEY` is not read. Model weights for the render stay on the pod later; setup only downloads three small CPU models (face + segment), pinned by sha256.

## Clone and setup

```bash
git clone -b gj-v2 https://github.com/ayoomeco/farquad-v2
cd farquad-v2
bash setup.sh
```

`setup.sh` can run from any working directory (`bash /path/to/farquad-v2/setup.sh`). The install root is:

1. `GJ_HOME` if set
2. else `FARQUAD_HOME` if set
3. else the directory that contains `setup.sh`

Both variables are exported to that same directory for the rest of the setup. A successful setup ends with this line:

```text
GJ_V2_SETUP_OK FARQUAD_SETUP_OK  (activate with: . $GJ_HOME/venv/bin/activate)
```

Then activate the venv before later Python commands:

```bash
. "$GJ_HOME/venv/bin/activate"
```

Needs Python 3.11+, `ffmpeg`, `ffprobe`, and `curl`.

## Environment variables

| Name | Required | Used for |
|---|---|---|
| `RUNPOD_API_KEY` | only for a paid render | RunPod API. Never written to a file. Not used by setup. |
| `HF_TOKEN` | optional | Hugging Face downloads on the pod when a repo is gated. Public GJ V2 weights do not need it. |
| `DROPBOX_TOKEN` | optional | `dbx/upall.py` token mode only. |
| `GJ_HOME` | optional | Install root. Wins over `FARQUAD_HOME`. |
| `FARQUAD_HOME` | optional | Same root, if `GJ_HOME` is unset. |

## Free dry-runs

These do not create a pod and do not spend.

```bash
DRYCHECK=1 DSR_RUN=/tmp python plan/driver_v5b3.py
DRY=1 DRY_STATE='{"pods":[],"bal":10}' python plan/idle_watchdog.py
python polish/pod/post_gjv2.py --plan
python plan/test_gj_v2.py
```

`setup.sh` already runs the first two. A price quote (still no rent) is:

```bash
cd engine
FP_RUN_DIR=/tmp/gj-dry FP_GROUP=gjfast JUTSU_CONFIG=$PWD/dsr/genie_dsr.toml \
  python scripts/phase1_bakeoff.py plan --refine --fullperson --gpu any24 --cloud COMMUNITY_FIRST --cap 6
```

`launch` is the command that rents a GPU. Do not run it during setup, and do not run it until a per-job cap has been approved.
