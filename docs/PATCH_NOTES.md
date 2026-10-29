# Fire Marshal Bill — Patch Notes v1.1 (Oct 29, 2026)

Everything from Oct 28–29 in one place. Short version: H3 swaps work well on one or two people with good refs. Group shots and video upscaling are the weak spots. As of v1.1 the code runs on any box and checks its own work.

## New in v1.1 🆕
- **Runs on any box.** Everything ships inside `code/`: the graphs are built in code, plus the model manifest (Hugging Face repo@revision + sha256), the pinned node packs and the Dockerfile. Nothing is read from old project folders. I proved it on a clean copy with no extra env vars.
- **Neutral examples.** The three worked examples are a one-person swap, a two-person contact shot and a group shot, using made-up characters. A free demo script makes synthetic media so every dry run builds.
- **Checks are code now, not just rules:**
  - Trim to the planned frames, then put the source audio back on. This happens automatically after every render.
  - Every output gets a size, aspect and blank/washed-out check, including stills pulled from the video.
  - `qc/selfcheck.py pre` checks the quote line, the what's-different table, the caps, the budget gate, the refs (flat RGB, 1536) and the GPU key (by name).
  - `qc/selfcheck.py post` checks frame count vs the plan, fps, size, audio present and synced, luma, spend vs cap, and pods left. It prints a PASS/FAIL table to paste.
- **Budget gate fixed.** It scans a list of ledger roots you set and prints every ledger it counted. Old "pool" ledgers can't be dated, so they're listed with a warning and never counted.
- **Spend correction:** the final15 video job cost **$4.17, not $4.67**. The two crashed starts were already inside the per-clip totals and got counted twice.

## Keeping ✅
- **H3 Singularity v1.3 int8** is the video model. Refs at **1536 on the long side**, `ref_image_size` max, **seed 77**, the source fps, and the source audio put back on after.
- **The native-size render is the deliverable.** No upscale on video.
- **Angles for stills:** Qwen-Image-Edit-2511 fp8 + the Multiple-Angles LoRA at 0.9.
- **Texture-fix VAE on stills.** It's a small gain: finer pores, no faint grid. The waxy look stays, so it isn't a cure.
- **Phone-friendly Dropbox share links**, a **job record** for every job (README, recipe/ folder and a JOB_INDEX line), **live pod status lines**, and **per-shot prompts** with no face words on back views.

## Scrapped or parked 🗑️
- **ResCascade (parked):** sharper, but it takes 2x the time. It stays off for finals.
- **Fizgig (scrapped):** the effect wasn't visible.
- **Krea 2 refine, SeedVR2 still upscale, the 2MP hi-res pass (scrapped Oct 28, 6:19 PM):** they added time without adding quality.
- **SeedVR2 1080 on video (scrapped, removed from the code):** it over-smoothed the group clip into a plastic look and turned the contact clip all black.
- **2048 refs (scrapped):** slower, and they ran out of GPU memory on 3-character shots.

## Fixes 🔧
- **Node-81 bug:** the script wrote the realism strength (0.7) where the load check expects layer counts like [104, 0], which killed every run in seconds. The graph builder now refuses anything but layer counts.
- **Full error saved** on every failure: the whole error plus the ComfyUI log tail, in `fail_*.log`.
- **No rerolls on graph errors.** A setup or validation error stops the clip.
- **Black/washed-out check** on every video output.
- **Trim to the planned frames** + the audio re-muxed to match, so there's no frozen tail.
- **Flat-RGB ref check:** 8-bit RGB with no alpha, 1536 long side.
- **2x time stop** per piece and for the whole clip.
- **Budget gate** adds up every dated ledger since the budget started, not just today's.
- **Size + blank-canvas checks** on stills: the first miss stops the stage.

## Lessons 📚
- **Plastic skin comes from:** chaining edits on AI outputs, fp8 + the angle LoRA, 1MP with no detail pass, and flat white light. Make each image in one pass from real photos whenever you can.
- **H3 has NO mask input.** Masks are who's-who maps and after-checks only. Placement in multi-person shots comes from the prompt text alone.
- **Make a back-view ref of the outfit** if the character is ever seen from behind. The code now refuses to build a back-facing shot without one.
- **Source poses beat prompts.** If the source person crosses their arms, the swap does too. Pick different footage.
- **Pick source windows without crowd chaos.** Crowds make the tracker relabel people, and the assignments break.

## Runs and verdicts 💸
| Run | Spend | Verdict |
|---|---|---|
| Oct 28 fix round | $0.64 | fixes in |
| VAE / ResCascade / Fizgig test | $3.844 | texfix small gain (keep); ResCascade sharper but 2x time (park); Fizgig not visible (drop) |
| final15 video (3 clips, ~15 s each) | **$4.17** (was reported as $4.67) | **Clip A, one-person swap: PASS.** **Clip B, two-person slap: FAIL** (jersey front shown on his back, a missing forearm, weak slap contact). **Clip C, group shot with two puppets: FAIL** (wrong placement in 3 shots, crossed arms copied from the source). |

Testing budget on the original box: **$11.69 spent** if the budget counts all of Oct 28 (how the earlier gates counted it), or **$8.01** counting only from 8:18 PM Oct 28. Either way, it's well under $30.

## GPU image access (settled, don't re-raise)
Both images are PUBLIC on GHCR, so anyone can pull them with no token (checked 2026-10-29: anonymous tag list and manifest both work).
- Current fastboot: `ghcr.io/ayoomeco/genie-jutsu-worker:fastboot-comfy-0.42.5-r2` (rollbacks `fastboot-comfy-0.42.5`, `fastboot-v0.39.1`).
- Older Genie Jutsu image: `ghcr.io/ayoomeco/chywgja` (made public Oct 20-21; Sephiroth pins it by digest).
- `GJ_REGISTRY_AUTH_ID` and `image/Dockerfile` are only for someone who wants their own private build.
