---
name: job-record
description: >-
  Use at the end of EVERY job (any paid GPU run, test, render, catalog batch,
  research or build) and when asked 'what did we do / what settings did we use'
  for a past job: writes a README plus a recipe folder into the job's Dropbox
  folder and adds a line to the job index, so the work can be recalled and
  reproduced after the computer is wiped.
---
# Job record: every job must be recallable without the computer

The computer gets wiped to free space, so Dropbox is the memory. A job isn't done until its record is in Dropbox.

## When
- Write the record at the end of every job, including failed or partial ones and test runs, before telling the owner the job is done.
- To answer a question about a past job, read the index and the job's README first. Never answer from memory alone.

## 1. README.md in the job's Dropbox folder
Write it in plain English, so a person or a fresh bot with no computer access can understand and rerun the job. Cover:
- **What and why:** date and time (owner's time zone), the owner's ask in their own words, and every approval with its time (caps, budget, scope changes).
- **Exact recipe:** each model and file with repo@commit and hash where known, LoRAs and strengths, seeds, steps, CFG, sampler and scheduler, shift, sizes, frames and fps, refs used (paths), prompts (or the file that holds them), custom nodes at their pinned commits, GPU type, provider and pod ids.
- **What's different:** the table versus the last run and versus the current baseline.
- **Spend:** per pod, including failed starts, against each cap, plus the day's running total.
- **Problems and fixes:** every error, refusal or relaunch and what changed.
- **Results:** the verdict per arm, the self-check tables (qc/selfcheck.py pre and post), and what the owner decided to keep, scrap or park.
- **File map:** what each subfolder and filename pattern means, which links were shared, and any audio or source offsets.
- **Open decisions:** next steps.

## 2. recipe/ subfolder
Upload the small files needed to rerun the job: quotes, self-check tables, runner scripts, graph JSONs, prompt files, model manifests, status logs, spend ledgers, and proof notes. Keep the total under about 50 MB.
- **Never upload secrets.** Grep for API keys, tokens and passwords, and strip them before upload.

## 3. Job index
Append one line to the running index file in Dropbox: date, job name, folder path, one-line result, spend, and the README link. Create the file if it's missing. Use this index to find jobs later.

## 4. Verify
List the Dropbox folder and confirm README.md, recipe/ and the index line exist. Tell the owner the record is saved, with the folder link.

## Recall
When asked about past work, search the index, open the README, and quote the settings from it. If no record exists, say so plainly; don't reconstruct settings from guesses.
