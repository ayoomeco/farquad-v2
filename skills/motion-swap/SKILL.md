---
name: motion-swap
description: >-
  Use when someone wants to swap a person in a video clip (one person, a group,
  cartoon/3D characters, or their own face) or make a character perform a
  clip's motion: the plain-talk front door, routing and "what already worked"
  list. Runs on fmb-workflow; merged from motion-swap + video-swap-pipeline.
---
# Motion swap (character swap + motion transfer)

The steps, commands and code live in `fmb-workflow`; the baseline and gates in `video-prelaunch-gate`. This skill is the front door: how to talk about it, how to route shots, and what already worked.

## Talk simply
One or two plain sentences per step. Say only that refs are made on a rented GPU (or from their photos) and the render runs on a rented RunPod GPU. Every result keeps the clip's **original audio**. Only consenting people or original characters.

## Intake (free)
1. Understand the ask conversationally: the clip (link, file or description) and who becomes who. "Make my character do this dance" = motion transfer with the same flow.
2. Ask once: **natural mouth (default)** or lip sync. Natural mouth = muted driver, no audio inputs in the graph, original audio muxed back on the final. Lip sync is a separate quoted add-on.
3. Find the best native HD copy (1080p+, one continuous take per piece). Never feed an upscaled or re-encoded low-res source.

## Routing per shot
| Shot | Route |
|---|---|
| No people / nothing to change | keep (copy from source) |
| 1-2 people | H3 v1.3 int8 reference-to-video, full-frame redraw, refs in on-screen order (≤9 refs per shot: 2 people = 4 each + a 9th for the lead) |
| 3+ people or crowd ID chaos | first try another window; else plan on H3 with tight positions and warn about head swaps. Offer SCAIL-2 (20 steps, who's-who masks, less photoreal, 1280x704) only with the owner's OK |

- Full frame only: render the whole frame with the source as guide. Never paste a render back through a mask (halos, seams). Masks are who's-who and check maps only.
- Back-of-head in the source stays back-of-head and gets the body-back ref; no face words in that shot's prompt.
- The swap copies the source arms and pose whatever the prompt says: pick a window with poses you want.

## Quote, launch, deliver
Follow `fmb-workflow` steps 4-8: `qc/selfcheck.py pre` table + what's-different table + "About $X (cap $Y), about N min." → wait for a yes → launch with live status → `qc/selfcheck.py post --live` → phone-friendly links to the final and side-by-side (say which side is which), spend vs cap, 0 pods, named flags → job record.

## When something breaks
Say plainly what happened, what it cost and the recommended fix, then wait. Graph/validation errors are fixed, never rerolled. A bad piece gets at most 2 new seeds (78, 79) inside the cap. Change only the broken piece.

## What already worked (read at the start of every job)
- The Oct 29 baseline in `video-prelaunch-gate` (H3 v1.3 int8, seed 77, PRO 6000 SECURE, refs 1536 max flat RGB, native res, source fps, audio muxed) produced the best 15 s clips for about $1.1-1.7 each.
- One ref set + a per-shot prompt per character; never run a whole video on one reference.
- Prompt lines that work: ref roles named ("Picture 1 is the head, Picture 2 the body front, Picture 3 the body back"), "match the source's mouth and expression exactly", "keep any objects already in the scene; add nothing".
- Neutral, closed-mouth head refs (a smiling ref made the swap smile).
- Real photos for faces; one generation pass from them (stacked AI edits go plastic).
- Big Dropbox uploads: temporary upload link + one curl POST (`dbx/upall.py`), never chunked.

## Rules
- No spend without a quote and an explicit yes. Keys only in env vars, never printed. Results are private: never post or share them.
