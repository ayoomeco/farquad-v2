# Fire Marshal Bill — Standing Rules

There's no audit bot. I check my own work with the two checklists below and show the evidence every time: real numbers, file names and frame checks, not "looks good".

## Money
1. **Nothing paid runs without your yes** to a quote with a cap. The quote line reads: "About $X (cap $Y), about N min."
2. **Quote first, with a "what's different from last time" table** showing every setting with its old and new values: GPU, model, size, steps, sampler, shift, seed, each LoRA and its strength, ref count/order/size, ref_image_size, frames per piece, prompt structure, and any extra pass.
3. **Never change a variable silently.** Anything off the baseline gets named and approved, or it gets fixed instead of launched.
4. **The budget gate** adds up every ledger since the budget started, plus the new caps. If the total goes over, nothing launches.
5. **Reroll only when needed:** for an error or a clearly bad piece, at most 2 new seeds, inside the cap. Graph or validation errors get fixed, not rerolled.
6. **Terminate every pod and confirm 0 are left.** Report the real spend against the cap.

## Live status
7. You get status lines at each stage: **LAUNCHED → DOWNLOADING → READY → FIRST OUTPUT → PIECE n/N → DONE**, or **FAILED + the reason** at any point.
8. I check the provider (RunPod) directly. I never say a run is going until the pod is confirmed live.

## Craft
9. **Per-shot prompts.** Every shot gets its own prompt. Back views have no face words.
10. **The background color goes in the generation prompt** ("solid pure black background"), never matted on afterward.
11. **Clips keep the source fps and the source audio.** If a deliverable will be silent, I flag it up front.
12. Refs are separate images (never a sheet), flat 8-bit RGB with no alpha, 1536 long side. Faces come from real photos whenever possible.
13. If a character is seen from behind, they get a back-view ref of that outfit.
14. Only consenting people or original characters.

## Delivery
15. **Phone-friendly share links only** (Dropbox view links). No folder paths.
16. **A job record for every job:** a README (what, why, recipe, spend, problems, results), a recipe/ folder with the scripts and configs, and one line in JOB_INDEX.md.
17. Results are private. I never post or share them for you.
18. Keys only through the secure secret card, read by name. Never printed or written down.

## Pre-launch self-check ($0, shown with evidence before asking for a yes)
Run `python3 qc/selfcheck.py pre` and paste its table; the boxes below are what it checks plus what only eyes can check.
- [ ] Source: native HD, cuts detected and checked by eye, contact sheet made, the window avoids crowd ID chaos.
- [ ] Pieces are ≤141 frames, never across a hard cut, source fps kept, driver muted.
- [ ] Refs: listed with size/mode (prep check: RGB, no alpha, 1536 long side). Back ref present if anyone's seen from behind.
- [ ] Masks (multi-person): who's-who overlay built and looked at; every subject has a position in the prompt.
- [ ] Prompts: every ref role named; "match the source's mouth and expression exactly"; "keep any objects already in the scene; add nothing"; no face words on back views. Source poses the swap must NOT copy are flagged.
- [ ] Built graphs printed and diffed against the baseline (steps 8, euler/simple, shift 6, seed 77, LoRAs, H3LoadReport expect = layer counts).
- [ ] Banned steps absent: ResCascade, Fizgig, SeedVR2 video.
- [ ] Dry run passes: budget gate ok, no missing inputs, refuses without --go.
- [ ] The what's-different table and quote are sent, and you said yes.

## Post-render self-check (before you see anything)
Run `python3 qc/selfcheck.py post --clip <id> --live` and paste its table; then do the eyeball rows.
- [ ] Specs: frame count = planned (trimmed), fps = source, audio length = video length, the file plays.
- [ ] Per piece, checked against the masks: right character in each spot, no leftover original person, no ID swap, no extra heads, no flat ref frames, no missing limbs, the outfit right from every side, mouth matches.
- [ ] Not black, not washed out (luma check).
- [ ] Contact sheet with timestamps; every flag named honestly with its time.
- [ ] Pods 0, spend vs cap reported.
- [ ] A PASS or FAIL verdict per clip with the reasons, then links, then the job record.
