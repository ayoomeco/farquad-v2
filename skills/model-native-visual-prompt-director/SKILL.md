---
name: model-native-visual-prompt-director
description: >-
  Use whenever a character ref still, an image edit or a video shot needs a
  prompt: converts intent into a model-native prompt plus run sheet, with Bill's
  per-shot H3 rules and the background color in the prompt.
---
# Model-Native Visual Prompt Director


Use whenever a ref still, an edit or a video shot needs a production-ready prompt. Converts creative intent into a model-native prompt + UI run sheet. Replaces generic comma-chain prompts, adjective piles, and negative-heavy prompts.

## Principle

Maximum **decision density**, not maximum length. Every sentence creates a visible choice, protects continuity, or prevents a known failure. Settings that already have dedicated UI controls belong in the **run sheet**, not repeated as contradictory prose.

## Do not quiz the user for the template

Treat the user message as creative intent. Build missing detail from campaign objective, the job JSON, the knowledge base, project refs, platform, content class, and prior approved work. Ask only for a true blocking, irreversible, sensitive, or uninferable fact.

## Mandatory sequence

1. Identify asset job, exact visible moment or temporal arc, audience effect, platform, destination, content class, approval state.
2. Load the job brief and the character refs (vault names H1_front, B1_front, B2_back...).
3. Build reference stack. Label each source: identity · pose · wardrobe · hair · nails · beauty · accessory · prop · location · background · lighting · palette · camera · composition · texture · narrative · motion. State exact borrowed element and what must not be copied.
4. Choose model by **capability**, not familiarity or credits. Confirm exact current UI label/mode before paid run.
5. Separate **visible-scene prompt** from **UI run sheet** (identity selectors, Moodboards, HEX, refs, aspect, resolution, quality, duration, seed, presets, first/last frames, Elements, batch, audio).
6. Compile a **model-native** prompt. Detailed prose only when the mode benefits. Never pad with generic adjectives.
7. **Lint** — score ≥90/100 and pass all critical checks before credits.
8. Propose three scout candidates only when they test **named variables**. Keep other decisions locked.
9. After outputs: one highest-impact failure; preserve wins; one targeted repair. Two same-type failures → model/reference reassessment.
10. Record prompt, model/version, mode, settings, refs, output IDs, review, next action.

## Abstract → visible (required)

Never pass labels like expensive, confident, seductive, cool, cinematic without translating into observable decisions (chin, gaze, hands, fabric construction, practical light, camera distance, process).

## Model rules (summary — see playbook for full templates)

| Family | Use | Critical |
|--------|-----|----------|
| Soul 2.0 | Fashion-aware, culture-native editorial stills; full control | Soul ID + Personal Moodboard + Soul HEX in UI; **reference image disables prompt field** → return reference-mode recipe |
| Soul Cinema | Cinematic portraits, atmosphere, video keyframes | No preset/Moodboard; Soul ID + HEX; no empty “cinematic masterpiece”; reference disables prompt |
| Soul base | Fast preset-led lifestyle batches | Simpler prose; preset carries style |
| Popcorn | Connected carousels/storyboards ≤8 frames | Manual vs Auto; ≤4 refs with image-number roles |
| Nano Banana Pro / GPT Image 2 | Exact quantities, layouts, products, typography, multi-ref composition | Spatial relationships + quoted text |
| Recraft V4 | Style from reference set | Subject/action/context/composition; don’t fight refs with style essay |
| Higgsfield DoP | Image-to-video | Approved keyframe + motion preset first; describe change over time |
| Cinema Studio / Seedance | High-control video | Era/genre/camera/lens/aperture/light/palette/tempo/Elements in controls; prose = story/performance/beats/audio |
| Kling / Sora / Veo / Wan / Minimax / etc. | Per playbook router | Record exact UI version |
| Grok Imagine Image 2.0 | New comps / one edit / multi-image ≤5 roles | Edits: what changes + what stays |
| Grok Imagine Video 1.5 | T2V / I2V / ref-to-video ≤7 / video edit | Never mix incompatible modes |


## Detail standard

**Still:** exact moment; acting; posture/weight; shoulders/hips/legs/elbows/wrists/hands/fingers/chin/eyes/mouth; wardrobe cut/fabric/fit/drape/wear; hair/makeup/nails/jewelry/skin; FG / subject plane / mid / BG / props / signage; motivated key/fill/neg fill/practicals/catchlights/skin exposure/falloff; shot size/height/angle/lens/distance/focus/DoF/crop/neg space; palette/materials/process/grain/halation/contrast/sharpness/retouch; short critical exclusions.

**Video:** start → 2–4 performance beats → secondary motion → environment → camera → final; separate `Sound:` when supported. If approved still is source: only what changes + what stays locked.

## Required output packet (exact order)

```
RECOMMENDATION
Model: [CURRENT EXACT MODEL]
Mode: [TEXT / REFERENCE / IMAGE-TO-VIDEO / ETC.]
Reason: [ONE SENTENCE]

REFERENCE MANIFEST
REF_1 — role: [ROLE]; borrow: [EXACT ELEMENT]; do not copy: [BOUNDARY]
[REPEAT]

UI RUN SHEET
Identity: [SOUL ID / ELEMENT / REFERENCE]
Moodboard / style control: [VALUE]
Color control: [VALUE]
Aspect ratio: [VALUE]
Resolution / quality: [VALUE]
Duration: [VALUE]
Batch: [VALUE]
Preset / camera / lens / seed / audio: [VALUES]

COPY-PASTE PROMPT
[ONLY THE TEXT THE SELECTED MODEL CAN ACTUALLY ACCEPT]

SCOUT PLAN
A: [VARIABLE]
B: [VARIABLE]
C: [VARIABLE]

PASS CONDITIONS
1. [CONDITION]
2. [CONDITION]
3. [CONDITION]
4. [CONDITION]
5. [CONDITION]

REPAIR PLAN
If [FAILURE], preserve [WINS] and change only [MODULE OR SETTING].
```

If mode has no prompt field: `COPY-PASTE PROMPT: None — reference-driven mode` plus reference recipe + UI settings.

## Lint gate (≥90 required)

| Dimension | Weight |
|-----------|--------|
| Identity and brand | 15 |
| Visible moment | 10 |
| Physical acting | 10 |
| Styling and beauty | 10 |
| Set and prop logic | 10 |
| Lighting and camera | 15 |
| Reference discipline | 10 |
| Model-native correctness | 10 |
| Originality and taste | 5 |
| Rights, safety, destination | 5 |

**Critical blocks (any one fails the run):** age ambiguity; unapproved real-person likeness; prohibited content; missing identity method; incompatible prompt/reference mode; destination mismatch; unresolved rights; unverified paid settings. Never rewrite to bypass moderation.

## Repair loop

`lint → scout → select → repair → final → derivative` — not repeated full rerolls. Two consecutive same-type failures → router review.

## Safety

No generate / publish / spend / messaging / account change without exact scoped approval. Adult 25+ where relevant. Separate confirmed facts, assumptions, proposals, completed actions, blockers, unknowns.

## Bill's per-shot rules (H3 reference-to-video and Qwen refs)
- **One prompt per shot.** Never one prompt for a whole clip. Each shot names who is where (left/center/right), what each person wears, and which picture is which.
- Name the ref roles: "Picture 1 is the head (face), Picture 2 the full body front, Picture 3 the full body back." Up to 9 pictures per shot, in on-screen order.
- Describe the source person being replaced by **clothing and position only**, never by hair or face.
- Always include: "match the source's mouth and expression exactly" and "keep any objects already in the scene; add nothing". Never list example props (the model adds them).
- **Back views: no face words** (no eyes, smile, expression). Say "seen from behind, face not visible, head facing away exactly as in the source".
- The prompt can't override the source pose: arms and stance follow the driver. Pick a window instead of fighting it.
- **Background color goes in the generation prompt** for every ref still: "solid pure black #000000 background, evenly black edge to edge" (or pure white). Never generate on another color and matte afterward.
- Head refs: neutral, mouth closed, face filling the frame. Marks (tattoos, piercings) on the correct side of THEIR body (selfies are mirrored).
- Angle/back edits: angle LoRA 0.9, no face words. Front edits: LoRA off.
