"""Per-shot H3 prompt templates. One prompt per shot, built from the shot's subjects + the job's characters.
Rules baked in: describe the source person by clothing + position only; name every ref's role; back-facing subjects get the back-of-outfit ref and
'stays seen from behind'; multi-person shots say exactly N people and no face swaps; always 'match the source's mouth and expression exactly' and
'keep any objects already in the scene; add nothing'. Source poses win over prompts: if the source person crosses their arms, pick another window."""
def shot_refs(shot, chars):
    """ordered (character, kind, path) for this shot: per subject in on-screen order: head, body front, + body back when facing away."""
    out = []
    for s in shot['subjects']:
        c = chars[s['becomes']]; R = c['refs']
        out += [(s['becomes'], 'head', R['head']), (s['becomes'], 'body', R['body'])]
        if s.get('facing') == 'back':
            if not R.get('back'): raise ValueError(f"{s['becomes']} is seen from behind in {shot['piece']} but has no back-view ref of the outfit: make one first")
            out.append((s['becomes'], 'back', R['back']))
    if len(out) > 9: raise ValueError(f"{shot['piece']}: {len(out)} refs > 9 (H3 max); fewer characters per shot")
    return out
def build(shot, chars, scene):
    subs = shot['subjects']; refs = shot_refs(shot, chars); n = len(subs)
    pic = {}; [pic.setdefault((c, k), i + 1) for i, (c, k, _) in enumerate(refs)]
    p = 'In <Video 1>: '
    for i, s in enumerate(subs, 1):
        c = chars[s['becomes']]; p += f"Subject {i} is {s['source']}; Subject {i} becomes {s['becomes']}: {c['look']}. "
        if s.get('facing') == 'back': p += f"{s['becomes']} stays seen from behind, showing the back of the head and the back of the outfit. "
    p += 'Reference roles: '
    for (c, k), i in pic.items():
        o = chars[c].get('outfit', 'their outfit')
        p += {'head': f"<Picture {i}> is {c}'s head: a front head close-up, neutral expression, mouth closed; use it only for {c}'s face, hair and head. ",
              'body': f"<Picture {i}> is {c}'s full body from the front: use it for {c}'s build, proportions and clothing ({o}). ",
              'back': f"<Picture {i}> is {c}'s full body from behind in the same outfit: use it for the back of {c}'s outfit and hair. "}[k]
    p += 'The pictures are reference only: no frame ever cuts to any picture, and their background never appears in the scene. '
    if n > 1:
        p += ' '.join(f"{s['becomes']} always has the head from <Picture {pic[(s['becomes'], 'head')]}> and the outfit from <Picture {pic[(s['becomes'], 'body')]}>." for s in subs)
        p += f' They never swap faces, and there are exactly {n} people. '
        if shot.get('extras'): p += f"{shot['extras']} stay exactly as they are. "
    p += ('Each keeps the exact position, posture, gestures, contact, head turns and timing of the person they replace. ' if n > 1 else
          'Subject 1 keeps the posture, the gestures, the hand movements, the head turns and the natural mouth movement of <Video 1>. ')
    if shot.get('action'): p += shot['action'].strip() + ' '
    p += f"Match the source's mouth and expression exactly in every frame. {scene} stay exactly as in <Video 1>; keep any objects already in the scene; add nothing. "
    human = all(chars[s['becomes']].get('human', True) for s in subs)
    p += 'Photoreal live-action, lit like the scene.' if human else 'Each character keeps its own design and materials, lit like the scene.'
    return p, [r for _, _, r in refs], human
