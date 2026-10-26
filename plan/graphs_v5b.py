"""A08 default render: Singularity H3 v1.3 trimmed int8, realism 0.7, character-swap v1.5 at 1.0,
lightx2v Minimax-h3-Turbo 8-step v1.1 at 0.8. No personal likeness LoRA unless LIKENESS_LORA is set
and LIKENESS_STRENGTH is non-zero. One H3ExactLoraStack injection, H3LoadReport abort, Sage attention,
8 steps, shift 6 video / 3 audio, euler / simple, seed 42, ref_image_size match, 1152x640.
Refs are separate images: head close-up, body only for full-body shots, outfit last.
Linked 141-frame pieces (step 119, 22-frame AddGuide). Natural mouth (no audio in the graph)."""
import json, copy, pathlib, os
FH = os.environ.get('FARQUAD_HOME', os.path.expanduser('~/farquad'))
G = pathlib.Path(FH + '/plan/graphs')
BASE = json.load(open(FH + '/plan/base_graphs/gb_S1_t2.api.json'))
A = json.load(open(FH + '/plan/base_graphs/gjB_p1_s42.api.json'))
LEN, STEP = 141, 119
UNET = 'Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors'
REALISM = ('h3-realism-people-t2v-i2v-r2v.safetensors', 0.7, (104, 0))
SWAP = ('h3_character_swap_v1.5.safetensors', 1.0, (208, 0))
TURBO = ('minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors', 0.8, (208, 0))
KNOWN_EXPECT = {'m33k_lme_s0500.safetensors': (200, 0)}
STEPS = 8
SHIFT_VIDEO = 6.0
SHIFT_AUDIO = 3.0
SEED = 42
REF_IMAGE_SIZE = 'match'
INJECTION_KEY = 'h3_exact_stack'
MIN_COMFY = '0.42.5'
PHRASE_MOUTH = "match the source's mouth and expression exactly"
PHRASE_OBJECTS = "keep any objects already in the scene; add nothing"
CLIPS = {
 'lalaland': dict(full_body=True, look="Photoreal live-action, lit like the scene.", subj=[("the man in the white shirt and tie", "Gomez Addams", "gomez"), ("the woman in the yellow dress", "Morticia Addams", "morticia")],
   keep="The hilltop, dusk sky, city lights, lighting and camera stay as in <Video 1>."),
 'pulp': dict(full_body=True, look="Live-action costumed performers with stop-motion style character design, lit like the scene.", subj=[("the man in the black suit dancing on the right", "Jack Skellington", "jack"), ("the woman in the white shirt and black trousers dancing on the left", "Sally", "sally")],
   keep="The diner stage, audience, lighting and camera stay as in <Video 1>."),
 'thisisamerica': dict(full_body=True, look="Photoreal cinematic CGI character, lit like the scene.", subj=[("the shirtless dancing man in light trousers", "Thanos, wearing exactly the clothes shown for him", "thanos")],
   keep="The warehouse stays exactly as in <Video 1>: add no people, keep it as empty or as full as <Video 1> in every frame. Same framing, lighting and camera as <Video 1>."),
 'walken': dict(full_body=True, look="Photoreal live-action, lit like the scene.", subj=[("the dancing man in the suit", "Patrick Bateman", "bateman")],
   keep="The hotel lobby, lighting and camera stay as in <Video 1>."),
 'singleladies': dict(full_body=True, look="Photoreal live-action, black-and-white like the scene.", subj=[("the dancing woman in the center", "Michael Myers", "myers")],
   keep="The two dancers on the left and right stay exactly as they are. The white studio, lighting and camera stay as in <Video 1>.")}
FIX = {'orig': " Replace each Subject completely, head, face, hair and hands.",
       'bg': " Keep the exact original scene and colours of <Video 1> in every frame.",
       'leak': " Never show the reference picture itself; only the scene of <Video 1>."}

def likeness():
    name = os.environ.get('LIKENESS_LORA', '').strip()
    if not name or name.lower() == 'none':
        return None
    try:
        strength = float(os.environ.get('LIKENESS_STRENGTH', '0') or 0)
    except ValueError:
        strength = 0.0
    if strength == 0:
        return None
    return (name, strength)

def stack_entries():
    rows = [REALISM, SWAP, TURBO]
    extra = likeness()
    if extra:
        name, strength = extra
        rows.append((name, strength, KNOWN_EXPECT.get(name)))
    if len(rows) > 5:
        raise RuntimeError('A08 stack has more than 5 LoRAs')
    return rows

def require_for(rows):
    expect, strengths = {}, {}
    for name, strength, exp in rows:
        strengths[name] = strength
        if exp:
            expect[name] = [exp[0], exp[1]]
    fc2 = 154 if any(name == 'm33k_lme_s0500.safetensors' for name, _, _ in rows) else 104
    return {
        'max_lossy_merges': 0,
        'no_load_warnings': True,
        'expect': expect,
        'min_hooks': {'qkv_proj': 1, 'out_proj': 1, 'fc1': 1, 'fc2': fc2},
        'strengths': strengths,
        'injection_key': INJECTION_KEY,
        'min_comfyui': MIN_COMFY,
        'abort': True,
    }

def ref_plan(c):
    refs = []
    for _who, name, key in c['subj']:
        refs.append(('face', name, f'v5b_{key}_face.png'))
        if c.get('full_body'):
            refs.append(('body', name, f'v5b_{key}_body.png'))
        refs.append(('outfit', name, f'v5b_{key}_outfit.png'))
    return refs

def prompt_for(c):
    bits = ['In <Video 1>:']
    refs = ref_plan(c)
    pic = 1
    cursor = 0
    for j, (who, name, _key) in enumerate(c['subj'], 1):
        nroles = 3 if c.get('full_body') else 2
        group = refs[cursor:cursor + nroles]
        cursor += nroles
        role_s = []
        face_i = outfit_i = None
        for role, nm, _fn in group:
            role_s.append(f"image {pic} is <Picture {pic}>, {nm}'s {role}")
            if role == 'face':
                face_i = pic
            if role == 'outfit':
                outfit_i = pic
            pic += 1
        bits.append(f"Subject {j} is {who}; Subject {j} becomes {name}. " + '; '.join(role_s) + '.')
        bits.append(f"{name} has exactly the face from <Picture {face_i}> and wears exactly the outfit from <Picture {outfit_i}>.")
    bits.append('The pictures are reference only.')
    bits.append(f"Each Subject keeps the motion, gestures and head turns of <Video 1>. {PHRASE_MOUTH}.")
    bits.append(c['keep'])
    bits.append(PHRASE_OBJECTS + '.')
    bits.append(c['look'])
    return ' '.join(bits), [fn for _role, _name, fn in refs]

def prompt(slug):
    return prompt_for(CLIPS[slug])

def graph(slug, k, guide, seed=SEED, fixes=(), t=0):
    g = copy.deepcopy(BASE)
    i = g['15']['inputs']
    pr, refs = prompt_for(CLIPS[slug])
    pr += ''.join(FIX[f] for f in fixes)
    i.update(prompt=pr, length=LEN, width=1152, height=640, ref_image_size=REF_IMAGE_SIZE)
    for key in [key for key in list(i) if key.startswith('ref_images.')]:
        del i[key]
    for nd in ('5', '50', '52', '80', '81', '82', '90', '91', '92', '93', '94', '95', '96'):
        g.pop(nd, None)
    for j, im in enumerate(refs):
        n = str(200 + j)
        g[n] = {'class_type': 'LoadImage', 'inputs': {'image': im}}
        i[f'ref_images.ref_image_{j}'] = [n, 0]
    rows = stack_entries()
    req = require_for(rows)
    if req.get('abort') is not True:
        raise RuntimeError('A08 load check must abort')
    stack_inputs = {'model': ['1', 0]}
    for slot in range(1, 6):
        if slot <= len(rows):
            name, strength, _exp = rows[slot - 1]
            stack_inputs[f'lora_{slot}'] = name
            stack_inputs[f'strength_{slot}'] = strength
        else:
            stack_inputs[f'lora_{slot}'] = 'None'
            stack_inputs[f'strength_{slot}'] = 0.0
    g['1']['inputs']['unet_name'] = UNET
    g['80'] = {'class_type': 'H3ExactLoraStack', 'inputs': stack_inputs}
    rid = f"v5b_{slug}{os.environ.get('RIDTAG', '')}_p{k}_t{t}"
    g['81'] = {'class_type': 'H3LoadReport', 'inputs': {'model': ['80', 0], 'tag': rid, 'require': json.dumps(req)}}
    g['82'] = {'class_type': 'MiniMaxH3SigmaShift', 'inputs': {'model': ['81', 0], 'shift_video': SHIFT_VIDEO, 'shift_audio': SHIFT_AUDIO}}
    g['54'] = {'class_type': 'PathchSageAttentionKJ', 'inputs': {'model': ['82', 0], 'sage_attention': 'auto'}}
    g['20']['inputs']['sampler_name'] = 'euler'
    g['21']['inputs'].update(model=['54', 0], scheduler='simple', steps=STEPS, denoise=1.0)
    g['22']['inputs']['noise_seed'] = seed
    g['23']['inputs']['model'] = ['54', 0]
    g['23']['inputs']['conditioning'] = ['15', 0]
    if any(n.get('class_type') == 'LoraLoaderModelOnly' for n in g.values()):
        raise RuntimeError('A08 graph still has LoraLoaderModelOnly')
    TG = os.environ.get('RIDTAG', '')
    d = f"v5b_{slug}{TG}_d{k}.mp4"
    g['6']['inputs']['file'] = d
    up = [d] + refs
    if guide:
        g['70'] = {'class_type': 'LoadVideo', 'inputs': {'file': guide}}
        g['71'] = {'class_type': 'GetVideoComponents', 'inputs': {'video': ['70', 0]}}
        g['72'] = copy.deepcopy(A['72'])
        g['23']['inputs']['conditioning'] = ['72', 0]
        up.append(guide)
    g['33']['inputs']['filename_prefix'] = f'v5b/{rid}'
    p = G / f'{rid}.api.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(g))
    return {'id': rid, 'engine': 'l21', 'seed': seed, 'graph_path': str(p), 'version': rid, 'uploads': up,
            'params': {'steps': STEPS, 'length': LEN, 'W': 1152, 'H': 640}, 'est_s': 60 + 3 * LEN, 'protect': True, 'rung': rid, 'tier': 'L',
            'needs_stage2': False, 'question': f"v5b {slug} p{k} t{t} {list(fixes)}", 'spec': {}}

if __name__ == '__main__':
    for s in CLIPS:
        print(s, '|', prompt(s)[0], '\n')
