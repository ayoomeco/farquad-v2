"""Character-ref job planner ($0). Reads the job's 'ref_job' block and builds three stages of stills graphs:
  T  fresh whites: Qwen-Image 2.1 on a blank white canvas (sets the frame: head 896x1152, body 768x1376, ~1 MP) + REAL photos for identity,
     texture-fix VAE decode, background colour in the prompt. Only for shots with "stage": "white".
  R  edits to black: Qwen-Image-Edit-2511 fp8 1 MP (front: LoRA off; back/angle: Multiple-Angles LoRA 0.9, no face words). Source = a real photo or a T white.
  U  SeedVR2 7B image upscale to 1536 short side (optional, ref_job.upscale).
Lesson built in: one pass from real photos where possible; every chained edit on an AI output costs skin texture."""
import sys, os, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'lib'))
import fmbcfg as CFG, graphs as G
from PIL import Image
CANVAS = {'head': (896, 1152), 'body': (768, 1376)}
BLACK = 'solid pure black #000000 background, evenly black edge to edge, with no gradient, glow or halo'
T_HEAD = ("<image1> is a blank solid white canvas: use it only for the frame shape. {photos} A front head close-up photo of {name}: {look}. Neutral expression, mouth closed, "
          "face filling the frame, soft natural light, real skin texture with pores, no retouching. Solid pure white background (#FFFFFF), evenly white edge to edge.")
T_BODY = ("<image1> is a blank solid white canvas: use it only for the frame shape. {photos} A full-body photo of {name} standing straight, facing the camera, arms relaxed at the sides, "
          "head to feet in frame, wearing {outfit}. {look}. Soft natural light, real skin and fabric texture. Solid pure white background (#FFFFFF), evenly white edge to edge.")
R_FRONT = f"Change only the background to {BLACK}. Keep the person, the clothing, the skin texture, the pose and the lighting exactly the same."
R_BACK = ("<sks> back view eye-level shot medium shot\nThe same person seen from directly behind, full body, showing the back of the hair and the back of the outfit "
          "({outfit}), arms relaxed at the sides; " + BLACK + ".")
def canvases(outdir):
    os.makedirs(outdir, exist_ok=True); out = {}
    for k, s in CANVAS.items():
        p = f'{outdir}/canvas_white_{s[0]}x{s[1]}.png'
        if not os.path.exists(p): Image.new('RGB', s, (255, 255, 255)).save(p)
        out[k] = p
    return out
def plan():
    J = CFG.job(); RJ = J['ref_job']; W = CFG.DATA / J['job'] / 'refjob'; cv = canvases(str(W / 'inputs')); T, R, U = [], [], []
    for name, ch in RJ['characters'].items():
        photos = [CFG.P(p) for p in ch.get('photos', [])]; c = J['characters'].get(name, {})
        for s in ch['shots']:
            tag = f"{name.lower()}_{s['shot']}"; kind = s['kind']
            if s.get('stage') == 'white':
                ph = ' '.join(f'<image{i + 2}> is a real photo of {name}: use it only for identity.' for i in range(len(photos)))
                p = s.get('prompt') or (T_HEAD if kind == 'head' else T_BODY).format(photos=ph, name=name, look=c.get('look', ''), outfit=c.get('outfit', 'their outfit'))
                ups = {os.path.basename(cv[kind]): cv[kind], **{os.path.basename(x): x for x in photos}}
                T.append(dict(name=f't_{tag}_white', graph=G.qwen21_white(p, list(ups), f"fmb/{J['job']}/t_{tag}_white", res=1024, seed=J.get('seed', 77)),
                              uploads=ups, expect_size=CANVAS[kind], kind=kind, prompt=p))
                src = f't_{tag}_white'
            else: src = s['from'] if s['from'].startswith('t_') else CFG.P(s['from'])
            back = kind == 'back'; p = s.get('edit_prompt') or (R_BACK.format(outfit=c.get('outfit', 'their outfit')) if back else R_FRONT)
            if back: assert not any(w in p.lower() for w in G.FACE_WORDS), f'{tag}: back views carry no face words'
            R.append(dict(name=f'r_{tag}_black', src=src, angle=back, prompt=p, kind=kind,
                          graph=G.edit2511(p, 'SRC.png', f"fmb/{J['job']}/r_{tag}_black", angle=back, seed=J.get('seed', 77))))
            if RJ.get('upscale', True): U.append(dict(name=f'u_{tag}_1536', src=f'r_{tag}_black', graph=G.seedvr2_image('SRC.png', f"fmb/{J['job']}/u_{tag}_1536", 1536)))
    return dict(T=T, R=R, U=U, canvases=cv)
if __name__ == '__main__':
    P = plan(); J = CFG.job(); W = CFG.DATA / J['job'] / 'refjob'; W.mkdir(parents=True, exist_ok=True)
    for k in 'TRU': print(f"{k}: {len(P[k])}  " + ', '.join(x['name'] for x in P[k]))
    json.dump(P, open(W / 'graphs_refs.json', 'w'), indent=1); print('graphs asserted (bg colour in every prompt, no face words on backs) -> graphs_refs.json')
