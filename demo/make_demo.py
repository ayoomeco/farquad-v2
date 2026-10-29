"""Synthetic demo media ($0, local): photos, refs, source clips with audio, per-piece drivers, a demo quote, and a demo 'render' for the post-check.
Everything lands in $FMB_DATA/demo so the example job's dry runs build with NO missing inputs on any box. Not real people: noise + shapes."""
import sys, os, json, subprocess, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'lib'))
import fmbcfg as CFG
import numpy as np
from PIL import Image, ImageDraw
D = CFG.DATA / 'demo'
def img(path, w, h, seed, rgba=False):
    r = np.random.default_rng(seed); a = (r.random((h, w, 3)) * 80 + np.linspace(40, 160, w)[None, :, None]).astype(np.uint8)
    im = Image.fromarray(a); d = ImageDraw.Draw(im); d.ellipse([w * .3, h * .1, w * .7, h * .45], fill=(200, 150, 120)); d.rectangle([w * .25, h * .45, w * .75, h * .95], fill=(90, 60, 30))
    if rgba: im = im.convert('RGBA')
    os.makedirs(os.path.dirname(path), exist_ok=True); im.save(path)
def vid(path, w, h, fps, secs, audio=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'testsrc2=size={w}x{h}:rate={fps}:duration={secs}']
    if audio: cmd += ['-f', 'lavfi', '-i', f'sine=frequency=440:sample_rate=48000:duration={secs}', '-c:a', 'aac', '-shortest']
    subprocess.run(cmd + ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', path], check=True)
def cut(src, out, start_f, n, fps):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-vf', f'select=between(n\\,{start_f}\\,{start_f + n - 1}),setpts=N/FRAME_RATE/TB', '-frames:v', str(n), '-an', '-r', str(fps),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out], check=True)
def main():
    J = CFG.job(); s = 1
    for ch, c in J['characters'].items():
        for k, p in c['refs'].items(): s += 1; img(CFG.P(p), 1600 if k == 'head' else 1100, 2000 if k == 'head' else 2400, s, rgba=(k == 'body'))   # one RGBA on purpose: prep flattens it
    for ch, c in J.get('ref_job', {}).get('characters', {}).items():
        for p in c.get('photos', []): s += 1; img(CFG.P(p), 1200, 1600, s)
    for cid, c in J['clips'].items():
        fps = c['fps']; src = CFG.P(c['audio']['src']); vid(src, c['W'], c['H'], fps, round(c['frames_out'] / fps + 1, 2))
        cur = 0; st = {}
        for sh in c['shots']:
            start = st[sh['guide'][0]] + sh['guide'][1] if sh.get('guide') else cur
            st[sh['piece']] = start; cut(src, CFG.P(sh['driver']), start, sh['frames'], fps); cur = start + sh.get('trim_to', sh['frames'])
    q = CFG.DATA / J['job'] / 'QUOTE.md'; os.makedirs(q.parent, exist_ok=True)
    q.write_text("# QUOTE (demo)\nAbout $3.10 (cap $4.00), about 60 min.\n\n## What's different from last time\n| Setting | Old | New |\n|---|---|---|\n| Seed | 42 | 77 |\n"
                 "| Refs | 2048 any bg | 1536 long side, flat RGB |\n| Post | SeedVR2 1080 | native res + trim + audio re-mux |\n")
    # demo 'render' for the post-check: the one_person source padded with 12 extra frozen frames (the Oct 29 frozen-tail bug), then run the real trim+re-mux
    c = J['clips']['one_person']; raw = str(D / 'render_one_person_raw.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', CFG.P(c['audio']['src']), '-vf', f"tpad=stop_mode=clone:stop=12,trim=end_frame={c['frames_out'] + 12}", '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', raw], check=True)
    print('demo media ->', D); return raw
if __name__ == '__main__': main()
