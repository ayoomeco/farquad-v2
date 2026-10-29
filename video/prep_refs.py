"""Prepare refs ($0, local, no AI): each character's head/body/back ref -> Lanczos to 1536 on the LONG side, flattened to 8-bit RGB (no alpha;
H3 drops alpha and shows whatever colour is under it). Writes $FMB_DATA/<job>/refs/<basename> + refs_map.json. Smaller sources are NOT upscaled here (flagged)."""
import sys, os, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'lib'))
import fmbcfg as CFG
from PIL import Image
def prep():
    J = CFG.job(); L = J.get('ref_long_side', 1536); O = CFG.DATA / J['job'] / 'refs'; O.mkdir(parents=True, exist_ok=True); out = {}
    for ch, c in J['characters'].items():
        for kind, src in c['refs'].items():
            src = CFG.P(src); n = os.path.basename(src); im = Image.open(src)
            if im.mode in ('RGBA', 'LA', 'P'):
                im = im.convert('RGBA'); bg = Image.new('RGB', im.size, tuple(c.get('flatten_on', (0, 0, 0)))); bg.paste(im, mask=im.getchannel('A')); im = bg
            im = im.convert('RGB'); s = L / max(im.size); flag = None
            if s < 1: im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            elif s > 1: flag = f'source long side {max(im.size)} < {L}: upscale it first (SeedVR2 image mode), never stretch'
            im.save(O / n); out[n] = dict(character=ch, kind=kind, src=src, size=im.size, flag=flag)
    json.dump(out, open(O / 'refs_map.json', 'w'), indent=1); return out
if __name__ == '__main__':
    for k, v in prep().items(): print(f"{k:24s} {v['size'][0]}x{v['size'][1]} {v['character']} {v['kind']}" + (f"  FLAG {v['flag']}" if v['flag'] else ''))
