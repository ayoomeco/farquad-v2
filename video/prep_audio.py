"""Cut each clip's source audio window -> $FMB_DATA/<job>/audio/<clip>.wav ($0, ffmpeg). audio: {src, start (s), dur (s)}; dur defaults to frames_out / fps.
If a clip has no audio block, it is flagged SILENT (tell the owner up front)."""
import sys, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'lib'))
import fmbcfg as CFG, checks as CK
def prep(cid):
    J = CFG.job(); c = J['clips'][cid]; a = c.get('audio'); O = CFG.DATA / J['job'] / 'audio'; O.mkdir(parents=True, exist_ok=True)
    if not a: return None, 'SILENT deliverable (no audio block): flag it to the owner before the quote'
    out = str(O / f'{cid}.wav'); ok, err = CK.cut_audio(CFG.P(a['src']), float(a['start']), float(a.get('dur', c['frames_out'] / c['fps'])), out)
    return (out if ok else None), ('ok' if ok else f'audio cut failed: {err}')
if __name__ == '__main__':
    J = CFG.job()
    for cid in (sys.argv[1:] or list(J['clips'])): print(cid, *prep(cid))
