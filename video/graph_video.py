"""Build every H3 graph for the job (FMB_JOB) from its clips + characters. $0, local. Prints a per-piece table and writes $FMB_DATA/<job>/graphs_video.json.
Refs used are the PREPARED ones (prep_refs.py -> $FMB_DATA/<job>/refs/<file>), fed by file name in Picture order."""
import sys, os, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'lib'))
import fmbcfg as CFG, graphs as G, prompts as PR
def build(clip_id):
    J = CFG.job(); c = J['clips'][clip_id]; chars = J['characters']; P = []
    for k, s in enumerate(c['shots']):
        prompt, refs, human = PR.build(s, chars, c['scene'])
        names = [os.path.basename(r) for r in refs]
        g = G.h3(prompt, names, os.path.basename(s['driver']), c['W'], c['H'], s['frames'], c['fps'], human, f"fmb/{J['job']}/{clip_id}_{s['piece']}_s{J.get('seed', 77)}",
                 guide=('guide.mp4' if s.get('guide') else None), seed=J.get('seed', 77))
        P.append(dict(piece=s['piece'], graph=g, driver=CFG.P(s['driver']), refs=names, guide=s.get('guide'), frames=s['frames'], est=s['est'],
                      trim_to=s.get('trim_to', s['frames']), realism=0.7 if human else 0.0, prompt=prompt))
    return dict(clip=clip_id, name=c.get('name', clip_id), fps=c['fps'], W=c['W'], H=c['H'], pieces=P, assembly=c.get('assembly', 'shots'),
                frames_out=c['frames_out'], audio=c.get('audio'), cap=J['caps'][clip_id])
def refs_dir(): return CFG.DATA / CFG.job()['job'] / 'refs'
if __name__ == '__main__':
    J = CFG.job(); out = {}; os.makedirs(CFG.DATA / J['job'], exist_ok=True)
    for cid in (sys.argv[1:] or list(J['clips'])):
        C = build(cid); out[cid] = C
        print(f"{cid} ({C['name']}): {C['W']}x{C['H']} @ {C['fps']} fps, {len(C['pieces'])} pieces -> {C['frames_out']} frames, cap ${C['cap']}")
        for p in C['pieces']: print(f"   {p['piece']:8s} {p['frames']:4d} f  est {p['est']}s  realism {p['realism']}  refs {p['refs']}  guide {p['guide']}")
    json.dump(out, open(CFG.DATA / J['job'] / 'graphs_video.json', 'w'), indent=1); print('graphs asserted (8 steps euler/simple shift 6, seed, LoRAs, expect=layer counts, no ResCascade/Fizgig, no audio inputs) -> graphs_video.json')
