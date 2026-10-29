"""Demo of the post-render path ($0): takes the demo 'render' (planned frames + 12 frozen padded frames, no audio), runs the REAL trim-to-planned-frames +
source-audio re-mux from run_video.post(), and writes post_/end_ files so qc/selfcheck.py post can be shown end to end. No pod involved (end file says 0 pods, $0)."""
import sys, os, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'lib')); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'video'))
import fmbcfg as CFG, checks as CK, prep_audio as PA
sys.argv = [sys.argv[0], '--clip', 'one_person']; import run_video as RV
J = CFG.job(); W = CFG.DATA / J['job']; (W / 'final').mkdir(parents=True, exist_ok=True)
raw = str(CFG.DATA / 'demo' / 'render_one_person_raw.mp4'); print('raw render:', CK.probe(raw))
print('audio:', *PA.prep('one_person')); fin, r = RV.post(raw); print('trim + re-mux:', r, '->', fin, CK.probe(fin))
json.dump(dict(end='demo', spend=0.0, cap=J['caps']['one_person'], pieces={}, pods_left=0, pod_prefix='fmb-demo-'), open(W / 'end_one_person.json', 'w'), indent=1)
