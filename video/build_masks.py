"""Who's-who maps for multi-person shots ($0, CPU): YOLO11m-seg + ByteTrack person tracks -> one role per subject (by screen side, largest tracks)
-> SAM2.1-tiny box prompt (box +12 px) on every frame. H3 has NO mask input: these maps are for (a) writing each subject's position into the prompt
and (b) checking every rendered piece (right person, no leftover original, no ID swap). Writes $FMB_DATA/<job>/masks/<clip>_<piece>_<character>.mkv,
overlay_<clip>_<piece>.mp4 and masks_report.json (missing frames, extra-person frames = crowd/ID-chaos warning).
Shot fields used: subjects[].becomes and subjects[].side ('left'|'right'|'center', screen side in the source).   python3 build_masks.py <clip> [piece ...]"""
import sys, os, json, subprocess, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'lib'))
import fmbcfg as CFG
def main():
    import cv2, numpy as np
    from ultralytics import YOLO, SAM
    J = CFG.job(); cid = sys.argv[1]; c = J['clips'][cid]; O = CFG.DATA / J['job'] / 'masks'; O.mkdir(parents=True, exist_ok=True)
    seg = YOLO(str(CFG.CV / 'yolo11m-seg.pt')); sam = SAM(str(CFG.CV / 'sam2.1_t.pt')); W, H = c['W'], c['H']
    rp = O / 'masks_report.json'; rep = json.load(open(rp)) if rp.exists() else {}
    COL = [(0, 0, 255), (255, 120, 0), (0, 200, 0), (200, 0, 200), (0, 200, 200)]
    for s in c['shots']:
        if len(s['subjects']) < 2 or (sys.argv[2:] and s['piece'] not in sys.argv[2:]): continue
        cap = cv2.VideoCapture(CFG.P(s['driver'])); fps = cap.get(5); Fr = []
        while True:
            ok, f = cap.read()
            if not ok: break
            Fr.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
        tr = {}; boxes = []
        for f in Fr:
            r = seg.track(f, persist=True, classes=[0], tracker='bytetrack.yaml', verbose=False)[0]
            b = {} if r.boxes.id is None else {int(t): bb.tolist() for t, bb in zip(r.boxes.id, r.boxes.xyxy)}; boxes.append(b)
            for t, bb in b.items(): tr.setdefault(t, []).append(((bb[0] + bb[2]) / 2, (bb[2] - bb[0]) * (bb[3] - bb[1])))
        n = len(s['subjects']); big = sorted(tr, key=lambda t: -sum(a for _, a in tr[t]))[:n]; key = f"{cid}_{s['piece']}"
        if len(big) < n: rep[key] = dict(error=f'{len(big)} main tracks < {n} subjects (edge person untracked?)', tracks=len(tr)); continue
        cx = {t: np.mean([x for x, _ in tr[t]]) for t in big}; order = sorted(big, key=lambda t: cx[t])
        sides = {'left': 0, 'center': n // 2, 'right': n - 1}; role = {}
        for i, sub in enumerate(s['subjects']): role[sub['becomes']] = order[sides.get(sub.get('side'), i)]
        wr = {ch: subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-', '-c:v', 'ffv1', str(O / f'{key}_{ch}.mkv')], stdin=subprocess.PIPE) for ch in role}
        ov = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', str(O / f'overlay_{key}.mp4')], stdin=subprocess.PIPE)
        miss = {ch: [] for ch in role}; extra = []
        for i, f in enumerate(Fr):
            o = f.copy()
            if [t for t in boxes[i] if t not in role.values()]: extra.append(i)
            for j, (ch, t) in enumerate(role.items()):
                m = np.zeros((H, W), np.uint8)
                if t in boxes[i]:
                    x0, y0, x1, y1 = boxes[i][t]; rr = sam(f, bboxes=[[max(0, x0 - 12), max(0, y0 - 12), min(W, x1 + 12), min(H, y1 + 12)]], verbose=False)[0].masks
                    if rr is not None: m = (cv2.resize(rr.data.cpu().numpy().max(0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) * 255
                if not m.any(): miss[ch].append(i)
                wr[ch].stdin.write(m.tobytes()); o[m > 0] = (0.5 * o[m > 0] + 0.5 * np.array(COL[j % 5])).astype(np.uint8)
                if t in boxes[i]: cv2.putText(o, ch, (int(boxes[i][t][0]), int(boxes[i][t][1]) + 24), 0, 0.8, COL[j % 5], 2)
            cv2.putText(o, f"{key} f{i}", (10, 30), 0, 0.8, (255, 255, 255), 2); ov.stdin.write(o.tobytes())
        for p in list(wr.values()) + [ov]: p.stdin.close(); p.wait()
        rep[key] = dict(frames=len(Fr), fps=fps, tracks={ch: int(t) for ch, t in role.items()}, missing={ch: v[:50] for ch, v in miss.items()}, n_extra_person_frames=len(extra))
        print(key, rep[key], flush=True)
    json.dump(rep, open(rp, 'w'), indent=1)
if __name__ == '__main__': main()
