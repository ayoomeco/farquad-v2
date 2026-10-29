"""Render chains with ON-POD guides (VHS_LoadVideoPath on the previous render in the pod's output dir) and ON-POD assembly:
seam joins (10-frame hold + 12-frame linear crossfade in each 22-frame overlap), pad, original audio mux, H.264 yuv420p encode, side-by-side and a QC contact sheet."""
import os, json, subprocess, time, threading, requests
import gj
C = gj.COMFY
def nframes(p):
    o = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', p], capture_output=True, text=True)
    try: return int(o.stdout.strip())
    except Exception: return -1
def check_loader(job, pid, tag):
    lg = gj.comfy_log(pid); open(job.dir / f'comfylog_{tag}_{pid}.txt', 'w').write(lg)
    mem = [l for l in lg.splitlines() if 'exact_stack.mem' in l][-2:]; lossy = [l for l in lg.splitlines() if 'lossy' in l.lower()][-3:]
    job.log(f'LOADER {pid}: {" | ".join(mem)[:300]} || {" | ".join(lossy)[:300]}')
def lvp(path, skip=0, cap=0):
    return {'class_type': 'VHS_LoadVideoPath', 'inputs': {'video': path, 'force_rate': 0, 'custom_width': 0, 'custom_height': 0, 'frame_load_cap': cap, 'skip_first_frames': skip, 'select_every_nth': 1, 'format': 'None'}}
def render_chain(job, pid, pieces, graphs, refs, drvdir, outdir, fps, guide_name=None):
    """pieces: [(piece, driver, (prev_piece, start) or None)]. Guide = 22 frames of the previous render, loaded on the pod. Returns {piece: (local_mp4, pod_path)}."""
    done = {}
    for k, (pc, drv, guide) in enumerate(pieces):
        if job.stopped(): job.log(f'STOP before {pc}'); break
        up = dict(refs); up[drv] = f'{drvdir}/{drv}'; g = json.loads(json.dumps(graphs[pc]))
        if guide:
            prev, st = guide
            if prev not in done: job.log(f'{pc}: previous {prev} missing; chain stops'); break
            g['400'] = lvp(done[prev][1], st, 22); g.pop('401', None); g['16']['inputs']['image'] = ['400', 0]
        ok, files, err, secs = gj.run_graph(job, pid, g, up, outdir, pc)
        if k == 0: check_loader(job, pid, pc)
        if not ok:
            if err and ('out of memory' in err.lower() or 'outofmemory' in err.lower()): job.log(f'{pc} OOM -> job stops (no setting changes)'); (job.dir / 'STOP').write_text('OOM'); break
            job.log(f'{pc} failed: {str(err)[:300]} -> retry once'); ok, files, err, secs = gj.run_graph(job, pid, g, up, outdir, pc)
            if not ok: job.log(f'{pc} failed twice; chain stops'); break
        i = [n for n, f in enumerate(files) if f.endswith('.mp4')][0]; done[pc] = (files[i], gj.PODFILES[pc][i])
        job.log(f'{pc} done {secs}s frames {nframes(files[i])} spend ${job.spend():.2f}')
    return done
def push(pid, local, name):
    """make a file available on the pod (input dir); returns pod path."""
    gj.upload(pid, local, name); return f'{C}/input/{name}'
def seq_nodes(g, segs, tag):
    """segs: list of ('seg', path, skip, cap) | ('xf', pathA, skipA, pathB, skipB, n). Adds nodes; returns node id of the concatenated IMAGE batch."""
    ids = []
    for i, s in enumerate(segs):
        if s[0] == 'seg':
            k = f'{tag}{i}'; g[k] = lvp(s[1], s[2], s[3]); ids.append(k)
        else:
            _, pa, sa, pb, sb, n = s; a, b, k = f'{tag}{i}a', f'{tag}{i}b', f'{tag}{i}'
            g[a] = lvp(pa, sa, n); g[b] = lvp(pb, sb, n)
            g[k] = {'class_type': 'CrossFadeImages', 'inputs': {'images_1': [a, 0], 'images_2': [b, 0], 'interpolation': 'linear', 'transition_start_index': 0, 'transitioning_frames': n, 'start_level': 0.0, 'end_level': 1.0}}
            ids.append(k)
    if len(ids) == 1: return ids[0]
    k = f'{tag}cat'; g[k] = {'class_type': 'ImageBatchMulti', 'inputs': {'inputcount': len(ids), **{f'image_{j+1}': [x, 0] for j, x in enumerate(ids)}}}; return k
def chain_segs(pieces):
    """pieces: [(pod_path, guide_start_on_prev or None)] -> segs implementing hold-10 + crossfade-12 inside each 22-frame overlap; returns (segs, total)."""
    segs = []; tot = 0
    for k, (p, gs) in enumerate(pieces):
        nxt = pieces[k + 1][1] if k + 1 < len(pieces) else None; start = 22 if k else 0
        if k:
            pp, _ = pieces[k - 1]; segs.append(('seg', pp, gs, 10)); segs.append(('xf', pp, gs + 10, p, 10, 12)); tot += 22
        cap = (nxt - start) if nxt is not None else 0
        segs.append(('seg', p, start, cap)); tot += cap if nxt is not None else None or 0
    return segs
def assembly_graph(name, ren_segs, src_segs, audio_path, fps, pad=None, qc_idx=None):
    g = {}; r = seq_nodes(g, ren_segs, 'r'); s = seq_nodes(g, src_segs, 's')
    if pad:
        for t in (('r', r), ('s', s)):
            k = f'pad{t[0]}'; g[k] = {'class_type': 'ImagePadKJ', 'inputs': {'image': [t[1], 0], 'left': pad[0], 'right': pad[1], 'top': 0, 'bottom': 0, 'extra_padding': 0, 'pad_mode': 'color', 'color': '0, 0, 0'}}
        r, s = 'padr', 'pads'
    if audio_path: g['aud'] = {'class_type': 'VHS_LoadAudio', 'inputs': {'audio_file': audio_path, 'seek_seconds': 0, 'duration': 0}}
    def comb(img, pre): return {'class_type': 'VHS_VideoCombine', 'inputs': {'images': [img, 0], 'audio': ['aud', 0], 'frame_rate': fps, 'loop_count': 0, 'filename_prefix': pre, 'format': 'video/h264-mp4',
                                                                            'pingpong': False, 'save_output': True, 'pix_fmt': 'yuv420p', 'crf': 17, 'save_metadata': False, 'trim_to_audio': False}}
    _comb = comb
    def comb(img, pre):
        n = _comb(img, pre)
        if not audio_path: n['inputs'].pop('audio')
        return n
    g['final'] = comb(r, f'final/{name}')
    g['sbs'] = {'class_type': 'ImageStitch', 'inputs': {'image1': [s, 0], 'image2': [r, 0], 'direction': 'right', 'match_image_size': True, 'spacing_width': 0, 'spacing_color': 'black'}}
    g['sbsv'] = comb('sbs', f'final/{name}_SBS_source_vs_new')
    idx = ','.join(str(i) for i in qc_idx)
    g['qs'] = {'class_type': 'VHS_SelectImages', 'inputs': {'image': [s, 0], 'indexes': idx, 'err_if_missing': False, 'err_if_empty': True}}
    g['qr'] = {'class_type': 'VHS_SelectImages', 'inputs': {'image': [r, 0], 'indexes': idx, 'err_if_missing': False, 'err_if_empty': True}}
    g['qb'] = {'class_type': 'ImageBatch', 'inputs': {'image1': ['qs', 0], 'image2': ['qr', 0]}}
    g['qg'] = {'class_type': 'ImageConcatFromBatch', 'inputs': {'images': ['qb', 0], 'num_columns': len(qc_idx), 'match_image_size': False, 'max_resolution': 8192}}
    g['qsave'] = {'class_type': 'SaveImage', 'inputs': {'images': ['qg', 0], 'filename_prefix': f'final/{name}_QC_contact'}}
    return g
def run_assembly(job, pid, name, g, outdir):
    ok, files, err, secs = gj.run_graph(job, pid, g, {}, outdir, name, timeout=2400)
    if not ok: job.log(f'ASSEMBLY {name} failed: {str(err)[:600]}')
    return ok, files
