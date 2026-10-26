"""A08 default gates: graph, load check, and final encode. No GPU, no keys, no network."""
import importlib.util, json, os, pathlib, shutil, subprocess, sys, tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
PHRASE_MOUTH = "match the source's mouth and expression exactly"
PHRASE_OBJECTS = "keep any objects already in the scene; add nothing"
TURBO = 'minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors'
REALISM = 'h3-realism-people-t2v-i2v-r2v.safetensors'
SWAP = 'h3_character_swap_v1.5.safetensors'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    check = load_module('h3check', REPO / 'engine/worker/custom_nodes/h3_exact_lora/check.py')
    post = load_module('post_a08', REPO / 'engine/worker/post_a08.py')
    home = pathlib.Path(tempfile.mkdtemp(prefix='a08home_'))
    os.environ['FARQUAD_HOME'] = str(home)
    os.environ.pop('LIKENESS_LORA', None)
    os.environ.pop('LIKENESS_STRENGTH', None)
    shutil.copytree(REPO / 'plan/base_graphs', home / 'plan/base_graphs')
    (home / 'plan/graphs').mkdir(parents=True)
    sys.path.insert(0, str(REPO / 'plan'))
    import graphs_v5b as G

    def built(slug='walken', **kw):
        info = G.graph(slug, 0, kw.get('guide'), kw.get('seed', G.SEED), (), 0)
        return json.loads(pathlib.Path(info['graph_path']).read_text()), info

    g, info = built()
    assert info['params']['steps'] == 8 and info['seed'] == 42
    assert g['1']['inputs']['unet_name'].endswith('Pruned_v1.3_int8.safetensors')
    assert not any(n.get('class_type') == 'LoraLoaderModelOnly' for n in g.values())
    blob = json.dumps(g)
    assert 'pro4500' not in blob and 'turbo_v4' not in blob and 'm33k' not in blob
    stack = g['80']['inputs']
    assert stack['lora_1'] == REALISM and stack['strength_1'] == 0.7
    assert stack['lora_2'] == SWAP and stack['strength_2'] == 1.0
    assert stack['lora_3'] == TURBO and stack['strength_3'] == 0.8
    assert stack['lora_4'] == 'None' and stack['strength_4'] == 0.0
    assert stack['lora_5'] == 'None' and stack['strength_5'] == 0.0
    req = json.loads(g['81']['inputs']['require'])
    assert req['abort'] is True and req['max_lossy_merges'] == 0 and req['no_load_warnings'] is True
    assert req['min_hooks']['fc2'] == 104 and req['min_hooks']['qkv_proj'] == 1
    assert req['injection_key'] == 'h3_exact_stack' and req['min_comfyui'] == '0.42.5'
    assert req['strengths'][TURBO] == 0.8 and req['expect'][REALISM] == [104, 0]
    assert g['82']['inputs']['shift_video'] == 6.0 and g['82']['inputs']['shift_audio'] == 3.0
    assert g['82']['inputs']['model'] == ['81', 0]
    assert g['54']['inputs']['model'] == ['82', 0] and g['54']['inputs']['sage_attention'] == 'auto'
    assert g['21']['inputs']['steps'] == 8 and g['21']['inputs']['scheduler'] == 'simple'
    assert g['20']['inputs']['sampler_name'] == 'euler'
    assert g['22']['inputs']['noise_seed'] == 42
    assert g['15']['inputs']['ref_image_size'] == 'match'
    assert g['15']['inputs']['width'] == 1152 and g['15']['inputs']['height'] == 640
    prompt = g['15']['inputs']['prompt']
    assert PHRASE_MOUTH in prompt and PHRASE_OBJECTS in prompt
    assert "image 1 is <Picture 1>, Patrick Bateman's face" in prompt
    assert "image 2 is <Picture 2>, Patrick Bateman's body" in prompt
    assert "image 3 is <Picture 3>, Patrick Bateman's outfit" in prompt
    assert [g[str(200 + i)]['inputs']['image'] for i in range(3)] == [
        'v5b_bateman_face.png', 'v5b_bateman_body.png', 'v5b_bateman_outfit.png']

    america, _ = built('thisisamerica')
    assert america['80']['inputs']['lora_1'] == REALISM and america['80']['inputs']['strength_1'] == 0.7

    bare = dict(G.CLIPS['walken'])
    bare['full_body'] = False
    text, refs = G.prompt_for(bare)
    assert refs == ['v5b_bateman_face.png', 'v5b_bateman_outfit.png']
    assert "image 1 is <Picture 1>, Patrick Bateman's face" in text
    assert "image 2 is <Picture 2>, Patrick Bateman's outfit" in text
    assert "_body.png" not in text and PHRASE_MOUTH in text and PHRASE_OBJECTS in text

    guided, _ = built(guide='guide.mp4')
    assert guided['72']['class_type'] == 'MiniMaxH3AddGuide'
    assert guided['23']['inputs']['conditioning'] == ['72', 0]

    os.environ['LIKENESS_LORA'] = 'm33k_lme_s0500.safetensors'
    os.environ['LIKENESS_STRENGTH'] = '0.5'
    lik, _ = built()
    assert lik['80']['inputs']['lora_4'] == 'm33k_lme_s0500.safetensors'
    assert lik['80']['inputs']['strength_4'] == 0.5
    lik_req = json.loads(lik['81']['inputs']['require'])
    assert lik_req['min_hooks']['fc2'] == 154
    assert lik_req['expect']['m33k_lme_s0500.safetensors'] == [200, 0]
    os.environ['LIKENESS_STRENGTH'] = '0'
    off, _ = built()
    assert off['80']['inputs']['lora_4'] == 'None'
    os.environ['LIKENESS_LORA'] = 'custom_person.safetensors'
    os.environ['LIKENESS_STRENGTH'] = '0.4'
    custom, _ = built()
    creq = json.loads(custom['81']['inputs']['require'])
    assert custom['80']['inputs']['lora_4'] == 'custom_person.safetensors'
    assert 'custom_person.safetensors' not in creq['expect']
    assert creq['strengths']['custom_person.safetensors'] == 0.4
    assert creq['min_hooks']['fc2'] == 104
    os.environ.pop('LIKENESS_LORA', None)
    os.environ.pop('LIKENESS_STRENGTH', None)

    gb = json.loads((REPO / 'plan/base_graphs/gb_S1_t2.api.json').read_text())
    gj = json.loads((REPO / 'plan/base_graphs/gjB_p1_s42.api.json').read_text())
    for base in (gb, gj):
        raw = json.dumps(base)
        assert 'pro4500' not in raw and 'turbo_v4' not in raw
        assert PHRASE_MOUTH in base['15']['inputs']['prompt'] and PHRASE_OBJECTS in base['15']['inputs']['prompt']
        assert base['21']['inputs']['steps'] == 8 and base['20']['inputs']['sampler_name'] == 'euler'
        assert base['21']['inputs']['scheduler'] == 'simple'
    assert gb['80']['class_type'] == 'H3ExactLoraStack'
    assert gj['72']['class_type'] == 'MiniMaxH3AddGuide'
    assert gj['80']['class_type'] == 'ModelPatchLoader'
    assert gj['96']['inputs']['shift_video'] == 6.0

    def report(lora_rows, hooks, lossy=0, injection=520, comfy='0.42.5'):
        return {
            'comfyui': comfy,
            'probe': {'lossy_merges': lossy, 'injection_keys': {'h3_exact_stack': injection} if injection else {}},
            'exact_stack': {'loras': lora_rows, 'hooks_by_type': hooks},
        }

    rows = [
        {'name': REALISM, 'strength': 0.7, 'adapters': 104, 'dense_modules': 0, 'n_load_warnings': 0, 'unsupported': []},
        {'name': SWAP, 'strength': 1.0, 'adapters': 208, 'dense_modules': 0, 'n_load_warnings': 0, 'unsupported': []},
        {'name': TURBO, 'strength': 0.8, 'adapters': 208, 'dense_modules': 0, 'n_load_warnings': 0, 'unsupported': []},
    ]
    hooks = {'qkv_proj': {'bf16': 1}, 'out_proj': {'bf16': 1}, 'fc1': {'bf16': 1}, 'fc2': {'bf16': 50, 'int8/quant': 54}}
    good = report(rows, hooks)
    assert check.evaluate(good, req) == []
    bad_lossy = report(rows, hooks, lossy=2)
    assert any('lossy' in f for f in check.evaluate(bad_lossy, req))
    missing_key = dict(good)
    missing_key['probe'] = {'injection_keys': {}}
    assert any('lossy_merges missing' in f for f in check.evaluate(missing_key, req))
    weak = json.loads(json.dumps(rows))
    weak[2]['strength'] = 1.0
    assert any('strength' in f for f in check.evaluate(report(weak, hooks), req))
    dropped = rows[:2]
    assert any('missing' in f for f in check.evaluate(report(dropped, hooks), req))
    short_hooks = {'qkv_proj': {'bf16': 1}, 'out_proj': {'bf16': 1}, 'fc1': {'bf16': 1}, 'fc2': {'bf16': 10}}
    assert any('fc2' in f for f in check.evaluate(report(rows, short_hooks), req))
    assert any('comfyui' in f for f in check.evaluate(report(rows, hooks, comfy='0.39.1'), req))
    assert check.evaluate(report(rows, hooks, comfy='0.43.0'), req) == []
    no_inj = report(rows, hooks, injection=0)
    assert any('injection' in f for f in check.evaluate(no_inj, req))

    cmd = post.encode_cmd(1152, 640, '24000/1001', 'out.mp4', audio_from='native.mp4')
    joined = ' '.join(cmd)
    for flag in ('-profile:v', 'high', '-level:v', '4.2', 'yuv420p', 'bt709', 'aac', '+faststart'):
        assert flag in cmd, flag
    assert '-r' in cmd and cmd[cmd.index('-r') + 1] == '24000/1001'
    assert '30' not in cmd[cmd.index('-r'):cmd.index('-r') + 2]
    assert '-preset' in cmd and cmd[cmd.index('-preset') + 1] == 'slow'

    if shutil.which('ffmpeg') and shutil.which('ffprobe'):
        td = pathlib.Path(tempfile.mkdtemp(prefix='a08enc_'))
        raw = td / 'f.mp4'
        wav = td / 'a.wav'
        dst = td / 'out.mp4'
        subprocess.check_call(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=gray:s=64x64:r=24',
                               '-frames:v', '4', '-pix_fmt', 'yuv420p', str(raw)])
        subprocess.check_call(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000',
                               '-t', '0.2', str(wav)])
        frames_mod = load_module('post_a08_frames', REPO / 'engine/worker/post_a08.py')
        frames = frames_mod.read_bgr(str(raw))
        frames_mod.write_raw(frames, str(dst), '24/1', str(wav))
        ok, why = frames_mod.probe_ok(str(dst), '24/1')
        assert ok, why
        shutil.rmtree(td)
    shutil.rmtree(home)

    for script, env in ((REPO / 'polish/pod/s3.py', {}), (REPO / 'polish/pod/s4.py', {})):
        proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env={**os.environ, **env})
        assert proc.returncode != 0, script
        assert 'A08' in (proc.stderr + proc.stdout)

    sys.path.insert(0, str(REPO / 'engine'))
    from app.genjutsu.h3_prompt import context_ir
    from app.genjutsu.shotspec import Ref, ShotSpec
    spec = ShotSpec(mode='character_swap', source_video='clip.mp4', subject='human',
                    refs=[Ref(path='f.png'), Ref(path='b.png'), Ref(path='o.png')])
    text = context_ir(spec, 3, 5.0)
    assert PHRASE_MOUTH in text and PHRASE_OBJECTS in text
    assert 'image 1 is the face, image 2 is the body, image 3 is the outfit.' in text
    assert 'image 2 is the outfit.' in context_ir(spec, 2, 5.0)
    print('A08_DEFAULTS_OK')


if __name__ == '__main__':
    main()
