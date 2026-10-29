"""ComfyUI API graphs, built in code (no external templates). Every value is the Oct 29 recipe; builders assert it.
Video: H3 Singularity v1.3 int8 reference-to-video (+ optional 22-frame guide). Stills: Qwen-Image 2.1 white (texfix decode),
Qwen-Image-Edit-2511 fp8 edit (+ Multiple-Angles LoRA 0.9 for angle/back views), SeedVR2 7B image upscale."""
import json
H3_UNET = 'Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors'
H3_TE, H3_VAE, H3_AVAE = 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'minimax_h3_video_vae_fp16.safetensors', 'minimax_h3_audio_vae_fp32.safetensors'
REAL, SWAP, TURBO = 'h3-realism-people-t2v-i2v-r2v.safetensors', 'h3_character_swap_v1.5.safetensors', 'minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors'
# H3LoadReport 'expect' = hooked-layer counts [n, lossy], NEVER strengths (Oct 29 bug: 0.7 written here aborted every run in 1-7 s).
EXPECT = {REAL: [104, 0], SWAP: [208, 0], TURBO: [208, 0]}
SEED = 77
def h3(prompt, refs, driver, W, H, length, fps, realism, prefix, guide=None, seed=SEED):
    """refs: list of input image names (in Picture order). driver: input video name. guide: input video name of the previous piece's 22-frame tail, or None."""
    assert length <= 141, 'pieces are <= 141 frames'
    assert 1 <= len(refs) <= 9, 'H3 takes 1-9 refs'
    exp = {k: v for k, v in EXPECT.items() if realism or k != REAL}
    g = {
     '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': H3_UNET, 'weight_dtype': 'default'}},
     '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': H3_TE, 'type': 'minimax', 'device': 'default'}},
     '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': H3_VAE}},
     '4': {'class_type': 'VAELoader', 'inputs': {'vae_name': H3_AVAE}},
     '6': {'class_type': 'LoadVideo', 'inputs': {'file': driver}},
     '7': {'class_type': 'GetVideoComponents', 'inputs': {'video': ['6', 0]}},
     '80': {'class_type': 'H3ExactLoraStack', 'inputs': {'model': ['1', 0], 'lora_1': REAL if realism else 'None', 'strength_1': 0.7 if realism else 0.0,
            'lora_2': SWAP, 'strength_2': 1.0, 'lora_3': TURBO, 'strength_3': 0.8, 'lora_4': 'None', 'strength_4': 0.0, 'lora_5': 'None', 'strength_5': 0.0}},
     '81': {'class_type': 'H3LoadReport', 'inputs': {'model': ['80', 0], 'tag': prefix.replace('/', '_'), 'require': json.dumps(
            {'max_lossy_merges': 0, 'no_load_warnings': True, 'expect': exp, 'min_hooks': {'fc2': 104}, 'abort': True})}},
     '82': {'class_type': 'MiniMaxH3SigmaShift', 'inputs': {'model': ['81', 0], 'shift_video': 6.0, 'shift_audio': 3.0}},
     '54': {'class_type': 'PathchSageAttentionKJ', 'inputs': {'model': ['82', 0], 'sage_attention': 'auto'}},
     '15': {'class_type': 'MiniMaxH3ReferenceToVideo', 'inputs': {'clip': ['2', 0], 'vae': ['3', 0], 'audio_vae': ['4', 0], 'prompt': prompt, 'width': W, 'height': H,
            'length': length, 'ref_image_size': 'max', 'ref_videos.ref_video_0': ['7', 0], **{f'ref_images.ref_image_{i}': [str(300 + i), 0] for i in range(len(refs))}}},
     '20': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': 'euler'}},
     '21': {'class_type': 'BasicScheduler', 'inputs': {'model': ['54', 0], 'scheduler': 'simple', 'steps': 8, 'denoise': 1.0}},
     '22': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': seed, 'control_after_generate': 'fixed'}},
     '23': {'class_type': 'BasicGuider', 'inputs': {'model': ['54', 0], 'conditioning': ['16' if guide else '15', 0]}},
     '25': {'class_type': 'SamplerCustomAdvanced', 'inputs': {'noise': ['22', 0], 'guider': ['23', 0], 'sampler': ['20', 0], 'sigmas': ['21', 0], 'latent_image': ['15', 1]}},
     '30': {'class_type': 'VAEDecode', 'inputs': {'samples': ['25', 0], 'vae': ['3', 0]}},
     '32': {'class_type': 'CreateVideo', 'inputs': {'images': ['30', 0], 'fps': float(fps), 'bit_depth': 8, 'color_space': 'sRGB', 'codec': 'none'}},
     '33': {'class_type': 'SaveVideo', 'inputs': {'video': ['32', 0], 'filename_prefix': prefix, 'format': 'auto', 'codec': 'auto'}}}
    for i, r in enumerate(refs): g[str(300 + i)] = {'class_type': 'LoadImage', 'inputs': {'image': r}}
    if guide:
        g['400'] = {'class_type': 'LoadVideo', 'inputs': {'file': guide}}; g['401'] = {'class_type': 'GetVideoComponents', 'inputs': {'video': ['400', 0]}}
        g['16'] = {'class_type': 'MiniMaxH3AddGuide', 'inputs': {'positive': ['15', 0], 'latent': ['15', 1], 'vae': ['3', 0], 'image': ['401', 0], 'frame_idx': 0}}
    check_h3(g, realism); return g
def check_h3(g, realism):
    i = g['15']['inputs']; rq = json.loads(g['81']['inputs']['require'])
    assert g['1']['inputs']['unet_name'] == H3_UNET and i['ref_image_size'] == 'max' and i['length'] <= 141
    assert g['21']['inputs']['steps'] == 8 and g['20']['inputs']['sampler_name'] == 'euler' and g['21']['inputs']['scheduler'] == 'simple' and g['82']['inputs']['shift_video'] == 6.0
    assert all(isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) for x in v) for v in rq['expect'].values()), ('expect must be layer counts', rq['expect'])
    assert (REAL in rq['expect']) == bool(realism)
    assert not any(v['class_type'] in ('H3HiresFix', 'FizgigH3Tweaks') for v in g.values()), 'ResCascade/Fizgig are off'
    assert not any('audio' in k and 'vae' not in k for k in i), 'no audio inputs: the driver is muted, source audio goes back on after'
# ---------- stills ----------
QW21, QW21_TE, QW21_VAE, TEXFIX = 'qwen_image_2.1_bf16.safetensors', 'qwen3vl_8b_bf16.safetensors', 'qwen_image_2.1_vae_bf16.safetensors', 'texture_fix_vae_for_qwen_image_2.1_bf16.safetensors'
E2511, E2511_TE, E2511_VAE, MVA = 'qwen_image_edit_2511_fp8mixed.safetensors', 'qwen_2.5_vl_7b_fp8_scaled.safetensors', 'qwen_image_vae.safetensors', 'qwen-image-edit-2511-multiple-angles-lora.safetensors'
SV2, SV2_VAE = 'seedvr2_7b_fp16.safetensors', 'seedvr2_ema_vae_fp16.safetensors'
FACE_WORDS = ('face', 'eyes', 'mouth', 'nose', 'smile', 'expression', 'lips', 'eyebrow')
def qwen21_white(prompt, images, prefix, res=1024, seed=SEED):
    """images[0] = blank white canvas (sets the frame shape); images[1:] = real photos for identity. Decoded with the texture-fix VAE."""
    g = {'1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': QW21, 'weight_dtype': 'default'}},
         '2': {'class_type': 'QwenImage21Cache', 'inputs': {'model': ['1', 0], 'device': 'auto', 'dtype': 'default'}},
         '3': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': QW21_TE, 'type': 'qwen_image', 'device': 'default'}},
         '4': {'class_type': 'VAELoader', 'inputs': {'vae_name': QW21_VAE}}, '5': {'class_type': 'VAELoader', 'inputs': {'vae_name': TEXFIX}},
         '7': {'class_type': 'TextEncodeQwenImage21', 'inputs': {'clip': ['3', 0], 'prompt': prompt, 'negative_prompt': '', 'resolution': res, 'vae': ['4', 0]}},
         '8': {'class_type': 'KSampler', 'inputs': {'model': ['2', 0], 'seed': seed, 'steps': 25, 'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'simple',
               'positive': ['7', 0], 'negative': ['7', 1], 'latent_image': ['7', 2], 'denoise': 1.0}},
         '9': {'class_type': 'VAEDecode', 'inputs': {'samples': ['8', 0], 'vae': ['5', 0]}},
         '10': {'class_type': 'SaveImage', 'inputs': {'images': ['9', 0], 'filename_prefix': prefix}}}
    for i, im in enumerate(images):
        g[str(60 + i)] = {'class_type': 'LoadImage', 'inputs': {'image': im}}; g['7']['inputs'][f'images.image_{i + 1}'] = [str(60 + i), 0]
    assert 'background' in prompt.lower(), 'the background colour goes IN the prompt'
    return g
def edit2511(prompt, image, prefix, angle=False, seed=SEED):
    """edit one image (to black, or to an angle/back view with the Multiple-Angles LoRA 0.9). Stock Qwen VAE (note: partly undoes texfix)."""
    if angle: assert not any(w in prompt.lower() for w in FACE_WORDS) or 'back' not in prompt.lower(), 'back views carry no face words'
    m = '1'; g = {'1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': E2511, 'weight_dtype': 'default'}}}
    if angle: g['1a'] = {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['1', 0], 'lora_name': MVA, 'strength_model': 0.9}}; m = '1a'
    g.update({'2': {'class_type': 'ModelSamplingAuraFlow', 'inputs': {'model': [m, 0], 'shift': 3.1}},
              '3': {'class_type': 'CFGNorm', 'inputs': {'model': ['2', 0], 'strength': 1.0}},
              '4': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': E2511_TE, 'type': 'qwen_image', 'device': 'default'}},
              '5': {'class_type': 'VAELoader', 'inputs': {'vae_name': E2511_VAE}},
              '6': {'class_type': 'LoadImage', 'inputs': {'image': image}},
              '7': {'class_type': 'ImageScaleToTotalPixels', 'inputs': {'image': ['6', 0], 'upscale_method': 'lanczos', 'megapixels': 1.0, 'resolution_steps': 16}},
              '8': {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': {'clip': ['4', 0], 'prompt': prompt, 'vae': ['5', 0], 'image1': ['7', 0]}},
              '9': {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': {'clip': ['4', 0], 'prompt': ' ', 'vae': ['5', 0], 'image1': ['7', 0]}},
              '10': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['7', 0], 'vae': ['5', 0]}},
              '11': {'class_type': 'KSampler', 'inputs': {'model': ['3', 0], 'seed': seed, 'steps': 40, 'cfg': 4.0, 'sampler_name': 'euler', 'scheduler': 'simple',
                     'positive': ['8', 0], 'negative': ['9', 0], 'latent_image': ['10', 0], 'denoise': 1.0}},
              '12': {'class_type': 'VAEDecode', 'inputs': {'samples': ['11', 0], 'vae': ['5', 0]}},
              '13': {'class_type': 'SaveImage', 'inputs': {'images': ['12', 0], 'filename_prefix': prefix}}})
    assert 'background' in prompt.lower()
    return g
def seedvr2_image(image, prefix, short_side=1536, seed=SEED):
    return {'1': {'class_type': 'LoadImage', 'inputs': {'image': image}},
            '2': {'class_type': 'ResizeImageMaskNode', 'inputs': {'input': ['1', 0], 'resize_type': 'scale shorter dimension', 'resize_type.shorter_size': short_side, 'scale_method': 'lanczos'}},
            '3': {'class_type': 'SeedVR2Preprocess', 'inputs': {'resized_images': ['2', 0]}},
            '4': {'class_type': 'VAELoader', 'inputs': {'vae_name': SV2_VAE}}, '5': {'class_type': 'UNETLoader', 'inputs': {'unet_name': SV2, 'weight_dtype': 'default'}},
            '6': {'class_type': 'VAEEncodeTiled', 'inputs': {'pixels': ['3', 0], 'vae': ['4', 0], 'tile_size': 512, 'overlap': 128, 'temporal_size': 64, 'temporal_overlap': 8}},
            '7': {'class_type': 'SeedVR2Conditioning', 'inputs': {'model': ['5', 0], 'vae_conditioning': ['6', 0]}},
            '8': {'class_type': 'KSampler', 'inputs': {'model': ['5', 0], 'positive': ['7', 0], 'negative': ['7', 1], 'latent_image': ['6', 0], 'seed': seed, 'steps': 1, 'cfg': 1.0,
                  'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}},
            '9': {'class_type': 'VAEDecodeTiled', 'inputs': {'samples': ['8', 0], 'vae': ['4', 0], 'tile_size': 512, 'overlap': 128, 'temporal_size': 64, 'temporal_overlap': 8}},
            '10': {'class_type': 'SeedVR2PostProcessing', 'inputs': {'images': ['9', 0], 'original_resized_images': ['2', 0], 'color_correction_method': 'lab'}},
            '11': {'class_type': 'SaveImage', 'inputs': {'images': ['10', 0], 'filename_prefix': prefix}}}
