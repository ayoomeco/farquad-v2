"""JUTSU LatentSync VAE fix + fail-closed check (Oct 17 2026). Usage: python ls_vae_patch.py <ComfyUI-LatentSyncWrapper dir>
Root cause (pod dd0407wf9914xw): the manifest ships stabilityai/sd-vae-ft-mse *diffusers-format* weights as
checkpoints/sd-vae-ft-mse.safetensors; the wrapper fed it to AutoencoderKL.from_single_file -> "Cannot copy out of meta
tensor" -> it silently built a RANDOM VAE -> colour noise in the mouth. This patch edits scripts/inference.py so that:
  1. checkpoints/vae/{config.json, diffusion_pytorch_model.safetensors -> ../sd-vae-ft-mse.safetensors} is created and
     loaded with from_pretrained(local_files_only, low_cpu_mem_usage=False) (single-file path also gets low_cpu_mem_usage=False);
  2. the random/default-VAE fallback RAISES instead of continuing;
  3. before inference: sha256 of the weight file must equal the manifest hash AND an fp32 encode/decode of a fixed face crop
     (deterministic synthetic gradient/sine/disc frame) must reconstruct at PSNR >= 25 dB, else RuntimeError("JUTSU vae-check FAIL ...") -> node error -> abort rule.
Idempotent; exits non-zero if the anchors are not found (fail closed at install/boot)."""
import pathlib, shutil, sys
MARK = "# JUTSU_VAE_FIX_v1"
SHA = "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815"
CONFIG = ('{"_class_name":"AutoencoderKL","act_fn":"silu","block_out_channels":[128,256,512,512],"down_block_types":'
          '["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],"in_channels":3,'
          '"latent_channels":4,"layers_per_block":2,"norm_num_groups":32,"out_channels":3,"sample_size":256,'
          '"up_block_types":["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"]}')
PRE = f'''    {MARK}
    _jck = os.path.join(base_dir, "checkpoints")
    _jsrc = os.path.join(_jck, "sd-vae-ft-mse.safetensors")
    _jdir = os.path.join(_jck, "vae")
    if os.path.isfile(_jsrc) and not os.path.isfile(os.path.join(_jdir, "config.json")):
        os.makedirs(_jdir, exist_ok=True)
        open(os.path.join(_jdir, "config.json"), "w").write({CONFIG!r})
        _jl = os.path.join(_jdir, "diffusion_pytorch_model.safetensors")
        if not os.path.lexists(_jl):
            os.symlink(os.path.realpath(_jsrc), _jl)
        print("[jutsu-vae] diffusers vae/ dir prepared ->", os.path.realpath(_jsrc))
'''
POST = f'''    {MARK} fail-closed check (weights hash + reconstruction) BEFORE any inference
    def _jutsu_vae_check(_vae):
        import hashlib, copy, numpy as _np, torch as _t
        _w = os.path.realpath(os.path.join(base_dir, "checkpoints", "sd-vae-ft-mse.safetensors"))
        _h = hashlib.sha256()
        with open(_w, "rb") as _f:
            for _b in iter(lambda: _f.read(1 << 24), b""): _h.update(_b)
        if _h.hexdigest() != "{SHA}":
            raise RuntimeError("JUTSU vae-check FAIL: weight sha256 " + _h.hexdigest() + " != manifest")
        _yy, _xx = _np.mgrid[0:128, 0:128].astype(_np.float32) / 127.0   # deterministic synthetic frame (no file in the payload)
        _img = _np.stack([0.8 * _xx - 0.4, 0.6 * _np.sin(6.28 * _yy) * 0.5, 0.7 * (((_xx - .5) ** 2 + (_yy - .5) ** 2) < .09) - 0.35], -1).astype(_np.float32)
        _x = _t.from_numpy(_img).permute(2, 0, 1)[None]
        _m = copy.deepcopy(_vae).float().cpu().eval()
        with _t.no_grad():
            _y = _m.decode(_m.encode(_x).latent_dist.mean).sample.clamp(-1, 1)
        _mse = float(((_y - _x) / 2).pow(2).mean())
        _psnr = 10 * _np.log10(1.0 / max(_mse, 1e-12))
        del _m
        print(f"[jutsu-vae] check sha256 OK, recon PSNR {{_psnr:.2f}} dB")
        if not _psnr >= 25.0:
            raise RuntimeError(f"JUTSU vae-check FAIL: recon PSNR {{_psnr:.2f}} dB < 25 (random/default VAE?)")
    _jutsu_vae_check(vae)
'''


def main(d):
    d = pathlib.Path(d); p = d / "scripts" / "inference.py"; s = p.read_text()
    if MARK in s:
        print("[ls-vae] already patched"); return 0
    a1 = "    # Try multiple VAE locations in order of preference\n"
    a2 = "AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)"
    a3 = "AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype, local_files_only=True)"
    a4 = '        print("Local VAE not found in any location, creating VAE with standard configuration")\n'
    a5 = "    # Set VAE configuration\n"
    miss = [a for a in (a1, a2, a3, a4, a5) if a not in s]
    if miss:
        print("[ls-vae] FATAL anchors not found:", [m[:50] for m in miss]); return 1
    s = s.replace(a1, PRE + a1, 1)
    s = s.replace(a2, "AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype, low_cpu_mem_usage=False)", 1)
    s = s.replace(a3, "AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=False)", 1)
    s = s.replace(a4, '        raise RuntimeError("JUTSU vae-check FAIL: no real VAE weights loaded; refusing the random default VAE " + str(vae_locations))\n' + a4, 1)
    s = s.replace(a5, POST + a5, 1)
    p.write_text(s); print("[ls-vae] patched scripts/inference.py (vae dir + no random fallback + pre-inference check)"); return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
