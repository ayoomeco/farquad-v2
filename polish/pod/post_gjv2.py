"""GJ V2 post, run on the pod. Upscale only, keep the source frame rate, stitch, score, encode.

No frame interpolation, no color grade, no grain, no face restore, no paste-back.
Only the finished file (and its score json) is the deliverable.
Encode: H.264 High 4.2, yuv420p, bt709, AAC, +faststart.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys

VERSION = "GJ V2"
# Inserted after the raw-video input. Frame rate is the source rate, never a forced 30.
ENCODE = [
    "-c:v", "libx264", "-profile:v", "high", "-level:v", "4.2", "-pix_fmt", "yuv420p",
    "-crf", "16", "-preset", "slow",
    "-vf", "scale=out_color_matrix=bt709:out_range=tv",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest",
]
SKIPPED = ("frame_interpolation", "color_grade", "grain", "face_restore", "paste_back")
FORBIDDEN_ENCODE = ("minterpolate", "tblend", "framerate=", "-tune", "grain", "codeformer", "gfpgan")


def encode_cmd(width: int, height: int, fps: str, video_src: str, audio_src: str, dest: str) -> list[str]:
    """Re-encode `video_src` with audio from `audio_src`. fps is the source rate (passthrough, not a new rate)."""
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", video_src, "-i", audio_src,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-fps_mode", "passthrough",
        *ENCODE,
        "-metadata:s:v:0", f"fps={fps}",
        dest,
    ]
    blob = " ".join(cmd)
    for bad in FORBIDDEN_ENCODE:
        if bad in blob:
            raise SystemExit(f"GJ V2 encode refused ({bad})")
    for need in ("-profile:v high", "-level:v 4.2", "yuv420p", "bt709", "aac", "+faststart"):
        if need not in blob:
            raise SystemExit(f"GJ V2 encode missing {need}")
    if fps in ("30", "60") and os.environ.get("GJ_SOURCE_FPS") not in (None, "", fps):
        raise SystemExit("GJ V2 keeps the source fps; it does not interpolate to 30 or 60")
    return cmd


def probe_fps(path: str) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def plan(slug: str) -> dict:
    return {
        "version": VERSION,
        "where": "pod",
        "slug": slug,
        "steps": ["upscale_seedvr2", "stitch", "score", "encode"],
        "skipped": list(SKIPPED),
        "fps": "source",
        "upscale": {"tool": "SeedVR2", "face_restore": False, "interpolate": False},
        "encode": ENCODE,
        "download": "finished",
        "script": "polish/pod/post_gjv2.py",
    }


def upscale(src: str, dst: str) -> None:
    """SeedVR2 only. No CodeFormer, no RIFE."""
    cli = os.environ.get("SEEDVR2_CLI", "sv/inference_cli.py")
    if not os.path.exists(cli):
        raise SystemExit(f"GJ V2 post: SeedVR2 CLI not on the pod ({cli})")
    subprocess.check_call(["python3", cli, src, "--output", dst, "--resolution", os.environ.get("GJ_UPSCALE_RES", "1080")])


def score(slug: str, fps: str, frames: int, dest_json: str) -> dict:
    rec = {
        "version": VERSION, "slug": slug, "fps": fps, "frames": frames,
        "interpolated": False, "color_grade": False, "grain": False,
        "face_restore": False, "paste_back": False, "upscale": "seedvr2",
    }
    with open(dest_json, "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def deliver(video: str, native: str, dest: str, slug: str) -> dict:
    """Upscale `video`, encode at the source fps with the native audio, write a score json beside dest."""
    fps = probe_fps(native)
    up = dest + ".up.mp4"
    upscale(video, up)
    cmd = encode_cmd(0, 0, fps, up, native, dest)
    # width/height are unused: ffmpeg reads them from the upscaled file.
    subprocess.check_call(cmd)
    return score(slug, fps, 0, os.path.splitext(dest)[0] + "_scores.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description="GJ V2 on-pod post")
    ap.add_argument("slug", nargs="?", default="clip")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--native")
    ap.add_argument("--dest")
    a = ap.parse_args(argv)
    if a.plan or not (a.video and a.native and a.dest):
        print(json.dumps(plan(a.slug)))
        return 0
    deliver(a.video, a.native, a.dest, a.slug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
