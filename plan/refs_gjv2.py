"""GJ V2 reference images. Separate crops, never a combined character sheet.

Order: neutral closed-mouth head close-up first, body only on a full-body shot, outfit last.
Crops are upscaled with SeedVR2 in image mode to a 2048 short side. No face restore.
"""
from __future__ import annotations
import os

VERSION = "GJ V2"
SHORT_SIDE = 2048
ROLES_FULL = ("face", "body", "outfit")
ROLES_MED = ("face", "outfit")
# A single collage / 9-piece pack / character sheet is not a ref.
SHEET_MARKERS = ("sheet", "combined", "9pack", "9-piece", "9_piece", "character_sheet", "char_sheet", "grid")

ROLE_TEXT = {
    "face": "image {n} face: a neutral closed-mouth head close-up; use it only for the face and head",
    "body": "image {n} body: use it only for the body",
    "outfit": "image {n} outfit: use it only for the clothing",
}


def roles(full_body: bool) -> tuple[str, ...]:
    return ROLES_FULL if full_body else ROLES_MED


def filenames(key: str, full_body: bool) -> list[str]:
    return [f"v5b_{key}_{role}.png" for role in roles(full_body)]


def is_sheet(path: str) -> bool:
    name = os.path.basename(path).lower()
    return any(m in name for m in SHEET_MARKERS)


def reject_sheet(path: str) -> None:
    if is_sheet(path):
        raise ValueError(f"GJ V2 refs are separate images, never a combined character sheet: {path}")


def seedvr2_image_cmd(src: str, dst: str, short: int = SHORT_SIDE, cli: str = "inference_cli.py") -> list[str]:
    """SeedVR2 image mode. Short side `short`. No face-restore flag and no CodeFormer stage."""
    reject_sheet(src)
    reject_sheet(dst)
    return [cli, src, "--output", dst, "--mode", "image", "--resolution", str(int(short))]
