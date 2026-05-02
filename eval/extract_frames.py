#!/usr/bin/env python3
"""
Extract up to 800 uniformly sampled frames from each video as 896x896 JPEGs
for SigLIP-based vision-language models (MedGemma, Qwen, Hulu-Med, ColonR1).
1350x1080 -> crop 135 each side -> 1080x1080 -> resize 896x896
900x720 -> crop 2 each side -> 896x720 -> pad 88 top+bottom -> 896x896

Usage:
    python eval/extract_frames.py
    python eval/extract_frames.py --cases-dir data/cases/
    python eval/extract_frames.py --workers 8
    python eval/extract_frames.py --case-ids e1012 c1003

Reads defaults from ../config.yaml if present.
Skips cases that already have a _siglip_896/ directory with files.
"""

import argparse
import os
import re
import sys
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── Defaults ─────────────────────────────────────────────

DEFAULT_WORKERS = 4
MAX_FRAMES = 800
TARGET_SIZE = 896
JPEG_QUALITY = 95
SUBDIR_SUFFIX = "_siglip_896"

_CASE_ID_RE = re.compile(r"^[ec]\d+$")


# ── Transforms ───────────────────────────────────────────

def transform_1080p(frame: np.ndarray) -> np.ndarray:
    """1350x1080 -> crop 135 each side -> 1080x1080 -> resize 896x896."""
    cropped = frame[:, 135:1350 - 135]  # 1080x1080
    return cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE),
                      interpolation=cv2.INTER_AREA)


def transform_720p(frame: np.ndarray) -> np.ndarray:
    """900x720 -> crop 2 each side -> 896x720 -> pad 88 top+bottom -> 896x896."""
    cropped = frame[:, 2:900 - 2]  # 896x720
    return cv2.copyMakeBorder(cropped, 88, 88, 0, 0,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def transform_generic(frame: np.ndarray) -> np.ndarray:
    """Generic fallback: center-crop to square, then resize to 896x896."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = frame[y0:y0 + side, x0:x0 + side]
    return cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE),
                      interpolation=cv2.INTER_AREA)


def get_transform(width: int, height: int):
    """Pick the right transform based on video resolution."""
    if width == 1350:
        return transform_1080p, True   # (transform, burn_frame_num)
    elif width == 900:
        return transform_720p, False
    else:
        return transform_generic, False


# ── Per-video extraction ─────────────────────────────────

def process_video(video_path: str, out_dir: str, case_id: str,
                  max_frames: int = MAX_FRAMES) -> int:
    """Extract frames from one video. Returns number of frames saved."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    transform, burn_frame_num = get_transform(width, height)

    n = min(max_frames, total)
    indices = np.linspace(0, total - 1, n, dtype=int).tolist()

    os.makedirs(out_dir, exist_ok=True)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    saved = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = transform(frame)
        if burn_frame_num:
            cv2.putText(frame, str(idx), (8, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        filename = f"{case_id}_f{idx:05d}.jpg"
        cv2.imwrite(os.path.join(out_dir, filename), frame, encode_params)
        saved += 1

    cap.release()
    return saved


# ── Config loading ───────────────────────────────────────

def load_config_defaults() -> dict:
    """Try to read benchmark/config.yaml for path and frame defaults."""
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir.parent / "config.yaml"

    defaults = {
        "cases_dir": str(script_dir.parent / "data" / "cases"),
        "max_frames": MAX_FRAMES,
        "subdir_suffix": SUBDIR_SUFFIX,
        "workers": DEFAULT_WORKERS,
    }

    if not config_path.exists():
        return defaults

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

        paths = cfg.get("paths", {})
        if "cases_dir" in paths:
            p = paths["cases_dir"]
            if not os.path.isabs(p):
                p = str(script_dir.parent / p)
            defaults["cases_dir"] = p

        frames = cfg.get("frames", {})
        if "max_frames" in frames:
            defaults["max_frames"] = frames["max_frames"]
        if "subdir_suffix" in frames:
            defaults["subdir_suffix"] = frames["subdir_suffix"]
        if "workers" in frames:
            defaults["workers"] = frames["workers"]

    except ImportError:
        pass  # pyyaml not installed — use hardcoded defaults

    return defaults


# ── Main ─────────────────────────────────────────────────

def main():
    cfg = load_config_defaults()

    parser = argparse.ArgumentParser(
        description="Extract 896x896 frames from benchmark videos."
    )
    parser.add_argument(
        "--cases-dir", default=cfg["cases_dir"],
        help=f"Root directory of per-case folders (default: {cfg['cases_dir']})"
    )
    parser.add_argument(
        "--workers", type=int, default=cfg["workers"],
        help=f"Number of parallel workers (default: {cfg['workers']}, from config.yaml)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=cfg["max_frames"],
        help=f"Max frames per video (default: {cfg['max_frames']})"
    )
    parser.add_argument(
        "--case-ids", nargs="+", default=None,
        help="Process only these case IDs (e.g., e1012 c1003)"
    )
    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    if not cases_dir.is_dir():
        print(f"ERROR: Cases directory not found: {cases_dir}")
        print("Run prepare_data.sh first to create it.")
        sys.exit(1)

    suffix = cfg["subdir_suffix"]

    # Discover cases
    tasks = []
    skipped = 0

    case_dirs = sorted(
        d for d in cases_dir.iterdir()
        if d.is_dir() and _CASE_ID_RE.match(d.name)
    )

    if args.case_ids:
        allowed = set(args.case_ids)
        case_dirs = [d for d in case_dirs if d.name in allowed]

    for case_dir in case_dirs:
        case_id = case_dir.name
        video_path = case_dir / f"{case_id}.mp4"
        out_dir = case_dir / f"{case_id}{suffix}"

        if not video_path.is_file():
            continue

        if out_dir.is_dir() and any(out_dir.iterdir()):
            skipped += 1
            continue

        tasks.append((str(video_path), str(out_dir), case_id))

    print(f"To process: {len(tasks)}, skipped (already exist): {skipped}")
    if not tasks:
        print("Nothing to do.")
        return

    print(f"Workers: {args.workers}, max frames: {args.max_frames}")
    print(f"Output suffix: {suffix}")
    print()

    processed = 0
    errors = []
    total_frames = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_video, vp, od, cid, args.max_frames): cid
            for vp, od, cid in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Extracting frames"):
            case_id = futures[future]
            try:
                n = future.result()
                total_frames += n
                processed += 1
            except Exception as e:
                errors.append(f"{case_id}: {e}")

    print(f"\nProcessed: {processed}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Total frames saved: {total_frames}")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors:
            print(f"  - {e}")
    print("Done.")


if __name__ == "__main__":
    main()
