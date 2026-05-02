#!/usr/bin/env bash
# ===========================================================
# prepare_data.sh — Unzip Dataverse download and organize
#                    into per-case directories, then extract frames.
#
# Usage:
#   bash prepare_data.sh
#   bash prepare_data.sh --force     # overwrite existing cases/
#   bash prepare_data.sh --zip /path/to/dataverse_files.zip
#
# Run from the distribution/ directory.
# ===========================================================
set -euo pipefail

# ── Resolve project root (where config.yaml lives) ───────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$SCRIPT_DIR"   # distribution/ root

# ── Defaults ─────────────────────────────────────────────
ZIP_PATH=""
FORCE=false

# ── Parse arguments ──────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)  FORCE=true; shift ;;
        --zip)    ZIP_PATH="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--force] [--zip /path/to/dataverse_files.zip]"
            echo ""
            echo "  --force   Overwrite existing cases/ directory"
            echo "  --zip     Path to dataverse_files.zip (default: reads from config.yaml)"
            exit 0 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Read zip path from config.yaml if not specified ──────
if [[ -z "$ZIP_PATH" ]]; then
    CONFIG="$BENCH_DIR/config.yaml"
    if [[ -f "$CONFIG" ]]; then
        # Simple grep — avoids python/yq dependency
        ZIP_PATH=$(grep 'dataverse_zip:' "$CONFIG" | head -1 | sed 's/.*: *//' | sed 's/ *#.*//')
    fi
    # Resolve relative to benchmark/
    if [[ -n "$ZIP_PATH" && "${ZIP_PATH:0:1}" != "/" ]]; then
        ZIP_PATH="$BENCH_DIR/$ZIP_PATH"
    fi
fi

# Fallback default
: "${ZIP_PATH:=$BENCH_DIR/data/dataverse_files.zip}"

DATA_DIR="$(dirname "$ZIP_PATH")"
CASES_DIR="$DATA_DIR/cases"

# ── Validate ─────────────────────────────────────────────
if [[ ! -f "$ZIP_PATH" ]]; then
    echo "ERROR: Dataverse zip not found: $ZIP_PATH"
    echo "Download it from Harvard Dataverse and place it at the path above."
    exit 1
fi

if [[ -d "$CASES_DIR" ]] && [[ "$FORCE" != true ]]; then
    n_dirs=$(find "$CASES_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    if [[ "$n_dirs" -gt 0 ]]; then
        echo "cases/ already exists with $n_dirs case directories."
        echo "Use --force to overwrite, or remove $CASES_DIR manually."
        exit 0
    fi
fi

echo "=== EndoSCOP-V Data Preparation ==="
echo "Zip:    $ZIP_PATH"
echo "Output: $CASES_DIR"
echo ""

# ── Step 1: Unzip outer archive to temp dir ──────────────
TMPDIR=$(mktemp -d "${DATA_DIR}/prepare_XXXXXX")
trap 'rm -rf "$TMPDIR"' EXIT

echo "[1/4] Extracting outer archive..."
unzip -q "$ZIP_PATH" -d "$TMPDIR"

# ── Step 2: Unzip nested archives ────────────────────────
echo "[2/4] Extracting nested archives..."

# QA files
if [[ -f "$TMPDIR/qa.zip" ]]; then
    unzip -q "$TMPDIR/qa.zip" -d "$TMPDIR"
    rm "$TMPDIR/qa.zip"
fi

# Case reports
if [[ -f "$TMPDIR/case_reports.zip" ]]; then
    unzip -q "$TMPDIR/case_reports.zip" -d "$TMPDIR"
    rm "$TMPDIR/case_reports.zip"
fi

# Video sub-archives (small_videos_*.zip, etc.)
for nested_zip in "$TMPDIR"/videos/*.zip; do
    [[ -f "$nested_zip" ]] || continue
    echo "  Extracting $(basename "$nested_zip")..."
    unzip -q -o "$nested_zip" -d "$TMPDIR/videos/"
    rm "$nested_zip"
done

# ── Step 3: Organize into per-case directories ───────────
echo "[3/4] Organizing into per-case directories..."

if [[ "$FORCE" == true ]] && [[ -d "$CASES_DIR" ]]; then
    rm -rf "$CASES_DIR"
fi
mkdir -p "$CASES_DIR"

total_cases=0
cases_with_video=0
cases_with_report=0

# Use QA files as the authoritative case list
for qa_file in "$TMPDIR"/qa/*_qa.json; do
    [[ -f "$qa_file" ]] || continue

    filename=$(basename "$qa_file")
    # Extract case_id: e1012_qa.json -> e1012
    case_id="${filename%_qa.json}"

    # Validate case_id pattern
    if [[ ! "$case_id" =~ ^[ec][0-9]+$ ]]; then
        echo "  WARNING: Skipping unexpected file: $filename"
        continue
    fi

    case_dir="$CASES_DIR/$case_id"
    mkdir -p "$case_dir"

    # Move QA JSON
    mv "$qa_file" "$case_dir/"

    # Move case report (if exists)
    report_file="$TMPDIR/case_reports/${case_id}_case_report.json"
    if [[ -f "$report_file" ]]; then
        mv "$report_file" "$case_dir/"
        cases_with_report=$((cases_with_report + 1))
    fi

    # Move video (if exists) — check multiple locations
    video_file=""
    for vdir in "$TMPDIR/videos" "$TMPDIR"; do
        if [[ -f "$vdir/${case_id}.mp4" ]]; then
            video_file="$vdir/${case_id}.mp4"
            break
        fi
    done
    if [[ -n "$video_file" ]]; then
        mv "$video_file" "$case_dir/"
        cases_with_video=$((cases_with_video + 1))
    fi

    total_cases=$((total_cases + 1))
done

# ── Summary ──────────────────────────────────────────────
echo ""
echo "=== Done ==="
echo "Total cases:       $total_cases"
echo "  with video:      $cases_with_video"
echo "  with report:     $cases_with_report"
echo "  without video:   $((total_cases - cases_with_video))"
echo "Output directory:  $CASES_DIR"

# ── Step 4: Extract frames ───────────────────────────────
if [[ "$cases_with_video" -gt 0 ]]; then
    echo ""
    echo "[4/4] Extracting frames from videos..."
    # Check Python dependencies
    if ! python3 -c "import cv2, numpy, tqdm, yaml" 2>/dev/null; then
        echo ""
        echo "ERROR: Missing Python dependencies for frame extraction."
        echo "Install them with:"
        echo "  pip install -r requirements.txt"
        echo ""
        echo "Data organization is complete. Re-run this script after installing"
        echo "dependencies to extract frames."
        exit 1
    fi
    python3 "$(dirname "$0")/eval/extract_frames.py" --cases-dir "$CASES_DIR"
else
    echo ""
    echo "No videos found — skipping frame extraction."
fi
