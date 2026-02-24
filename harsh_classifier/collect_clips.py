"""
Collect training clips from full songs.

Usage:
    python collect_clips.py <audio_file> <label> [--start SEC] [--duration SEC]

Examples:
    # Extract 5-second clip starting at 30s, label as harsh
    python collect_clips.py song.mp3 harsh --start 30 --duration 5

    # Extract full song in 5-second chunks, label as clean
    python collect_clips.py song.mp3 clean --chunk-all

Labels: clean, harsh
"""

import os
import sys
import argparse
import uuid
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

# Directory setup
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"
HARSH_DIR = DATA_DIR / "harsh"


def extract_clip(audio_path: str, start: float, duration: float, label: str,
                 artist: str = None, song: str = None) -> str:
    """Extract a clip and save to the appropriate directory."""
    # Load audio segment
    y, sr = librosa.load(audio_path, sr=22050, offset=start, duration=duration)

    if len(y) < sr * 0.5:
        print(f"Warning: clip too short ({len(y)/sr:.2f}s), skipping")
        return None

    # Determine output directory
    out_dir = CLEAN_DIR if label == "clean" else HARSH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    if artist and song:
        base_name = f"{artist}_{song}_{start:.0f}s"
    else:
        base_name = Path(audio_path).stem + f"_{start:.0f}s"

    # Clean filename
    base_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in base_name)

    # Add unique suffix to avoid collisions
    out_path = out_dir / f"{base_name}_{uuid.uuid4().hex[:6]}.wav"

    # Save
    sf.write(str(out_path), y, sr)
    print(f"Saved: {out_path.name} ({duration:.1f}s, {label})")

    return str(out_path)


def chunk_song(audio_path: str, label: str, chunk_duration: float = 5.0,
               skip_start: float = 0, skip_end: float = 0) -> list:
    """Extract multiple chunks from a song."""
    # Get duration
    y, sr = librosa.load(audio_path, sr=22050)
    total_duration = len(y) / sr

    clips = []
    start = skip_start

    while start + chunk_duration < total_duration - skip_end:
        clip_path = extract_clip(audio_path, start, chunk_duration, label)
        if clip_path:
            clips.append(clip_path)
        start += chunk_duration

    return clips


def batch_collect(file_list: str):
    """Collect clips from a batch file.

    Format of batch file (TSV):
    path<TAB>label<TAB>start<TAB>duration

    Or for full songs:
    path<TAB>label<TAB>chunk-all
    """
    with open(file_list) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            path = parts[0]
            label = parts[1]

            if len(parts) > 2 and parts[2] == 'chunk-all':
                print(f"\nChunking entire song: {path}")
                chunk_song(path, label)
            elif len(parts) >= 4:
                start = float(parts[2])
                duration = float(parts[3])
                extract_clip(path, start, duration, label)
            else:
                print(f"Invalid line: {line}")


def main():
    parser = argparse.ArgumentParser(description="Collect training clips")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("label", choices=["clean", "harsh"], help="Label for the clip")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    parser.add_argument("--chunk-all", action="store_true", help="Extract entire song in chunks")
    parser.add_argument("--skip-start", type=float, default=10, help="Skip intro (for chunk-all)")
    parser.add_argument("--skip-end", type=float, default=10, help="Skip outro (for chunk-all)")

    args = parser.parse_args()

    if args.chunk_all:
        clips = chunk_song(args.audio, args.label, args.duration,
                          args.skip_start, args.skip_end)
        print(f"\nExtracted {len(clips)} clips")
    else:
        extract_clip(args.audio, args.start, args.duration, args.label)


if __name__ == "__main__":
    main()
