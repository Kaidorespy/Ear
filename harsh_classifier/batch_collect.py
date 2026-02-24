"""
Batch collect clips from known discographies.

Run this to auto-collect training data from your downloads.
"""

import os
import sys
from pathlib import Path
import random

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
from collect_clips import extract_clip, chunk_song

# Your download directory
DOWNLOADS = Path("C:/Users/Casey/Downloads")


def find_tracks(directory: Path, extensions=('.mp3', '.flac', '.wav')) -> list:
    """Find all audio tracks in a directory recursively."""
    tracks = []
    for ext in extensions:
        tracks.extend(directory.rglob(f"*{ext}"))
    return tracks


def collect_from_artist(artist_dir: Path, label: str, max_clips: int = 20,
                       clip_duration: float = 5.0):
    """Collect clips from an artist's discography."""
    tracks = find_tracks(artist_dir)
    if not tracks:
        print(f"No tracks found in {artist_dir}")
        return 0

    # Shuffle and limit
    random.shuffle(tracks)
    tracks = tracks[:max_clips]

    count = 0
    for track in tracks:
        try:
            # Extract clip from middle of song (skip intro/outro)
            import librosa
            y, sr = librosa.load(str(track), sr=22050)
            duration = len(y) / sr

            if duration < 30:
                start = duration / 4  # Short songs: start at 25%
            else:
                start = 30  # Start 30 seconds in

            extract_clip(str(track), start, clip_duration, label)
            count += 1
        except Exception as e:
            print(f"Error with {track.name}: {e}")

    return count


def main():
    print("=== Batch Clip Collection ===\n")

    # HARSH sources (never clean sing)
    harsh_sources = [
        (DOWNLOADS / "Cannibal Corpse - Discography [1990 - 2023]", "harsh"),
        (DOWNLOADS / "Hatebreed - The Divinity Of Purpose (2013) [mp3@320]", "harsh"),
        (DOWNLOADS / "Behemoth - The Shit Ov God (2025)", "harsh"),
    ]

    # CLEAN sources (never scream)
    clean_sources = [
        (DOWNLOADS / "Billie Eilish - Discography [FLAC] [PMEDIA] ⭐️", "clean"),
        (DOWNLOADS / "Britney Spears - The Essential Britney Spears (2013 Pop) [Flac 16-44]", "clean"),
        (DOWNLOADS / "VA - NOW That's What I Call Music Forever 90s (2020) Mp3 320kbps [PMEDIA] ⭐️", "clean"),
        (DOWNLOADS / "Beyonce Discography 2003 - 2012 [MP3 320 - Stepherd]", "clean"),
        (DOWNLOADS / "Adele - Discography [FLAC] [PMEDIA] ⭐️", "clean"),
    ]

    # MANUAL SORT (skip here - Casey will label):
    # - Linkin Park - Discography (2000-2017) [FLAC] vtwin88cube
    # - Limp Bizkit - Discography [FLAC Songs] [PMEDIA] ⭐️

    total_harsh = 0
    total_clean = 0

    # Collect harsh
    print("--- Collecting HARSH clips ---")
    for source_dir, label in harsh_sources:
        if source_dir.exists():
            print(f"\nFrom: {source_dir.name}")
            count = collect_from_artist(source_dir, label, max_clips=25)
            total_harsh += count
        else:
            print(f"Not found: {source_dir}")

    # Collect clean
    print("\n--- Collecting CLEAN clips ---")
    for source_dir, label in clean_sources:
        if source_dir.exists():
            print(f"\nFrom: {source_dir.name}")
            count = collect_from_artist(source_dir, label, max_clips=20)
            total_clean += count
        else:
            print(f"Not found: {source_dir}")

    print(f"\n=== Summary ===")
    print(f"Harsh clips: {total_harsh}")
    print(f"Clean clips: {total_clean}")
    print(f"Total: {total_harsh + total_clean}")
    print("\nRun 'python train.py' to train the model!")


if __name__ == "__main__":
    main()
