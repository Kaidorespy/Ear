"""
Ear - Audio description for Claude
Listens to a song and writes out what it hears.
Not transcription. Description.
"""

import sys
import numpy as np
import librosa
from pathlib import Path


def analyze_audio(filepath):
    """Analyze audio and return musical descriptions."""
    print(f"Loading: {filepath}")

    # Load audio
    y, sr = librosa.load(filepath, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)

    print(f"Duration: {duration:.1f}s")
    print()

    # Get tempo and beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if hasattr(tempo, '__float__') else tempo[0]
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # Energy over time (RMS)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # Onset detection (hits, attacks)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    onset_times = librosa.frames_to_time(onsets, sr=sr)

    # Segment the song into sections based on self-similarity
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    bounds = librosa.segment.agglomerative(chroma, k=8)
    bound_times = librosa.frames_to_time(bounds, sr=sr)

    # Generate description
    description = []
    description.append("=" * 60)
    description.append("EAR ANALYSIS")
    description.append("=" * 60)
    description.append("")
    description.append(f"File: {Path(filepath).name}")
    description.append(f"Duration: {int(duration // 60)}:{int(duration % 60):02d}")
    description.append(f"Tempo: ~{tempo:.0f} BPM")
    description.append("")

    # Describe energy arc
    description.append("ENERGY ARC:")
    description.append("-" * 40)

    # Divide into chunks and describe each
    chunk_duration = 15  # seconds per chunk
    num_chunks = int(np.ceil(duration / chunk_duration))

    for i in range(num_chunks):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, duration)

        # Get indices for this time range
        start_idx = int(start_time / duration * len(rms))
        end_idx = int(end_time / duration * len(rms))

        if start_idx >= end_idx:
            continue

        chunk_rms = rms[start_idx:end_idx]
        chunk_centroid = spectral_centroid[start_idx:end_idx]

        # Characterize energy
        avg_energy = np.mean(chunk_rms)
        max_energy = np.max(rms)
        energy_pct = (avg_energy / max_energy) * 100 if max_energy > 0 else 0

        # Characterize brightness
        avg_brightness = np.mean(chunk_centroid)
        max_brightness = np.max(spectral_centroid)
        brightness_pct = (avg_brightness / max_brightness) * 100 if max_brightness > 0 else 0

        # Energy trend in this section
        if len(chunk_rms) > 1:
            trend = chunk_rms[-1] - chunk_rms[0]
            if trend > 0.01:
                trend_word = "building"
            elif trend < -0.01:
                trend_word = "dropping"
            else:
                trend_word = "steady"
        else:
            trend_word = "steady"

        # Count hits in this section
        hits_in_chunk = np.sum((onset_times >= start_time) & (onset_times < end_time))
        hits_per_sec = hits_in_chunk / (end_time - start_time)

        # Generate description
        timestamp = f"{int(start_time // 60)}:{int(start_time % 60):02d}-{int(end_time // 60)}:{int(end_time % 60):02d}"

        # Energy words
        if energy_pct > 80:
            energy_word = "LOUD - full blast"
        elif energy_pct > 60:
            energy_word = "high energy"
        elif energy_pct > 40:
            energy_word = "medium energy"
        elif energy_pct > 20:
            energy_word = "pulled back"
        else:
            energy_word = "quiet/sparse"

        # Brightness words
        if brightness_pct > 70:
            bright_word = "bright/cutting"
        elif brightness_pct > 40:
            bright_word = "full-bodied"
        else:
            bright_word = "dark/warm"

        # Density words
        if hits_per_sec > 8:
            density_word = "dense, rapid attacks"
        elif hits_per_sec > 4:
            density_word = "driving"
        elif hits_per_sec > 2:
            density_word = "moderate pulse"
        else:
            density_word = "sparse/breathing"

        desc_line = f"[{timestamp}] {energy_word}, {bright_word}, {density_word}, {trend_word}"
        description.append(desc_line)

    description.append("")
    description.append("SECTION BOUNDARIES:")
    description.append("-" * 40)

    for i, t in enumerate(bound_times):
        timestamp = f"{int(t // 60)}:{int(t % 60):02d}"
        description.append(f"  Section {i+1} starts at {timestamp}")

    description.append("")
    description.append("NOTABLE MOMENTS:")
    description.append("-" * 40)

    # Find biggest energy changes
    rms_diff = np.diff(rms)
    big_changes = np.where(np.abs(rms_diff) > np.std(rms_diff) * 2)[0]

    notable = []
    for idx in big_changes[:10]:  # Top 10 changes
        t = times[idx]
        change = rms_diff[idx]
        direction = "energy spike" if change > 0 else "energy drop"
        notable.append((t, direction))

    # Sort by time
    notable.sort(key=lambda x: x[0])

    # Remove duplicates that are too close
    filtered = []
    last_t = -5
    for t, direction in notable:
        if t - last_t > 3:  # At least 3 seconds apart
            timestamp = f"{int(t // 60)}:{int(t % 60):02d}"
            filtered.append(f"  {timestamp}: {direction}")
            last_t = t

    description.extend(filtered[:8])  # Top 8 moments

    description.append("")
    description.append("=" * 60)

    return "\n".join(description)


def main():
    if len(sys.argv) < 2:
        print("Usage: python ear.py <audio_file>")
        print("Example: python ear.py song.mp3")
        sys.exit(1)

    filepath = sys.argv[1]

    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    result = analyze_audio(filepath)
    print(result)

    # Also save to file
    output_path = Path(filepath).with_suffix('.ear.txt')
    with open(output_path, 'w') as f:
        f.write(result)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
