"""
Harmony Analyzer - key, chords, tension/resolution, harmonic content
"""

import numpy as np
import librosa


# Note names for key detection
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Major and minor profiles for key detection (Krumhansl-Schmuckler)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Common chord templates (simplified)
CHORD_TEMPLATES = {
    'maj': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'dim': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'aug': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'sus4': [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus2': [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
}


def estimate_key(chroma):
    """Estimate key using Krumhansl-Schmuckler algorithm."""
    chroma_avg = np.mean(chroma, axis=1)

    best_corr = -1
    best_key = 'C'
    best_mode = 'major'

    for i in range(12):
        # Rotate profiles to each key
        major_rot = np.roll(MAJOR_PROFILE, i)
        minor_rot = np.roll(MINOR_PROFILE, i)

        maj_corr = np.corrcoef(chroma_avg, major_rot)[0, 1]
        min_corr = np.corrcoef(chroma_avg, minor_rot)[0, 1]

        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = NOTE_NAMES[i]
            best_mode = 'major'

        if min_corr > best_corr:
            best_corr = min_corr
            best_key = NOTE_NAMES[i]
            best_mode = 'minor'

    return best_key, best_mode, float(best_corr)


def detect_chord(chroma_frame):
    """Detect most likely chord from a chroma frame."""
    best_match = None
    best_score = -1
    best_root = 0

    for root in range(12):
        for chord_type, template in CHORD_TEMPLATES.items():
            rotated = np.roll(template, root)
            score = np.dot(chroma_frame, rotated)
            if score > best_score:
                best_score = score
                best_match = chord_type
                best_root = root

    root_name = NOTE_NAMES[best_root]
    if best_match == 'maj':
        return root_name
    elif best_match == 'min':
        return f"{root_name}m"
    else:
        return f"{root_name}{best_match}"


def compute_tension(chroma):
    """Compute harmonic tension over time based on dissonance."""
    # Dissonance approximation: how far from consonant intervals
    tension = []
    for frame in chroma.T:
        # Normalize
        if np.sum(frame) > 0:
            frame = frame / np.sum(frame)
        # Tension = entropy-like measure (more spread = more tension)
        entropy = -np.sum(frame * np.log(frame + 1e-10))
        tension.append(entropy)
    return np.array(tension)


def analyze(y, sr):
    """Extract harmonic features from audio."""
    duration = librosa.get_duration(y=y, sr=sr)

    # Chroma features (pitch class representation)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_times = librosa.times_like(chroma, sr=sr)

    # Key estimation
    key, mode, confidence = estimate_key(chroma)

    # Chord progression - sample every ~2 seconds
    hop_frames = max(1, int(2 * sr / 512))  # ~2 second hops
    chords = []
    for i in range(0, chroma.shape[1], hop_frames):
        frame = np.mean(chroma[:, i:i+hop_frames], axis=1)
        chord = detect_chord(frame)
        time = float(chroma_times[min(i, len(chroma_times)-1)])
        if not chords or chords[-1]['chord'] != chord:
            chords.append({'time': time, 'chord': chord})

    # Tension curve
    tension = compute_tension(chroma)
    tension_times = chroma_times

    # Summarize tension over chunks
    chunk_duration = 10
    num_chunks = int(np.ceil(duration / chunk_duration))
    tension_arc = []

    for i in range(num_chunks):
        start = i * chunk_duration
        end = min((i + 1) * chunk_duration, duration)
        mask = (tension_times >= start) & (tension_times < end)

        if np.any(mask):
            chunk_tension = tension[mask]
            tension_arc.append({
                'start': start,
                'end': end,
                'tension_avg': float(np.mean(chunk_tension)),
                'tension_max': float(np.max(chunk_tension)),
                'tension_trend': 'rising' if len(chunk_tension) > 1 and chunk_tension[-1] > chunk_tension[0]
                                 else 'falling' if len(chunk_tension) > 1 and chunk_tension[-1] < chunk_tension[0]
                                 else 'stable'
            })

    # Harmonic rhythm - how often chords change
    if len(chords) > 1:
        chord_durations = []
        for i in range(len(chords) - 1):
            chord_durations.append(chords[i+1]['time'] - chords[i]['time'])
        avg_chord_duration = np.mean(chord_durations)
    else:
        avg_chord_duration = duration

    # Modal character
    chroma_avg = np.mean(chroma, axis=1)
    major_minor_ratio = (chroma_avg[4] / (chroma_avg[3] + 1e-10))  # major 3rd vs minor 3rd

    return {
        'key': key,
        'mode': mode,
        'key_confidence': confidence,
        'chords': chords[:50],  # first 50 chord changes
        'chord_count': len(chords),
        'harmonic_rhythm': float(avg_chord_duration),  # avg seconds per chord
        'tension_arc': tension_arc,
        'major_minor_character': 'major' if major_minor_ratio > 1.2 else 'minor' if major_minor_ratio < 0.8 else 'ambiguous',
        'chroma_summary': chroma_avg.tolist()  # average pitch class energy
    }
