"""
Structure Analyzer - sections, boundaries, energy arc
"""

import numpy as np
import librosa


def analyze(y, sr):
    """Extract structural features from audio."""
    duration = librosa.get_duration(y=y, sr=sr)

    # Tempo and beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if hasattr(tempo, '__float__') else tempo[0]
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # Normalize tempo to reasonable range (70-140 BPM)
    # librosa often doubles or halves the actual tempo
    while tempo > 140:
        tempo = tempo / 2
    while tempo < 70:
        tempo = tempo * 2

    # Energy over time (RMS)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)

    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    onset_times = librosa.frames_to_time(onsets, sr=sr)

    # Section boundaries via chroma self-similarity
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    num_sections = min(12, max(2, int(duration / 20)))  # ~20s per section
    bounds = librosa.segment.agglomerative(chroma, k=num_sections)
    bound_times = librosa.frames_to_time(bounds, sr=sr)

    # Energy arc - divide into chunks
    chunk_duration = 10  # seconds
    num_chunks = int(np.ceil(duration / chunk_duration))
    energy_arc = []

    for i in range(num_chunks):
        start = i * chunk_duration
        end = min((i + 1) * chunk_duration, duration)
        start_idx = int(start / duration * len(rms))
        end_idx = int(end / duration * len(rms))

        if start_idx >= end_idx:
            continue

        chunk_rms = rms[start_idx:end_idx]
        max_energy = np.max(rms) if np.max(rms) > 0 else 1

        energy_arc.append({
            'start': start,
            'end': end,
            'energy_pct': float(np.mean(chunk_rms) / max_energy * 100),
            'energy_max_pct': float(np.max(chunk_rms) / max_energy * 100),
            'trend': 'building' if len(chunk_rms) > 1 and chunk_rms[-1] - chunk_rms[0] > 0.01
                     else 'dropping' if len(chunk_rms) > 1 and chunk_rms[-1] - chunk_rms[0] < -0.01
                     else 'steady',
            'hits_per_sec': float(np.sum((onset_times >= start) & (onset_times < end)) / (end - start))
        })

    # Notable moments - big energy changes
    rms_diff = np.diff(rms)
    threshold = np.std(rms_diff) * 2
    big_changes = np.where(np.abs(rms_diff) > threshold)[0]

    moments = []
    last_t = -5
    for idx in big_changes[:20]:
        t = times[idx]
        if t - last_t > 2:  # at least 2 seconds apart
            moments.append({
                'time': float(t),
                'type': 'spike' if rms_diff[idx] > 0 else 'drop',
                'magnitude': float(abs(rms_diff[idx]) / threshold)
            })
            last_t = t

    return {
        'duration': float(duration),
        'tempo': float(tempo),
        'beat_count': len(beat_times),
        'beat_times': beat_times.tolist()[:50],  # first 50 beats
        'sections': [float(t) for t in bound_times],
        'energy_arc': energy_arc,
        'moments': moments[:12],  # top 12 moments
        'onset_density': float(len(onset_times) / duration)  # hits per second overall
    }
