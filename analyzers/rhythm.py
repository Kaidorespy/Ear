"""
Rhythm Analyzer - groove, swing, syncopation, rhythmic feel
"""

import numpy as np
import librosa


def analyze(y, sr):
    """Extract rhythmic features beyond simple tempo."""
    duration = librosa.get_duration(y=y, sr=sr)

    # Tempo and beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if hasattr(tempo, '__float__') else tempo[0]
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # Onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times_frames = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    onset_times = librosa.frames_to_time(onset_times_frames, sr=sr)

    # Tempogram - tempo over time
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)

    # Analyze inter-onset intervals for groove characteristics
    if len(onset_times) > 2:
        iois = np.diff(onset_times)  # inter-onset intervals

        # Remove outliers
        iois = iois[(iois > 0.05) & (iois < 2.0)]

        if len(iois) > 0:
            # Coefficient of variation - higher = more variable timing
            ioi_cv = np.std(iois) / np.mean(iois) if np.mean(iois) > 0 else 0

            # Check for swing - ratio between consecutive pairs
            if len(iois) > 3:
                pairs = iois[:-1] / iois[1:]
                swing_ratio = np.median(pairs[(pairs > 0.5) & (pairs < 3)])
            else:
                swing_ratio = 1.0
        else:
            ioi_cv = 0
            swing_ratio = 1.0
    else:
        ioi_cv = 0
        swing_ratio = 1.0

    # Determine groove feel
    if ioi_cv > 0.4:
        groove_feel = 'loose/human'
    elif ioi_cv > 0.2:
        groove_feel = 'natural'
    elif ioi_cv > 0.1:
        groove_feel = 'tight'
    else:
        groove_feel = 'quantized/mechanical'

    # Swing detection
    if swing_ratio > 1.3:
        swing_word = 'swung'
    elif swing_ratio > 1.1:
        swing_word = 'slightly swung'
    elif swing_ratio < 0.9:
        swing_word = 'pushed/anticipated'
    else:
        swing_word = 'straight'

    # Beat strength variation - syncopation indicator
    if len(beats) > 4:
        beat_strengths = onset_env[beats]
        # Compare on-beat vs off-beat strength
        on_beats = beat_strengths[::2] if len(beat_strengths) > 1 else beat_strengths
        off_beats = beat_strengths[1::2] if len(beat_strengths) > 1 else beat_strengths

        syncopation = np.mean(off_beats) / (np.mean(on_beats) + 1e-10)
    else:
        syncopation = 1.0

    if syncopation > 1.2:
        syncopation_word = 'heavily syncopated'
    elif syncopation > 0.9:
        syncopation_word = 'syncopated'
    elif syncopation > 0.7:
        syncopation_word = 'moderate syncopation'
    else:
        syncopation_word = 'on-the-beat'

    # Pulse clarity - how clear/defined the beat is
    # Based on tempogram peak sharpness
    if len(tempogram) > 0:
        tempo_profile = np.mean(tempogram, axis=1)
        peak_idx = np.argmax(tempo_profile)
        if peak_idx > 0 and peak_idx < len(tempo_profile) - 1:
            peak_sharpness = tempo_profile[peak_idx] / (
                np.mean([tempo_profile[peak_idx-1], tempo_profile[peak_idx+1]]) + 1e-10
            )
        else:
            peak_sharpness = 1.0
    else:
        peak_sharpness = 1.0

    if peak_sharpness > 3:
        pulse_word = 'driving/locked'
    elif peak_sharpness > 2:
        pulse_word = 'clear pulse'
    elif peak_sharpness > 1.3:
        pulse_word = 'relaxed pulse'
    else:
        pulse_word = 'floating/ambient'

    # Rhythmic density over time
    chunk_duration = 10
    num_chunks = int(np.ceil(duration / chunk_duration))
    rhythm_arc = []

    for i in range(num_chunks):
        start = i * chunk_duration
        end = min((i + 1) * chunk_duration, duration)

        chunk_onsets = onset_times[(onset_times >= start) & (onset_times < end)]
        density = len(chunk_onsets) / (end - start)

        if density > 10:
            density_word = 'frantic'
        elif density > 6:
            density_word = 'busy'
        elif density > 3:
            density_word = 'active'
        elif density > 1.5:
            density_word = 'moderate'
        else:
            density_word = 'sparse'

        rhythm_arc.append({
            'start': start,
            'end': end,
            'density': float(density),
            'density_word': density_word
        })

    # Tempo feel categories
    if tempo < 70:
        tempo_feel = 'slow/ballad'
    elif tempo < 100:
        tempo_feel = 'mid-tempo/groove'
    elif tempo < 130:
        tempo_feel = 'upbeat'
    elif tempo < 160:
        tempo_feel = 'driving/energetic'
    else:
        tempo_feel = 'fast/intense'

    return {
        'tempo': float(tempo),
        'tempo_feel': tempo_feel,
        'groove_feel': groove_feel,
        'swing': swing_word,
        'swing_ratio': float(swing_ratio),
        'syncopation': syncopation_word,
        'syncopation_ratio': float(syncopation),
        'pulse': pulse_word,
        'timing_variation': float(ioi_cv),
        'rhythm_arc': rhythm_arc,
        'total_hits': len(onset_times)
    }
