"""
Melody Analyzer - pitch contour, melodic shape, phrases
"""

import numpy as np
import librosa


def analyze(y, sr):
    """Extract melodic features from audio."""
    duration = librosa.get_duration(y=y, sr=sr)

    # Pitch tracking using piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=80, fmax=2000)
    times = librosa.times_like(pitches, sr=sr)

    # Extract most prominent pitch at each frame
    pitch_track = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            pitch_track.append({'time': float(times[i]), 'pitch': float(pitch)})

    # Convert to arrays for analysis
    if len(pitch_track) > 10:
        pitch_times = np.array([p['time'] for p in pitch_track])
        pitch_values = np.array([p['pitch'] for p in pitch_track])

        # Melodic range
        pitch_min = np.min(pitch_values)
        pitch_max = np.max(pitch_values)
        range_hz = pitch_max - pitch_min
        range_semitones = 12 * np.log2(pitch_max / pitch_min) if pitch_min > 0 else 0

        # Average pitch (melodic center)
        avg_pitch = np.mean(pitch_values)

        # Pitch variance - how much movement
        pitch_variance = np.std(pitch_values)

        # Melodic contour - overall direction
        first_quarter = np.mean(pitch_values[:len(pitch_values)//4])
        last_quarter = np.mean(pitch_values[-len(pitch_values)//4:])

        if last_quarter > first_quarter * 1.05:
            overall_contour = 'ascending'
        elif last_quarter < first_quarter * 0.95:
            overall_contour = 'descending'
        else:
            overall_contour = 'circular/stable'

        # Movement analysis - jumps vs steps
        pitch_diffs = np.abs(np.diff(pitch_values))
        avg_movement = np.mean(pitch_diffs)

        # Convert to semitones roughly
        semitone_hz = avg_pitch * 0.06  # rough semitone at average pitch
        if avg_movement > semitone_hz * 4:
            movement_type = 'leaping/dramatic'
        elif avg_movement > semitone_hz * 2:
            movement_type = 'mixed steps and leaps'
        elif avg_movement > semitone_hz:
            movement_type = 'stepwise'
        else:
            movement_type = 'static/droning'

    else:
        range_semitones = 0
        avg_pitch = 0
        pitch_variance = 0
        overall_contour = 'unclear'
        movement_type = 'minimal pitch content'

    # Analyze melodic arc over time
    chunk_duration = 10
    num_chunks = int(np.ceil(duration / chunk_duration))
    melody_arc = []

    for i in range(num_chunks):
        start = i * chunk_duration
        end = min((i + 1) * chunk_duration, duration)

        chunk_pitches = [p for p in pitch_track if start <= p['time'] < end]

        if len(chunk_pitches) > 2:
            chunk_values = [p['pitch'] for p in chunk_pitches]
            chunk_avg = np.mean(chunk_values)
            chunk_range = np.max(chunk_values) - np.min(chunk_values)

            # Local contour
            if chunk_values[-1] > chunk_values[0] * 1.03:
                local_contour = 'rising'
            elif chunk_values[-1] < chunk_values[0] * 0.97:
                local_contour = 'falling'
            else:
                local_contour = 'stable'

            # Register
            if chunk_avg > 500:
                register = 'high'
            elif chunk_avg > 250:
                register = 'mid'
            else:
                register = 'low'

            melody_arc.append({
                'start': start,
                'end': end,
                'contour': local_contour,
                'register': register,
                'activity': 'active' if len(chunk_pitches) > 5 else 'sparse'
            })
        else:
            melody_arc.append({
                'start': start,
                'end': end,
                'contour': 'unclear',
                'register': 'unclear',
                'activity': 'minimal'
            })

    # Range characterization
    if range_semitones > 24:
        range_word = 'very wide (2+ octaves)'
    elif range_semitones > 12:
        range_word = 'wide (octave+)'
    elif range_semitones > 7:
        range_word = 'moderate'
    elif range_semitones > 3:
        range_word = 'narrow'
    else:
        range_word = 'very narrow/monotone'

    return {
        'has_clear_melody': len(pitch_track) > 20,
        'range_semitones': float(range_semitones),
        'range_word': range_word,
        'overall_contour': overall_contour,
        'movement_type': movement_type,
        'melody_arc': melody_arc,
        'pitch_activity': float(len(pitch_track) / duration) if duration > 0 else 0,  # pitches per second
        'avg_pitch_hz': float(avg_pitch) if avg_pitch else 0
    }
