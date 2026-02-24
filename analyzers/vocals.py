"""
Vocals Analyzer - delivery, pacing, emotion markers, intensity, vocal character
Designed to work on isolated vocal tracks (from source separation)
or full mix (less accurate but still useful)
"""

import numpy as np
import librosa
from scipy import signal
from pathlib import Path
import joblib

# ML Harsh Vocal Classifier
def _get_model_path():
    """Get model path - handles both dev and PyInstaller bundled exe."""
    import sys
    if getattr(sys, 'frozen', False):
        # Running as bundled exe
        base = Path(sys._MEIPASS)
    else:
        # Running in dev
        base = Path(__file__).parent.parent
    return base / "harsh_classifier" / "models" / "harsh_classifier.joblib"

_HARSH_MODEL_PATH = _get_model_path()
_harsh_classifier = None

def _load_harsh_classifier():
    """Lazy-load the harsh vocal classifier."""
    global _harsh_classifier
    if _harsh_classifier is None and _HARSH_MODEL_PATH.exists():
        _harsh_classifier = joblib.load(_HARSH_MODEL_PATH)
    return _harsh_classifier


def _extract_ml_features(y: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract features for ML classifier (must match training features)."""
    features = []

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    features.append(np.mean(flatness))
    features.append(np.std(flatness))

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # RMS
    rms = librosa.feature.rms(y=y)[0]
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    return np.array(features, dtype=np.float32)


def detect_harsh_vocals_ml(y, sr, chunk_duration: float = 3.0):
    """ML-based harsh vocal detection using trained classifier.

    Returns harsh_segments list and harsh_character string.
    Much more accurate than ZCR-based detection - can distinguish
    powerful clean belting (Whitney, Adele) from actual harsh vocals.
    """
    clf = _load_harsh_classifier()
    if clf is None:
        # Fallback to empty if model not found
        return [], 'unknown (model not loaded)'

    duration = len(y) / sr
    chunk_samples = int(chunk_duration * sr)
    hop_samples = int(chunk_samples * 0.5)  # 50% overlap

    results = []
    pos = 0

    while pos + chunk_samples <= len(y):
        chunk = y[pos:pos + chunk_samples]
        time_start = pos / sr

        # Skip quiet sections
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms < 0.01:
            pos += hop_samples
            continue

        try:
            features = _extract_ml_features(chunk, sr)
            proba = clf.predict_proba([features])[0]
            pred = clf.predict([features])[0]

            results.append({
                'time_start': time_start,
                'time_end': time_start + chunk_duration,
                'prediction': 'harsh' if pred == 1 else 'clean',
                'harsh_prob': float(proba[1]),
                'clean_prob': float(proba[0]),
            })
        except Exception:
            pass

        pos += hop_samples

    if not results:
        return [], 'no vocal content detected'

    # Build harsh segments from consecutive harsh predictions
    harsh_segments = []
    in_harsh = False
    seg_start = 0
    seg_probs = []

    for r in results:
        if r['prediction'] == 'harsh' and not in_harsh:
            in_harsh = True
            seg_start = r['time_start']
            seg_probs = [r['harsh_prob']]
        elif r['prediction'] == 'harsh' and in_harsh:
            seg_probs.append(r['harsh_prob'])
        elif r['prediction'] == 'clean' and in_harsh:
            # End segment
            harsh_segments.append({
                'start': float(seg_start),
                'end': float(r['time_start']),
                'duration': float(r['time_start'] - seg_start),
                'avg_confidence': float(np.mean(seg_probs)),
                'peak_confidence': float(max(seg_probs)),
                'type': 'harsh_ml'
            })
            in_harsh = False
            seg_probs = []

    # Close final segment if still in harsh
    if in_harsh and results:
        harsh_segments.append({
            'start': float(seg_start),
            'end': float(results[-1]['time_end']),
            'duration': float(results[-1]['time_end'] - seg_start),
            'avg_confidence': float(np.mean(seg_probs)),
            'peak_confidence': float(max(seg_probs)),
            'type': 'harsh_ml'
        })

    # Calculate overall stats
    harsh_count = sum(1 for r in results if r['prediction'] == 'harsh')
    total_count = len(results)
    harsh_ratio = harsh_count / total_count if total_count > 0 else 0
    avg_harsh_prob = np.mean([r['harsh_prob'] for r in results])

    # Determine character (thresholds tuned to not flag powerful clean belting)
    # Adele belting = ~25% harsh ratio, should be "clean" or "occasional grit" at most
    if harsh_ratio > 0.7:
        harsh_character = 'extreme harsh vocals throughout'
    elif harsh_ratio > 0.5:
        harsh_character = 'predominantly harsh'
    elif harsh_ratio > 0.4:
        harsh_character = 'harsh vocal sections'
    elif harsh_ratio > 0.30:
        harsh_character = 'harsh moments'
    elif avg_harsh_prob > 0.4:
        harsh_character = 'occasional grit'
    else:
        harsh_character = 'clean'

    return harsh_segments, harsh_character


def highpass_filter(y, sr, cutoff=120):
    """Apply high-pass filter to remove bass frequencies before vocal analysis.

    This prevents bass guitar/synth bass from polluting vocal pitch detection.
    Cutoff at 120 Hz removes most bass while preserving even low male vocals.
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    # Use a gentler 2nd order filter to avoid phase artifacts
    b, a = signal.butter(2, normalized_cutoff, btype='high')
    return signal.filtfilt(b, a, y)


def estimate_genre_context(tempo: float = None, tonality: float = None,
                           energy_density: float = None) -> dict:
    """Estimate genre context to help weight harsh vocal detection.

    Returns a dict with:
    - metal_likelihood: 0-1 score of how likely this is metal/hardcore
    - density_threshold_adjustment: how much to lower density threshold

    High tempo (>130) + low tonality (distortion) + high density = likely metal
    """
    metal_score = 0.0

    # Fast tempo suggests metal/hardcore/punk
    if tempo:
        if tempo > 160:
            metal_score += 0.4
        elif tempo > 130:
            metal_score += 0.25
        elif tempo > 110:
            metal_score += 0.1

    # Low tonality suggests distorted guitars (noise-like)
    if tonality is not None:
        if tonality < 30:  # very noisy/distorted
            metal_score += 0.4
        elif tonality < 50:
            metal_score += 0.2

    # High energy density suggests wall-of-sound
    if energy_density:
        if energy_density > 4.0:
            metal_score += 0.2
        elif energy_density > 3.0:
            metal_score += 0.1

    # Cap at 1.0
    metal_score = min(1.0, metal_score)

    # Adjust density threshold based on metal likelihood
    # Metal context: lower threshold (more permissive)
    # Pop context: keep threshold high (conservative)
    threshold_adjustment = metal_score * 3.0  # up to 3.0 lower

    return {
        'metal_likelihood': metal_score,
        'density_threshold_adjustment': threshold_adjustment
    }


def detect_harsh_vocals(y, sr, rms, times, genre_context: dict = None):
    """Detect harsh/screamed vocals with genre context awareness.

    Uses conservative thresholds by default, but relaxes them if genre
    context suggests metal/hardcore (high tempo, distorted guitars, etc.)

    This prevents powerful clean vocalists (Whitney, Bonnie Tyler) from
    being flagged as "harsh" while still catching death metal growls.

    Returns harsh_segments list and harsh_character string.
    """
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # RMS for filtering out silence
    rms_normalized = rms / np.max(rms) if np.max(rms) > 0 else rms

    # Base thresholds (conservative for pop/jazz)
    ZCR_HARSH_THRESHOLD = 0.20
    ZCR_SCREAM_THRESHOLD = 0.30
    ZCR_PRIMAL_THRESHOLD = 0.40
    ENERGY_FLOOR = 0.20
    BASE_DENSITY_THRESHOLD = 7.0  # conservative default

    # Adjust density threshold based on genre context
    density_threshold = BASE_DENSITY_THRESHOLD
    if genre_context:
        adjustment = genre_context.get('density_threshold_adjustment', 0)
        density_threshold = max(3.0, BASE_DENSITY_THRESHOLD - adjustment)
        # Metal context can lower threshold to 4.0 (7.0 - 3.0)

    min_len = min(len(zcr), len(rms_normalized), len(times))
    duration = times[-1] if len(times) > 0 else 0

    # Count harsh frames (high ZCR + high energy)
    harsh_frames = []
    for i in range(min_len):
        if zcr[i] > ZCR_HARSH_THRESHOLD and rms_normalized[i] > ENERGY_FLOOR:
            harsh_frames.append({
                'time': times[i],
                'zcr': zcr[i],
                'energy': rms_normalized[i]
            })

    # Calculate density in 10-second windows
    window_size = 10.0
    num_windows = int(np.ceil(duration / window_size))
    dense_windows = []

    for w in range(num_windows):
        start = w * window_size
        end = min((w + 1) * window_size, duration)
        window_harsh = [f for f in harsh_frames if start <= f['time'] < end]

        window_duration = end - start
        if window_duration > 0:
            density = len(window_harsh) / window_duration
            # Density threshold varies by genre context
            # Pop/jazz: 7.0 (conservative), Metal: down to 4.0 (permissive)
            if density > density_threshold:
                avg_zcr = np.mean([f['zcr'] for f in window_harsh]) if window_harsh else 0
                dense_windows.append({
                    'start': start,
                    'end': end,
                    'density': density,
                    'avg_zcr': avg_zcr,
                    'frame_count': len(window_harsh)
                })

    # Build harsh_segments from dense windows (merge adjacent)
    harsh_segments = []
    if dense_windows:
        current_seg = {
            'start': dense_windows[0]['start'],
            'end': dense_windows[0]['end'],
            'zcr_values': [dense_windows[0]['avg_zcr']],
            'densities': [dense_windows[0]['density']]
        }

        for w in dense_windows[1:]:
            if w['start'] <= current_seg['end']:  # adjacent or overlapping
                current_seg['end'] = w['end']
                current_seg['zcr_values'].append(w['avg_zcr'])
                current_seg['densities'].append(w['density'])
            else:
                # Finalize current segment
                avg_zcr = np.mean(current_seg['zcr_values'])
                peak_zcr = np.max(current_seg['zcr_values'])

                if peak_zcr > ZCR_PRIMAL_THRESHOLD:
                    harsh_type = 'primal_scream'
                elif peak_zcr > ZCR_SCREAM_THRESHOLD:
                    harsh_type = 'harsh_scream'
                else:
                    harsh_type = 'gritty/distorted'

                harsh_segments.append({
                    'start': float(current_seg['start']),
                    'end': float(current_seg['end']),
                    'duration': float(current_seg['end'] - current_seg['start']),
                    'type': harsh_type,
                    'zcr_peak': float(peak_zcr),
                    'zcr_avg': float(avg_zcr),
                    'density': float(np.mean(current_seg['densities']))
                })

                current_seg = {
                    'start': w['start'],
                    'end': w['end'],
                    'zcr_values': [w['avg_zcr']],
                    'densities': [w['density']]
                }

        # Don't forget the last segment
        avg_zcr = np.mean(current_seg['zcr_values'])
        peak_zcr = np.max(current_seg['zcr_values'])

        if peak_zcr > ZCR_PRIMAL_THRESHOLD:
            harsh_type = 'primal_scream'
        elif peak_zcr > ZCR_SCREAM_THRESHOLD:
            harsh_type = 'harsh_scream'
        else:
            harsh_type = 'gritty/distorted'

        harsh_segments.append({
            'start': float(current_seg['start']),
            'end': float(current_seg['end']),
            'duration': float(current_seg['end'] - current_seg['start']),
            'type': harsh_type,
            'zcr_peak': float(peak_zcr),
            'zcr_avg': float(avg_zcr),
            'density': float(np.mean(current_seg['densities']))
        })

    # Overall harsh character based on total harsh time and density
    total_harsh_time = sum(s['duration'] for s in harsh_segments)
    harsh_ratio = total_harsh_time / duration if duration > 0 else 0

    primal_time = sum(s['duration'] for s in harsh_segments if s['type'] == 'primal_scream')
    harsh_scream_time = sum(s['duration'] for s in harsh_segments if s['type'] == 'harsh_scream')

    if harsh_ratio > 0.5 or primal_time > 30:
        harsh_character = 'extreme harsh vocals throughout'
    elif harsh_ratio > 0.3 or primal_time > 15:
        harsh_character = 'primal scream sections'
    elif harsh_ratio > 0.15 or harsh_scream_time > 20:
        harsh_character = 'harsh vocal sections'
    elif total_harsh_time > 10:
        harsh_character = 'harsh moments'
    elif total_harsh_time > 3:
        harsh_character = 'occasional grit'
    else:
        harsh_character = 'clean'

    return harsh_segments, harsh_character


def detect_vocal_intensity(y, sr, rms, times):
    """Detect belting/powerful moments via energy (clean vocal intensity).

    This complements detect_harsh_vocals - this catches LOUD clean vocals,
    that catches TEXTURED harsh vocals.

    Returns intensity segments with classifications.
    """
    # Spectral flatness - for distinguishing belt types
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    # Spectral centroid - brightness
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # Zero crossing rate - to avoid double-counting harsh segments
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # RMS energy
    rms_normalized = rms / np.max(rms) if np.max(rms) > 0 else rms

    # Detect sustained high energy (for clean/belt vocals)
    high_energy_threshold = 0.7
    high_energy_mask = rms_normalized > high_energy_threshold

    # Find segments of sustained high energy
    intensity_segments = []
    in_intense_segment = False
    segment_start = 0

    min_duration = 0.5  # minimum seconds

    for i, (is_high, t) in enumerate(zip(high_energy_mask, times)):
        if is_high and not in_intense_segment:
            in_intense_segment = True
            segment_start = t
        elif not is_high and in_intense_segment:
            in_intense_segment = False
            duration = t - segment_start
            if duration >= min_duration:
                start_idx = np.searchsorted(times, segment_start)
                end_idx = i

                seg_flatness = np.mean(flatness[start_idx:end_idx]) if end_idx > start_idx else 0
                seg_zcr = np.mean(zcr[start_idx:end_idx]) if end_idx > start_idx else 0
                seg_centroid = np.mean(centroid[start_idx:end_idx]) if end_idx > start_idx else 0

                # Skip if this is clearly a harsh segment (handled by detect_harsh_vocals)
                if seg_zcr > 0.15:
                    # High ZCR = harsh, let the other detector handle it
                    continue

                # Classify clean intensity type
                if seg_centroid > np.mean(centroid) * 1.3:
                    intensity_type = 'belting'
                elif seg_flatness > 0.1:
                    intensity_type = 'strained'
                else:
                    intensity_type = 'powerful'

                intensity_segments.append({
                    'start': float(segment_start),
                    'end': float(t),
                    'duration': float(duration),
                    'type': intensity_type,
                    'energy': float(np.mean(rms_normalized[start_idx:end_idx]))
                })

    # Overall intensity character (for clean vocals only now)
    total_intense_time = sum(s['duration'] for s in intensity_segments)
    belt_time = sum(s['duration'] for s in intensity_segments if s['type'] == 'belting')

    if belt_time > 15:
        intensity_character = 'sustained belting'
    elif belt_time > 5:
        intensity_character = 'belting sections'
    elif total_intense_time > 20:
        intensity_character = 'powerful throughout'
    elif total_intense_time > 5:
        intensity_character = 'powerful moments'
    else:
        intensity_character = 'controlled'

    return intensity_segments, intensity_character


def estimate_vocal_type(pitches, magnitudes, sr):
    """Estimate likely vocal type (male/female/mixed/ambiguous).

    Based on fundamental frequency ranges:
    - Bass: 80-180 Hz
    - Baritone: 100-260 Hz
    - Tenor: 130-350 Hz
    - Alto: 175-440 Hz
    - Soprano: 250-700 Hz
    """
    # Extract pitch values with stronger magnitude filtering
    pitch_values = []
    mag_threshold = np.percentile(magnitudes[magnitudes > 0], 50) if np.any(magnitudes > 0) else 0

    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        mag = magnitudes[index, i]
        # Only include strong, clearly vocal pitches
        if pitch > 80 and pitch < 600 and mag > mag_threshold:
            pitch_values.append(pitch)

    if len(pitch_values) < 10:
        return 'unclear', 0, []

    pitch_array = np.array(pitch_values)
    avg_pitch = np.mean(pitch_array)
    median_pitch = np.median(pitch_array)
    pitch_std = np.std(pitch_array)

    # Use median for classification (more robust to outliers)
    # Female voices typically center 220-400 Hz
    # Male voices typically center 100-180 Hz

    # Check for multiple distinct pitch centers (multiple voices)
    hist, bin_edges = np.histogram(pitch_array, bins=20)
    peaks = np.where(hist > np.mean(hist) + np.std(hist))[0]
    likely_multiple_voices = len(peaks) >= 2 and (bin_edges[peaks[-1]] - bin_edges[peaks[0]]) > 120

    # Count pitches in different ranges
    low_pitches = np.sum(pitch_array < 180)  # clearly male range
    high_pitches = np.sum(pitch_array > 240)  # clearly female range
    total = len(pitch_array)

    if likely_multiple_voices:
        vocal_type = 'multiple voices'
    elif median_pitch > 280:
        vocal_type = 'female'
    elif median_pitch > 220:
        # In the overlap zone - use proportion of high vs low
        if high_pitches > low_pitches * 1.5:
            vocal_type = 'likely female'
        elif low_pitches > high_pitches * 1.5:
            vocal_type = 'likely male'
        else:
            vocal_type = 'mixed/androgynous'
    elif median_pitch > 160:
        vocal_type = 'likely male'
    elif median_pitch > 0:
        vocal_type = 'male (low)'
    else:
        vocal_type = 'unclear'

    return vocal_type, float(median_pitch), pitch_array.tolist()[:100]


def detect_vocal_modes(y, sr, rms, times, onset_times):
    """Detect different vocal modes: whispered, spoken, sung, belted.

    Based on:
    - Whisper: low energy, high noise, minimal pitch
    - Spoken: rhythmic, limited pitch range, natural dynamics
    - Sung: sustained pitches, wider range, melodic contour
    - Belted: high energy, sustained, clear pitch
    """
    # Get spectral characteristics
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # High-pass filter before pitch tracking to isolate vocals from bass
    y_filtered = highpass_filter(y, sr, cutoff=120)

    # Pitch tracking on filtered audio
    pitches, magnitudes = librosa.piptrack(y=y_filtered, sr=sr, fmin=80, fmax=600)

    rms_normalized = rms / np.max(rms) if np.max(rms) > 0 else rms

    # Analyze in small windows
    window_duration = 2  # seconds
    num_windows = int(np.ceil(librosa.get_duration(y=y, sr=sr) / window_duration))

    mode_arc = []

    for i in range(num_windows):
        start = i * window_duration
        end = (i + 1) * window_duration

        mask = (times >= start) & (times < end)
        if not np.any(mask):
            continue

        window_rms = np.mean(rms_normalized[mask])
        window_flatness = np.mean(flatness[mask])
        window_centroid = np.mean(centroid[mask])

        # Count pitched frames in this window
        start_frame = int(start * sr / 512)
        end_frame = int(end * sr / 512)
        end_frame = min(end_frame, pitches.shape[1])

        pitched_frames = 0
        pitch_values = []
        for j in range(start_frame, end_frame):
            if j < pitches.shape[1]:
                idx = magnitudes[:, j].argmax()
                if pitches[idx, j] > 0:
                    pitched_frames += 1
                    pitch_values.append(pitches[idx, j])

        pitch_ratio = pitched_frames / max(1, end_frame - start_frame)
        pitch_range = np.ptp(pitch_values) if len(pitch_values) > 2 else 0

        # Classify mode
        if window_rms < 0.2 and window_flatness > 0.1:
            mode = 'whispered'
        elif pitch_ratio < 0.3 and window_rms > 0.3:
            mode = 'spoken/rap'
        elif window_rms > 0.7 and pitch_ratio > 0.5:
            mode = 'belted'
        elif pitch_ratio > 0.4 and pitch_range > 50:
            mode = 'sung'
        elif pitch_ratio > 0.3:
            mode = 'sung (soft)'
        else:
            mode = 'unclear'

        mode_arc.append({
            'start': start,
            'end': end,
            'mode': mode,
            'energy': float(window_rms * 100)
        })

    # Summarize dominant modes
    mode_counts = {}
    for m in mode_arc:
        mode = m['mode']
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    if mode_counts:
        dominant_mode = max(mode_counts, key=mode_counts.get)
        secondary_modes = [m for m, c in mode_counts.items() if m != dominant_mode and c > 1]
    else:
        dominant_mode = 'unclear'
        secondary_modes = []

    return mode_arc, dominant_mode, secondary_modes


def analyze(y, sr, is_isolated_vocals=False, genre_context: dict = None):
    """Extract vocal characteristics from audio.

    Args:
        y: audio signal
        sr: sample rate
        is_isolated_vocals: True if this is an isolated vocal track
        genre_context: Optional dict from estimate_genre_context() for weighted detection
    """
    duration = librosa.get_duration(y=y, sr=sr)

    # Energy/loudness
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)

    # High-pass filter to isolate vocals from bass before pitch tracking
    # This prevents bass guitar/synth from being detected as "low male voice"
    y_filtered = highpass_filter(y, sr, cutoff=120)

    # Pitch tracking for voice - using filtered audio
    pitches, magnitudes = librosa.piptrack(y=y_filtered, sr=sr, fmin=80, fmax=800)

    # Extract vocal pitch track
    vocal_pitches = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        mag = magnitudes[index, i]
        if pitch > 0 and mag > np.mean(magnitudes) * 0.5:
            vocal_pitches.append({
                'time': float(times[min(i, len(times)-1)]),
                'pitch': float(pitch),
                'strength': float(mag)
            })

    # Voiced vs unvoiced ratio
    if len(vocal_pitches) > 0:
        voiced_frames = len(vocal_pitches)
        total_frames = pitches.shape[1]
        voiced_ratio = voiced_frames / total_frames
    else:
        voiced_ratio = 0

    # Spectral flux - articulation
    spec = np.abs(librosa.stft(y))
    flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
    avg_flux = np.mean(flux)

    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env)
    onset_times = librosa.frames_to_time(onsets, sr=sr)

    # Phrasing analysis
    if len(onset_times) > 2:
        onset_gaps = np.diff(onset_times)
        phrase_breaks = onset_gaps[onset_gaps > 1.0]
        num_phrases = len(phrase_breaks) + 1
        rapid_sections = np.sum(onset_gaps < 0.15)
        pacing_cv = np.std(onset_gaps) / np.mean(onset_gaps) if np.mean(onset_gaps) > 0 else 0
    else:
        num_phrases = 1
        rapid_sections = 0
        pacing_cv = 0

    # Vibrato detection
    if len(vocal_pitches) > 10:
        pitch_values = np.array([p['pitch'] for p in vocal_pitches])
        pitch_diff = np.diff(pitch_values)
        sign_changes = np.sum(np.diff(np.sign(pitch_diff)) != 0)
        vibrato_score = sign_changes / len(pitch_diff) if len(pitch_diff) > 0 else 0
    else:
        vibrato_score = 0

    # ============ Harsh Vocal Detection (ML-based, falls back to ZCR) ============
    clf = _load_harsh_classifier()
    if clf is not None:
        # Use ML classifier (much more accurate - solves the Whitney problem)
        harsh_segments, harsh_character = detect_harsh_vocals_ml(y, sr)
    else:
        # Fallback to ZCR-based if model not available
        harsh_segments, harsh_character = detect_harsh_vocals(y, sr, rms, times, genre_context)

    # ============ Clean Intensity Detection (energy-based) ============
    intensity_segments, intensity_character = detect_vocal_intensity(y, sr, rms, times)

    # ============ Vocal Type Estimation ============
    vocal_type, avg_pitch, _ = estimate_vocal_type(pitches, magnitudes, sr)

    # ============ NEW: Vocal Mode Detection ============
    mode_arc, dominant_mode, secondary_modes = detect_vocal_modes(y, sr, rms, times, onset_times)

    # Dynamics characterization (incorporating both harsh and clean intensity)
    rms_normalized = rms / np.max(rms) if np.max(rms) > 0 else rms
    dynamics_range = np.max(rms) - np.min(rms)

    # Harsh vocals take priority in characterization
    if harsh_character in ['extreme harsh vocals throughout', 'primal scream sections']:
        intensity_word = harsh_character
    elif harsh_character == 'harsh vocal sections':
        intensity_word = 'harsh vocal sections'
    elif intensity_character in ['sustained belting', 'belting sections']:
        intensity_word = intensity_character
    elif harsh_character == 'harsh moments':
        intensity_word = 'harsh moments with clean sections'
    elif dynamics_range > 0.15:
        intensity_word = 'highly dynamic/emotional'
    elif dynamics_range > 0.08:
        intensity_word = 'expressive dynamics'
    elif dynamics_range > 0.03:
        intensity_word = 'moderate dynamics'
    else:
        intensity_word = 'consistent/controlled'

    # Pacing words
    if pacing_cv > 0.8:
        pacing_word = 'varied/dramatic'
    elif pacing_cv > 0.5:
        pacing_word = 'expressive'
    elif pacing_cv > 0.3:
        pacing_word = 'natural'
    else:
        pacing_word = 'steady/metered'

    if rapid_sections > len(onset_times) * 0.3:
        speed_word = 'rapid-fire sections'
    elif rapid_sections > len(onset_times) * 0.1:
        speed_word = 'some fast passages'
    else:
        speed_word = 'measured pace'

    if vibrato_score > 0.3:
        vibrato_word = 'prominent vibrato'
    elif vibrato_score > 0.15:
        vibrato_word = 'some vibrato'
    else:
        vibrato_word = 'straight tone'

    # Register analysis
    if avg_pitch > 400:
        register_word = 'high register (soprano/alto range)'
    elif avg_pitch > 250:
        register_word = 'mid register'
    elif avg_pitch > 150:
        register_word = 'low-mid register (tenor/alto)'
    elif avg_pitch > 0:
        register_word = 'low register (bass/baritone)'
    else:
        register_word = 'unclear'

    # Vocal presence arc (updated with mode info)
    chunk_duration = 10
    num_chunks = int(np.ceil(duration / chunk_duration))
    vocal_arc = []

    for i in range(num_chunks):
        start = i * chunk_duration
        end = min((i + 1) * chunk_duration, duration)

        chunk_pitches = [p for p in vocal_pitches if start <= p['time'] < end]

        mask = (times >= start) & (times < end)
        if np.any(mask):
            chunk_energy = np.mean(rms[mask])
            max_energy = np.max(rms) if np.max(rms) > 0 else 1
            energy_pct = chunk_energy / max_energy * 100
        else:
            energy_pct = 0

        # Check for harsh vocals in this chunk (ZCR-based)
        chunk_harsh = [s for s in harsh_segments
                      if s['start'] < end and s['end'] > start]
        chunk_primal = [s for s in chunk_harsh if s['type'] == 'primal_scream']

        # Check for belting in this chunk (energy-based)
        chunk_belts = [s for s in intensity_segments
                      if s['start'] < end and s['end'] > start]

        # Get dominant mode for this chunk
        chunk_modes = [m for m in mode_arc if m['start'] < end and m['end'] > start]
        chunk_mode = chunk_modes[0]['mode'] if chunk_modes else 'unclear'

        if len(chunk_pitches) > 3:
            presence = 'vocals present'
            if chunk_primal:
                movement = 'PRIMAL SCREAM'
            elif chunk_harsh:
                movement = 'HARSH/SCREAMING'
            elif chunk_belts:
                movement = 'BELTING'
            else:
                chunk_pitch_values = [p['pitch'] for p in chunk_pitches]
                if np.std(chunk_pitch_values) > 50:
                    movement = 'active melody'
                else:
                    movement = 'sustained/held'
        elif len(chunk_pitches) > 0:
            presence = 'sparse vocals'
            movement = 'occasional'
        else:
            presence = 'instrumental/no vocals'
            movement = 'none'

        vocal_arc.append({
            'start': start,
            'end': end,
            'presence': presence,
            'movement': movement,
            'intensity': float(energy_pct),
            'mode': chunk_mode
        })

    # Breath detection
    rms_diff = np.diff(rms_normalized)
    breath_candidates = np.where(
        (rms_diff[:-1] < -0.1) & (rms_diff[1:] > 0.05)
    )[0]
    breath_count = len(breath_candidates)

    return {
        'has_vocals': voiced_ratio > 0.1,
        'voiced_ratio': float(voiced_ratio),
        'vocal_type': vocal_type,
        'dominant_mode': dominant_mode,
        'secondary_modes': secondary_modes,
        'pacing': pacing_word,
        'speed': speed_word,
        'vibrato': vibrato_word,
        'dynamics': intensity_word,
        'intensity_character': intensity_character,
        'intensity_segments': intensity_segments[:10],  # top 10 clean intense moments
        'harsh_character': harsh_character,
        'harsh_segments': harsh_segments[:10],  # top 10 harsh vocal moments
        'register': register_word,
        'avg_pitch_hz': float(avg_pitch) if avg_pitch else 0,
        'phrase_count': num_phrases,
        'breath_count': breath_count,
        'articulation': 'crisp' if avg_flux > np.median(flux) * 1.5 else 'smooth',
        'vocal_arc': vocal_arc,
        'is_isolated': is_isolated_vocals
    }
