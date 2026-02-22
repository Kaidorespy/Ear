"""
Timbre Analyzer - brightness, texture, instrumentation, spectral character
"""

import numpy as np
import librosa


def analyze(y, sr):
    """Extract timbral features from audio."""
    duration = librosa.get_duration(y=y, sr=sr)

    # Spectral centroid - brightness
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times = librosa.times_like(centroid, sr=sr)

    # Spectral bandwidth - spread of frequencies
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    # Spectral rolloff - frequency below which 85% of energy exists
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]

    # Spectral flatness - tonal vs noisy (closer to 1 = more noise-like)
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    # Zero crossing rate - roughness/percussiveness
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # MFCCs - timbral texture coefficients
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Spectral contrast - valley/peak differences across bands
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Analyze in chunks
    chunk_duration = 10
    num_chunks = int(np.ceil(duration / chunk_duration))
    timbre_arc = []

    max_bandwidth = np.max(bandwidth) if np.max(bandwidth) > 0 else 1

    # Absolute brightness thresholds based on spectral centroid in Hz
    # Calibrated from real-world examples:
    # - Dark songs (Billie Eilish "bad guy"): ~1500-2000 Hz centroid
    # - Bright songs (The Weeknd "Blinding Lights"): ~3000-4000 Hz centroid
    BRIGHTNESS_DARK_THRESH = 2000      # below this = dark/muted
    BRIGHTNESS_WARM_THRESH = 2800      # 2000-2800 = warm
    BRIGHTNESS_CLEAR_THRESH = 3500     # 2800-3500 = clear
    # above 3500 = bright/cutting

    for i in range(num_chunks):
        start = i * chunk_duration
        end = min((i + 1) * chunk_duration, duration)
        mask = (times >= start) & (times < end)

        if not np.any(mask):
            continue

        chunk_centroid = centroid[mask]
        chunk_bandwidth = bandwidth[mask]
        chunk_rolloff = rolloff[mask]
        chunk_flatness = flatness[mask]
        chunk_zcr = zcr[mask]

        # Brightness: absolute centroid in Hz, mapped to 0-100 scale
        # Using 1000-5000 Hz as the practical range for pop music
        centroid_hz = np.mean(chunk_centroid)
        brightness = np.clip((centroid_hz - 1000) / 4000 * 100, 0, 100)

        # Texture density: bandwidth relative to max
        texture_spread = np.mean(chunk_bandwidth) / max_bandwidth * 100

        # Tonal vs noisy
        tonality = 1 - np.mean(chunk_flatness)  # 1 = tonal, 0 = noisy

        # Roughness/percussiveness
        percussiveness = np.mean(chunk_zcr) * 100

        # Warmth: inverse of high-frequency energy
        warmth = 100 - brightness

        # Characterize based on absolute Hz thresholds
        if centroid_hz > BRIGHTNESS_CLEAR_THRESH:
            brightness_word = 'bright/cutting'
        elif centroid_hz > BRIGHTNESS_WARM_THRESH:
            brightness_word = 'clear'
        elif centroid_hz > BRIGHTNESS_DARK_THRESH:
            brightness_word = 'warm'
        else:
            brightness_word = 'dark/muted'

        if tonality > 0.7:
            tonality_word = 'tonal/melodic'
        elif tonality > 0.4:
            tonality_word = 'mixed'
        else:
            tonality_word = 'noisy/textural'

        if percussiveness > 15:
            attack_word = 'aggressive/sharp'
        elif percussiveness > 8:
            attack_word = 'punchy'
        elif percussiveness > 4:
            attack_word = 'smooth'
        else:
            attack_word = 'soft/legato'

        if texture_spread > 70:
            texture_word = 'full/wide'
        elif texture_spread > 40:
            texture_word = 'balanced'
        else:
            texture_word = 'focused/narrow'

        timbre_arc.append({
            'start': start,
            'end': end,
            'brightness': float(brightness),
            'brightness_word': brightness_word,
            'warmth': float(warmth),
            'texture_spread': float(texture_spread),
            'texture_word': texture_word,
            'tonality': float(tonality * 100),
            'tonality_word': tonality_word,
            'percussiveness': float(percussiveness),
            'attack_word': attack_word
        })

    # Overall character - using absolute Hz values
    overall_centroid_hz = np.mean(centroid)
    overall_brightness = np.clip((overall_centroid_hz - 1000) / 4000 * 100, 0, 100)
    overall_tonality = 1 - np.mean(flatness)

    # Estimate instrumentation hints from spectral shape
    # Low MFCCs correlate with different instrument families
    mfcc_means = np.mean(mfccs, axis=1)

    # Very rough instrument hints based on spectral profile
    instrument_hints = []
    if overall_brightness < 40 and overall_tonality > 0.6:
        instrument_hints.append('bass-heavy')
    if overall_brightness > 60 and np.mean(zcr) > 0.1:
        instrument_hints.append('hi-hats/cymbals present')
    if overall_tonality > 0.8:
        instrument_hints.append('melodic instruments')
    if overall_tonality < 0.3:
        instrument_hints.append('noise/texture elements')
    if np.mean(zcr) > 0.15:
        instrument_hints.append('percussive elements')

    # Space/reverb estimation (very rough - based on decay characteristics)
    # High spectral contrast variance = more defined/dry
    # Low contrast variance = more washed/reverberant
    contrast_variance = np.var(contrast)
    if contrast_variance < 5:
        space_word = 'spacious/reverberant'
    elif contrast_variance < 15:
        space_word = 'medium room'
    else:
        space_word = 'tight/dry'

    return {
        'timbre_arc': timbre_arc,
        'overall_brightness': float(overall_brightness),
        'overall_warmth': float(100 - overall_brightness),
        'overall_tonality': float(overall_tonality * 100),
        'instrument_hints': instrument_hints,
        'space': space_word,
        'mfcc_signature': mfcc_means.tolist(),  # timbral fingerprint
        'spectral_contrast_avg': np.mean(contrast, axis=1).tolist()
    }
