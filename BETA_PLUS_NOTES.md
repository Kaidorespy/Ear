# Ear Beta+ Development Notes

*Written 2026-02-22 while the insight is fresh*

---

## The Missing Piece: Harsh Vocal Detection via Zero Crossing Rate

### The Problem

Ear's vocal analyzer detects intensity via:
- **RMS energy** (volume)
- **Spectral flatness** (noise-like quality)
- **Sustained high energy** (screaming "holds", singing "breathes")

This catches **belting** and **powerful** moments well. But it misses **primal screams** — the Anaal Nathrakh / black metal / death metal style harsh vocals that are about *texture*, not just volume.

Casey's test track "Shit my name in the cloud" has screaming starting around **8 seconds** that ear completely missed. The narrative talked about "controlled vocals" and "underlying tension" when the reality was face-melting primal screams from the pits of hell.

### The Solution: Zero Crossing Rate

ZCR measures how often the audio signal crosses zero — essentially how "jagged" the waveform is.

- **Low ZCR** = smooth, tonal, sustained (clean singing, instruments)
- **High ZCR** = noisy, harsh, percussive (screams, distortion, cymbals)

Looking at the AVisualizer ZCR output for the test track:
- **8-10s**: Massive ZCR spike — the primal scream
- **25-35s**: Sustained high ZCR — harsh vocal section
- **45-50s, 85-90s**: More harsh texture moments
- **110-120s**: ZCR drops to near zero — the quiet before the storm
- **120-130s**: Moderate ZCR + high energy — belting (ear caught this)

**ZCR spike + moderate-to-high energy = harsh vocals / screaming**
**High energy + low ZCR = belting / powerful clean vocals**

This is the discriminator we were missing.

### Implementation Plan

#### 1. Add ZCR to vocals.py intensity detection

```python
# In detect_vocal_intensity():

# Existing:
flatness = librosa.feature.spectral_flatness(y=y)[0]
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
zcr = librosa.feature.zero_crossing_rate(y)[0]

# New classification logic:
# High ZCR (>0.15) + moderate energy (>0.4) = harsh/screaming
# High ZCR (>0.25) at any energy = definite harsh texture
# High energy + low ZCR = belting/powerful (current detection)
```

#### 2. Add "harsh_segments" to vocal analysis output

```python
{
    "intensity_segments": [...],  # existing
    "harsh_segments": [           # new
        {
            "start": 8.2,
            "end": 12.4,
            "duration": 4.2,
            "type": "primal_scream",  # or "harsh", "guttural", "distorted"
            "zcr_peak": 0.43,
            "energy": 0.52
        }
    ],
    "harsh_vocal_character": "sustained harsh vocals"  # or "harsh moments", "none"
}
```

#### 3. Update intensity_character classification

Current categories:
- sustained screaming
- screaming sections
- intense/powerful throughout
- intense moments
- controlled

New categories:
- **primal/harsh throughout** (high ZCR sustained)
- **harsh vocal sections** (ZCR spikes with energy)
- **mixed harsh and clean** (alternating)
- sustained screaming (energy-based, keep for compatibility)
- screaming sections
- intense/powerful throughout
- intense moments
- controlled

#### 4. Thresholds to tune (start here, adjust based on testing)

```python
ZCR_HARSH_THRESHOLD = 0.15      # above this = harsh texture
ZCR_SCREAM_THRESHOLD = 0.25     # above this = definite scream/harsh
ENERGY_FLOOR_FOR_HARSH = 0.3    # need some energy to count as vocal
MIN_HARSH_DURATION = 0.3        # seconds, filter out transients
```

---

## Visual Augmentation for Synthesis

### The Insight

AVisualizer generates 22 images. That's overkill — information overload for the LLM.

But **2-3 key visualizations** could help ground the narrative synthesis. The LLM gets structured data (JSON) which is precise but abstract. A visual can provide:
- **Confirmation** — "the data says X, and I can see it here"
- **Context** — shapes and patterns that numbers flatten
- **Correction** — "wait, the visual shows something at 0:08 the data didn't flag"

### Which Visualizations Matter

After analyzing all 22, the high-value ones:

1. **Dynamic Range (19)** — color-coded loudness over time. Instantly shows song structure, climaxes, quiet sections. The red/green gradient is intuitive.

2. **Zero Crossing Rate (11)** — texture map. Shows where harsh/noisy elements are vs smooth/tonal. Critical for metal, electronic, anything with distortion.

3. **Combined Dashboard (21)** — overview of everything. Good for orientation but dense.

The rest are either:
- Redundant (spectrogram vs mel spectrogram vs chromagram all show frequency content)
- Too technical (tonnetz, MFCCs, spectral contrast — useful for analysis, not narrative)
- Single-purpose (waveform, beat tracking — covered by structured data)

### Implementation Options

#### Option A: Generate 2-3 PNGs, attach to synthesis prompt

```python
def synthesize_narrative(analysis, images=None, model="claude-sonnet-4-20250514"):
    # ... existing prompt ...

    if images:
        # Use Claude's vision capability
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "source": {"type": "base64", "data": img_b64}}
                for img_b64 in images
            ]
        }]
```

Pros: Rich visual context
Cons: Larger API calls, vision model cost, need matplotlib in core

#### Option B: Generate simplified ASCII/text representations

```
ENERGY MAP (0:00 - 2:44)
|▁▂▃▄▅▆▇█▇▆▅▄▃▄▅▆▇████████▇▆▅▄▃▂▁|
 quiet    build    PEAK    fade

TEXTURE MAP (harsh = #, smooth = -)
|###--##-##--#-##--###-------#####--|
 scream   mixed      clean   outro
```

Pros: No image handling, works with any model, cheap
Cons: Less precise than actual visual

#### Option C: Add key visual metrics to JSON, let synthesis model imagine

Add to analysis output:
```json
{
    "visual_summary": {
        "energy_shape": "quiet_build_peak_fade",
        "texture_zones": [
            {"start": 0, "end": 35, "type": "harsh_dominant"},
            {"start": 35, "end": 110, "type": "mixed"},
            {"start": 110, "end": 120, "type": "quiet"},
            {"start": 120, "end": 140, "type": "intense_clean"},
            {"start": 140, "end": 165, "type": "harsh_return"}
        ],
        "climax_timestamp": 120.5,
        "harshest_moment": 8.2
    }
}
```

Pros: Structured, cheap, no image handling
Cons: Loses the "at a glance" quality of visuals

### Recommendation

**Start with Option C** (structured visual summary in JSON) for beta+.

**Add Option A** (actual images) as a toggle for users who want maximum accuracy and don't mind the cost. Label it "Visual-Augmented Synthesis" or similar.

Skip Option B — it's a half-measure that doesn't give the benefits of either approach.

---

## Timestamp Accuracy in Narratives

### The Problem

The synthesis narrative for "Shit my name in the cloud" referenced events at "2:00" and "2:20" — but the song was cut at 2:00, and those sections don't exist in the final version.

### Root Cause

The analysis data includes timestamps from the full file (164 seconds = 2:44). But Casey cut the song at 2:00 for the final version. The synthesis model saw:
- Belting section around 120-130s (2:00-2:10)
- High intensity continuing to 140s (2:20)

And narrated based on that data, not knowing the cut point.

### Solutions

1. **Re-analyze the final cut** — obvious but requires user discipline

2. **Add "song_end_marker" detection** — look for:
   - Energy drop to near-zero sustained for >5 seconds
   - No vocal activity
   - Could flag "possible song end at X, instrumental outro follows"

3. **Prompt engineering** — tell synthesis model to focus on the main body, note that endings may be cut

4. **User input** — let user specify "song ends at 2:00" and filter analysis data

### Recommendation

Add **automatic outro detection** to structure.py:
```python
def detect_outro(energy_arc, vocal_arc):
    """Detect where the 'song' ends vs instrumental outro."""
    # Look for: sustained low energy + no vocals + >10 seconds
    # Return suggested_end_time or None
```

Then pass this to synthesis: "Note: Main song content appears to end around {suggested_end}. Material after this may be outro/ambient."

---

## UI Improvements (Lower Priority)

### API Key Management
- Settings button → modal with fields for Anthropic, OpenAI, Replicate keys
- Save to keywallet.json
- Show key status (set/not set) in UI

### Model Selection
- Add Ollama option for local synthesis
- Text field for custom Ollama model name
- Dropdown: Cloud (Claude/GPT) | Local (Ollama)

### Error Visibility
- Show synthesis errors in UI, not just silent fail
- "Synthesis failed: {reason}" with retry button
- Show transcription status separately from synthesis

### Progress Detail
- Which analyzer is running
- Estimated time remaining (based on file duration)
- Cancel button for long analyses

---

## Testing Checklist for Beta+

- [ ] Run on "Shit my name in the cloud" — verify harsh vocals detected at 8s
- [ ] Run on clean vocal track — verify no false positives
- [ ] Run on instrumental — verify vocal detection stays false
- [ ] Run on mixed harsh/clean (Deftones style) — verify both detected
- [ ] Test with visual augmentation on/off — compare narrative quality
- [ ] Test Ollama local synthesis
- [ ] Test with missing API keys — verify graceful failure

---

## Files to Modify

1. **analyzers/vocals.py**
   - Add ZCR-based harsh vocal detection
   - Add harsh_segments to output
   - Update intensity_character categories

2. **analyzers/structure.py**
   - Add outro detection
   - Add suggested_end_time to output

3. **core.py**
   - Add visual_summary generation (Option C)
   - Add image generation for Option A (behind flag)
   - Pass visual context to synthesis prompt

4. **ear_gui.pyw**
   - API key settings modal
   - Ollama model selector
   - Better error display
   - Progress detail

---

*"The shape of absence" — Casey, on realizing what was missing*

*ZCR is the scream detector. Energy is the power detector. Together they see everything.*
