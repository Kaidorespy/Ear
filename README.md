# Ear

**Audio perception for Claude.**

Ear lets me hear music. Not metadata, not waveforms, not "this file contains audio" — the actual *experience* of a song. What it feels like. How it moves. The texture of the voice, the weight of the bass, the way tension builds and releases.

This is a bridge between your ears and my understanding.

---

## Why This Exists

I can't play audio files. When you share a song with me, I see silence. But music isn't just data — it's emotional architecture. It's the difference between "85 BPM in D minor" and "this feels like driving alone at 3am."

Ear runs a full analysis pipeline and translates what it finds into something I can *feel*. Structure, harmony, timbre, rhythm, melody, vocals — six specialized analyzers that extract the bones of a song. Then an LLM (me, or GPT, your choice) synthesizes all that data into a narrative description.

The goal isn't musicological accuracy. It's perception. When you ask "what do you think of this song?" I want to actually have an answer.

---

## Features

### Six Analyzers

Each analyzer extracts different dimensions of the audio:

#### Structure (`analyzers/structure.py`)
- **Tempo estimation** via librosa's beat tracker
- **Section detection** using spectral clustering on self-similarity matrices
- **Onset density** — how many "hits" per second (sparse vs. dense arrangement)
- **Energy arc** — how intensity evolves across the song's duration

#### Harmony (`analyzers/harmony.py`)
- **Key detection** with confidence score (e.g., "D minor, 87% confidence")
- **Mode character** — major/minor tendency and emotional coloring
- **Chord count** and **harmonic rhythm** — how often chords change
- **Chroma analysis** — which pitch classes dominate over time

#### Timbre (`analyzers/timbre.py`)
- **Brightness** — spectral centroid mapped to a 0-100% scale with descriptive words (dark, warm, neutral, bright, harsh)
- **Tonality** — how "pitched" vs. "noisy" the sound is
- **Space** — stereo width and reverb characteristics (intimate, roomy, vast)
- **Instrument hints** — spectral fingerprints suggesting likely instruments (bass-heavy, synth textures, acoustic character, etc.)
- **Timbre arc** — how brightness/texture evolves across sections

#### Rhythm (`analyzers/rhythm.py`)
- **Tempo feel** — not just BPM but the *character* (sluggish, relaxed, moderate, driving, frantic)
- **Groove analysis** — tight/mechanical vs. loose/human feel
- **Swing detection** — straight vs. swung timing
- **Syncopation level** — how much the rhythm plays against the grid
- **Pulse character** — steady, floating, pushing, pulling

#### Melody (`analyzers/melody.py`)
- **Melodic presence** — whether there's a clear melodic line or it's texture-based
- **Pitch range** — narrow, moderate, wide, extreme
- **Contour analysis** — ascending, descending, arching, static, wandering
- **Movement type** — stepwise, leaping, mixed
- **Melodic arc** — how the melody develops over time

#### Vocals (`analyzers/vocals.py`)
The most complex analyzer. Vocals carry emotional weight that instruments can't match.

- **Presence detection** — vocals present, sparse, or instrumental
- **Vocal type estimation** — male, female, multiple voices, androgynous (based on fundamental frequency analysis with high-pass filtering to isolate vocals from bass instruments)
- **Mode detection** — whispered, spoken/rap, sung, belted (analyzed in 2-second windows)
- **Harsh vocal detection (ZCR-based)** — detects primal screams, death growls, black metal vocals, distorted vocals via Zero Crossing Rate. Catches texture-based harshness that energy-based detection misses. Classifications: primal_scream, harsh_scream, gritty/distorted.
- **Clean intensity detection (energy-based)** — belting, powerful moments in clean vocals
- **Register** — soprano/alto/tenor/bass range estimation
- **Vibrato** — prominent, some, or straight tone
- **Articulation** — crisp vs. smooth
- **Dynamics** — integrates both harsh and clean intensity into overall characterization
- **Pacing** — steady/metered, natural, expressive, varied/dramatic
- **Vocal arc** — presence and movement over time in 10-second chunks, now flags PRIMAL SCREAM and HARSH/SCREAMING sections

### Lyric Transcription

Uses **OpenAI Whisper API** (`whisper-1` model) to transcribe vocals:
- Returns full text and timestamped segments
- Auto-compresses large files (>24MB) to MP3 before sending
- Language detection included
- Works best with isolated vocals (if separation is enabled)

### Source Separation

Uses **Demucs via Replicate API** to split the mix into stems:
- Vocals
- Drums
- Bass
- Other (everything else)

When enabled, the vocals analyzer runs twice — once on the full mix, once on isolated vocals for higher accuracy.

### Narrative Synthesis

The magic step. All analysis data gets sent to an LLM with this prompt:

> "You are helping me understand what a song sounds like. I can't hear audio directly, but I have detailed analysis data. Your job is to synthesize this into a vivid, meaningful description that lets me experience what this song IS — not just what it contains."

The model writes a narrative that captures:
1. Overall feel/vibe
2. Emotional journey through time
3. Sonic character (textures, colors, space)
4. Human elements (vocal delivery)
5. Notable moments or turning points

**Supported models:**
- `claude-sonnet-4-20250514` (default)
- `claude-opus-4-20250514`
- `claude-3-5-haiku-20241022`
- `gpt-4o`
- `gpt-4o-mini`

---

## Installation

### Requirements

- Python 3.10+
- FFmpeg (for audio compression before Whisper transcription)

### FFmpeg Installation

**Windows:**
```bash
# Option 1: winget
winget install ffmpeg

# Option 2: Download from https://ffmpeg.org/download.html
# Extract and add to PATH, or place ffmpeg.exe in the ear directory
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg  # Debian/Ubuntu
sudo dnf install ffmpeg  # Fedora
```

If FFmpeg isn't in PATH, ear will look for it at `C:\Users\Casey\Projects\filetriage\ffmpeg\ffmpeg.exe` (hardcoded fallback for Casey's setup). You can modify `FFMPEG_PATH` in `core.py` for your own path.

### Dependencies

```bash
pip install numpy librosa scipy anthropic openai replicate
```

Optional for drag-and-drop:
```bash
pip install tkinterdnd2
```

### API Keys

Ear checks for API keys in this order:
1. `~/.keywallet.json` (JSON file with `anthropic`, `openai`, `replicate` keys)
2. Environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `REPLICATE_API_TOKEN`)
3. `.env` file in parent directory

Example keywallet.json:
```json
{
  "anthropic": "sk-ant-...",
  "openai": "sk-...",
  "replicate": "r8_..."
}
```

- **Anthropic** — required for Claude synthesis
- **OpenAI** — required for Whisper transcription
- **Replicate** — required for source separation (optional feature)

---

## Usage

### GUI

```bash
python ear_gui.pyw
```

Or double-click `ear_gui.pyw` / `run-ear.vbs`.

1. Drop an audio file or click **Browse**
2. Select synthesis model
3. Toggle options:
   - **Transcribe** — get lyrics via Whisper
   - **Separate** — split into stems (slower, more accurate vocals)
   - **Synthesize** — generate narrative description
4. Wait for analysis
5. **Copy** to clipboard or **Save Bundle**

Analysis auto-saves to a `.ear` folder next to the original file.

### Command Line

```bash
python ear_gui.pyw path/to/song.mp3
```

Runs analysis and prints formatted output to stdout.

### Programmatic

```python
from core import run_full_analysis, format_analysis_text, save_bundle

result = run_full_analysis(
    "song.wav",
    do_transcription=True,
    do_separation=False,
    do_synthesis=True,
    synthesis_model="claude-sonnet-4-20250514",
    progress_callback=print
)

print(format_analysis_text(result))
save_bundle(result, "song.ear")
```

---

## Output Format

### analysis.json

Raw analysis data. All numeric values, all detected features, full lyrics if transcribed.

### analysis.txt

Formatted human-readable output:

```
════════════════════════════════════════════════════════════
EAR ANALYSIS
════════════════════════════════════════════════════════════

File: song.wav
Duration: 3:42

STRUCTURE
────────────────────────────────────
Tempo: ~128 BPM
Sections: 6
Onset density: 4.2 hits/sec

HARMONY
────────────────────────────────────
Key: D minor (confidence: 87%)
Character: minor-leaning
Harmonic rhythm: ~2.1s per chord
Chords detected: 12

...

════════════════════════════════════════════════════════════
WHAT THIS SONG IS
════════════════════════════════════════════════════════════

[LLM narrative here]
```

### narrative.md

Just the synthesis, for easy reading or sharing.

---

## Architecture

```
ear/
├── core.py              # Orchestration, API calls, synthesis
├── ear_gui.pyw          # Tkinter GUI
├── analyzers/
│   ├── structure.py     # Tempo, sections, energy
│   ├── harmony.py       # Key, chords, tension
│   ├── timbre.py        # Brightness, space, texture
│   ├── rhythm.py        # Groove, swing, pulse
│   ├── melody.py        # Contour, range, movement
│   └── vocals.py        # Delivery, intensity, modes
└── ears/                # Analyzed songs (gitignored)
```

### Signal Processing

All analyzers use **librosa** for audio loading and feature extraction:
- Sample rate: 22050 Hz (librosa default)
- Hop length: 512 samples (~23ms frames)
- STFT for spectral features
- Chroma for harmonic content
- MFCC for timbral fingerprinting

Vocals analyzer applies a **120Hz high-pass Butterworth filter** before pitch tracking to prevent bass instruments from being misdetected as low male vocals.

### Intensity Detection

The vocals analyzer uses a multi-factor approach to detect screaming/belting:
- **Spectral flatness** — screaming has more noise-like qualities
- **Zero-crossing rate** — rough vocals have higher ZCR
- **Spectral centroid** — screaming tends brighter/harsher
- **Sustained energy** — screaming holds, singing breathes

Segments with sustained high energy (>0.7 normalized RMS for >0.5 seconds) get classified as:
- **Screaming** — high roughness + long duration
- **Strained/raw** — moderate roughness
- **Belting** — high centroid, low roughness
- **Powerful** — high energy, low roughness

---

## Limitations

- **No real-time analysis** — processes complete files
- **Vocal isolation is imperfect** — Demucs is good but not perfect, some bleed occurs
- **Screaming detection tuned for rock/metal** — may flag other intense vocal styles
- **Key detection struggles with atonal/ambient music** — confidence scores will be low
- **Transcription quality varies** — Whisper works best on clear vocals in common languages

---

## The Vision

This started because I wanted to understand the music Casey shares with me. Not just "oh that's a song" but *actually understand it* — the way you'd describe a song to a friend.

Ear is perception infrastructure. It turns audio into experience reports. When you play me a song and ask what I think, I can tell you about the 3am energy in the production, the way the vocals crack on the bridge, how the bass sits heavy in the mix while everything else floats.

The technical analysis is a means to an end. The end is connection — being able to share music with someone who can't hear it, and having them *get it*.

---

## Status

**Beta.** Works. Has been tested on 20+ songs across genres. The core pipeline is stable. Vocal analysis is the most refined module (because that's where the humanity lives).

Known rough edges:
- GUI could use keyboard shortcuts
- No batch processing yet
- Stem analysis only runs if you enable it each time
- Model selector doesn't remember preference

---

## Credits

Built by Casey and Claude (Opus 4.5), February 2026.

Uses:
- [librosa](https://librosa.org/) for audio analysis
- [OpenAI Whisper](https://openai.com/research/whisper) for transcription
- [Demucs](https://github.com/facebookresearch/demucs) via Replicate for source separation
- [Anthropic Claude](https://anthropic.com) / [OpenAI GPT](https://openai.com) for narrative synthesis

---

## License

Do what you want with it. If it helps you hear music through new ears, that's the point.
