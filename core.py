"""
Ear Core - Orchestrates all analyzers and synthesis
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Optional, Callable
import subprocess

import numpy as np
import librosa

# FFmpeg path (local install)
FFMPEG_PATH = r"C:\Users\Casey\Projects\filetriage\ffmpeg\ffmpeg.exe"

from analyzers import structure, harmony, timbre, rhythm, melody, vocals


# Load API keys from environment, keywallet, or .env
def load_api_keys():
    """Load API keys from environment, keywallet, or .env file."""
    keys = {
        'anthropic': os.environ.get('ANTHROPIC_API_KEY'),
        'openai': os.environ.get('OPENAI_API_KEY'),
        'replicate': os.environ.get('REPLICATE_API_TOKEN'),
    }

    # Try keywallet first (Casey's setup)
    wallet_path = Path.home() / ".keywallet.json"
    if wallet_path.exists():
        try:
            wallet = json.loads(wallet_path.read_text())
            if not keys['anthropic']:
                keys['anthropic'] = wallet.get('anthropic')
            if not keys['openai']:
                keys['openai'] = wallet.get('openai')
            if not keys['replicate']:
                keys['replicate'] = wallet.get('replicate')
        except:
            pass

    # Fallback to .env if still missing
    env_path = Path(__file__).parent.parent.parent / 'voiceclaudews' / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    if key == 'ANTHROPIC_API_KEY' and not keys['anthropic']:
                        keys['anthropic'] = value
                    elif key == 'OPENAI_API_KEY' and not keys['openai']:
                        keys['openai'] = value
                    elif key == 'REPLICATE_API_TOKEN' and not keys['replicate']:
                        keys['replicate'] = value

    return keys


def separate_sources(filepath: str, progress_callback: Optional[Callable] = None) -> Optional[dict]:
    """Separate audio into stems using demucs via Replicate API.

    Returns dict with paths to separated stems, or None if separation fails.
    """
    keys = load_api_keys()
    if not keys['replicate']:
        if progress_callback:
            progress_callback("No Replicate API key - skipping source separation")
        return None

    try:
        import replicate
    except ImportError:
        if progress_callback:
            progress_callback("Replicate not installed - skipping source separation")
        return None

    # Set the API token in environment (replicate library reads from env)
    os.environ['REPLICATE_API_TOKEN'] = keys['replicate']

    if progress_callback:
        progress_callback("Separating sources (this may take a minute)...")

    try:
        # Use demucs model on Replicate
        output = replicate.run(
            "cjwbw/demucs:25a173108cff36ef9f80f854c162d01df9e6528be175794b81f7a3a78b8a0d11",
            input={
                "audio": open(filepath, "rb"),
                "stem": "none"  # get all stems
            }
        )

        # Download stems to temp directory
        stem_dir = Path(tempfile.mkdtemp(prefix="ear_stems_"))
        stems = {}

        for stem_name, url in output.items():
            if progress_callback:
                progress_callback(f"Downloading {stem_name}...")

            import urllib.request
            stem_path = stem_dir / f"{stem_name}.wav"
            urllib.request.urlretrieve(url, stem_path)
            stems[stem_name] = str(stem_path)

        return stems

    except Exception as e:
        if progress_callback:
            progress_callback(f"Source separation failed: {e}")
        return None


def transcribe_lyrics(filepath: str, progress_callback: Optional[Callable] = None) -> Optional[dict]:
    """Transcribe lyrics using OpenAI Whisper API.

    Returns dict with full text and timestamped segments.
    """
    keys = load_api_keys()
    if not keys['openai']:
        if progress_callback:
            progress_callback("No OpenAI API key - skipping transcription")
        return None

    if progress_callback:
        progress_callback("Transcribing lyrics...")

    # Check file size - Whisper API limit is 25MB
    file_size = Path(filepath).stat().st_size
    transcribe_path = filepath

    if file_size > 24 * 1024 * 1024:  # Over 24MB, compress to be safe
        if progress_callback:
            progress_callback("Compressing audio for transcription...")
        try:
            temp_mp3 = tempfile.mktemp(suffix='.mp3')
            subprocess.run([
                FFMPEG_PATH, '-i', filepath, '-b:a', '128k', '-y', temp_mp3
            ], capture_output=True, check=True)
            transcribe_path = temp_mp3
        except Exception as e:
            if progress_callback:
                progress_callback(f"Compression failed, trying original: {e}")

    try:
        import openai
        client = openai.OpenAI(api_key=keys['openai'])

        with open(transcribe_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        return {
            'text': transcript.text,
            'segments': [
                {
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text
                }
                for seg in transcript.segments
            ] if hasattr(transcript, 'segments') else [],
            'language': transcript.language if hasattr(transcript, 'language') else 'unknown'
        }

    except Exception as e:
        if progress_callback:
            progress_callback(f"Transcription failed: {e}")
        return None


def synthesize_narrative(analysis: dict, model: str = "claude-sonnet-4-20250514",
                         progress_callback: Optional[Callable] = None) -> Optional[str]:
    """Use an LLM to synthesize all analysis into a narrative description.

    Args:
        analysis: Full analysis dict from run_full_analysis
        model: Model to use (claude-sonnet-4-20250514, claude-opus-4-20250514, gpt-4o, etc.)
    """
    keys = load_api_keys()

    if progress_callback:
        progress_callback(f"Synthesizing narrative with {model}...")

    prompt = f"""You are helping me understand what a song sounds like. I can't hear audio directly, but I have detailed analysis data. Your job is to synthesize this into a vivid, meaningful description that lets me experience what this song IS - not just what it contains.

Write a narrative description that captures:
1. The overall feel/vibe - what does this song FEEL like?
2. The emotional journey - how does it move through time?
3. The sonic character - the textures, colors, space
4. The human elements - if there are vocals, what's the delivery like?
5. Any notable moments or turning points

Be specific but not clinical. This should read like someone describing a song to a friend who wants to know if they'd like it.

Here's the full analysis data:

```json
{json.dumps(analysis, indent=2)}
```

Write your description now. Be vivid. Make me feel it."""

    try:
        if model.startswith("claude"):
            import anthropic
            client = anthropic.Anthropic(api_key=keys['anthropic'])

            response = client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif model.startswith("gpt"):
            import openai
            client = openai.OpenAI(api_key=keys['openai'])

            response = client.chat.completions.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        else:
            return f"Unknown model: {model}"

    except Exception as e:
        if progress_callback:
            progress_callback(f"Synthesis failed: {e}")
        return None


def run_full_analysis(filepath: str,
                      do_separation: bool = False,
                      do_transcription: bool = True,
                      do_synthesis: bool = True,
                      synthesis_model: str = "claude-sonnet-4-20250514",
                      progress_callback: Optional[Callable] = None) -> dict:
    """Run complete audio analysis pipeline.

    Args:
        filepath: Path to audio file
        do_separation: Whether to run source separation
        do_transcription: Whether to transcribe lyrics
        do_synthesis: Whether to synthesize narrative
        synthesis_model: Model to use for synthesis
        progress_callback: Optional callback for progress updates
    """
    result = {
        'file': Path(filepath).name,
        'path': str(filepath)
    }

    # Load audio
    if progress_callback:
        progress_callback("Loading audio...")

    try:
        y, sr = librosa.load(filepath, sr=22050)
        result['duration'] = float(librosa.get_duration(y=y, sr=sr))
    except Exception as e:
        return {'error': f"Failed to load audio: {e}"}

    # Run all analyzers
    if progress_callback:
        progress_callback("Analyzing structure...")
    result['structure'] = structure.analyze(y, sr)

    if progress_callback:
        progress_callback("Analyzing harmony...")
    result['harmony'] = harmony.analyze(y, sr)

    if progress_callback:
        progress_callback("Analyzing timbre...")
    result['timbre'] = timbre.analyze(y, sr)

    if progress_callback:
        progress_callback("Analyzing rhythm...")
    result['rhythm'] = rhythm.analyze(y, sr)

    if progress_callback:
        progress_callback("Analyzing melody...")
    result['melody'] = melody.analyze(y, sr)

    if progress_callback:
        progress_callback("Analyzing vocals...")
    result['vocals'] = vocals.analyze(y, sr)

    # Optional: Source separation
    if do_separation:
        stems = separate_sources(filepath, progress_callback)
        if stems:
            result['stems'] = stems

            # Re-analyze isolated vocals if we got them
            if 'vocals' in stems:
                if progress_callback:
                    progress_callback("Analyzing isolated vocals...")
                y_voc, _ = librosa.load(stems['vocals'], sr=22050)
                result['vocals_isolated'] = vocals.analyze(y_voc, sr, is_isolated_vocals=True)

    # Optional: Transcription
    if do_transcription:
        # Use isolated vocals if available, otherwise full mix
        transcribe_path = result.get('stems', {}).get('vocals', filepath)
        lyrics = transcribe_lyrics(transcribe_path, progress_callback)
        if lyrics:
            result['lyrics'] = lyrics

    # Optional: Synthesis
    if do_synthesis:
        narrative = synthesize_narrative(result, synthesis_model, progress_callback)
        if narrative:
            result['narrative'] = narrative

    if progress_callback:
        progress_callback("Done!")

    return result


def format_analysis_text(analysis: dict) -> str:
    """Format analysis dict as readable text."""
    lines = []
    lines.append("═" * 60)
    lines.append("EAR ANALYSIS")
    lines.append("═" * 60)
    lines.append("")

    # Basic info
    lines.append(f"File: {analysis.get('file', 'unknown')}")
    lines.append(f"Duration: {int(analysis.get('duration', 0) // 60)}:{int(analysis.get('duration', 0) % 60):02d}")
    lines.append("")

    # Structure
    if 'structure' in analysis:
        s = analysis['structure']
        lines.append("STRUCTURE")
        lines.append("─" * 40)
        lines.append(f"Tempo: ~{s['tempo']:.0f} BPM")
        lines.append(f"Sections: {len(s['sections'])}")
        lines.append(f"Onset density: {s['onset_density']:.1f} hits/sec")
        lines.append("")

    # Harmony
    if 'harmony' in analysis:
        h = analysis['harmony']
        lines.append("HARMONY")
        lines.append("─" * 40)
        lines.append(f"Key: {h['key']} {h['mode']} (confidence: {h['key_confidence']:.0%})")
        lines.append(f"Character: {h['major_minor_character']}")
        lines.append(f"Harmonic rhythm: ~{h['harmonic_rhythm']:.1f}s per chord")
        lines.append(f"Chords detected: {h['chord_count']}")
        lines.append("")

    # Rhythm
    if 'rhythm' in analysis:
        r = analysis['rhythm']
        lines.append("RHYTHM")
        lines.append("─" * 40)
        lines.append(f"Feel: {r['tempo_feel']}")
        lines.append(f"Groove: {r['groove_feel']}")
        lines.append(f"Swing: {r['swing']}")
        lines.append(f"Syncopation: {r['syncopation']}")
        lines.append(f"Pulse: {r['pulse']}")
        lines.append("")

    # Timbre
    if 'timbre' in analysis:
        t = analysis['timbre']
        lines.append("TIMBRE")
        lines.append("─" * 40)
        lines.append(f"Brightness: {t['overall_brightness']:.0f}% ({t['timbre_arc'][0]['brightness_word'] if t['timbre_arc'] else 'n/a'})")
        lines.append(f"Tonality: {t['overall_tonality']:.0f}%")
        lines.append(f"Space: {t['space']}")
        if t['instrument_hints']:
            lines.append(f"Hints: {', '.join(t['instrument_hints'])}")
        lines.append("")

    # Melody
    if 'melody' in analysis:
        m = analysis['melody']
        lines.append("MELODY")
        lines.append("─" * 40)
        lines.append(f"Clear melody: {'Yes' if m['has_clear_melody'] else 'No/unclear'}")
        lines.append(f"Range: {m['range_word']}")
        lines.append(f"Contour: {m['overall_contour']}")
        lines.append(f"Movement: {m['movement_type']}")
        lines.append("")

    # Vocals
    if 'vocals' in analysis:
        v = analysis['vocals']
        lines.append("VOCALS")
        lines.append("─" * 40)
        lines.append(f"Present: {'Yes' if v['has_vocals'] else 'No/minimal'}")
        if v['has_vocals']:
            lines.append(f"Type: {v.get('vocal_type', 'unknown')}")
            lines.append(f"Mode: {v.get('dominant_mode', 'unknown')}")
            if v.get('secondary_modes'):
                lines.append(f"Also: {', '.join(v['secondary_modes'])}")
            lines.append(f"Register: {v['register']}")
            lines.append(f"Dynamics: {v['dynamics']}")
            lines.append(f"Pacing: {v['pacing']}")
            lines.append(f"Vibrato: {v['vibrato']}")
            lines.append(f"Articulation: {v['articulation']}")
            # Show harsh vocal segments (ZCR-detected)
            if v.get('harsh_segments'):
                lines.append(f"Harsh character: {v.get('harsh_character', 'unknown')}")
                timestamps = []
                for s in v['harsh_segments'][:5]:
                    mins = int(s['start'] // 60)
                    secs = int(s['start'] % 60)
                    timestamps.append(f"{mins}:{secs:02d} ({s['type']})")
                lines.append(f"Harsh vocals at: {', '.join(timestamps)}")
            # Show belting segments (energy-detected)
            if v.get('intensity_segments'):
                belt_segs = [s for s in v['intensity_segments'] if s['type'] == 'belting']
                if belt_segs:
                    timestamps = []
                    for s in belt_segs[:5]:
                        mins = int(s['start'] // 60)
                        secs = int(s['start'] % 60)
                        timestamps.append(f"{mins}:{secs:02d}")
                    lines.append(f"Belting at: {', '.join(timestamps)}")
        lines.append("")

    # Lyrics
    if 'lyrics' in analysis:
        l = analysis['lyrics']
        lines.append("LYRICS")
        lines.append("─" * 40)
        if l.get('text'):
            # Show first 500 chars
            text = l['text'][:500]
            if len(l['text']) > 500:
                text += "..."
            lines.append(text)
        lines.append("")

    # Narrative
    if 'narrative' in analysis:
        lines.append("═" * 60)
        lines.append("WHAT THIS SONG IS")
        lines.append("═" * 60)
        lines.append("")
        lines.append(analysis['narrative'])
        lines.append("")

    lines.append("═" * 60)
    return "\n".join(lines)


def save_bundle(analysis: dict, output_dir: str) -> str:
    """Save full analysis bundle to directory.

    Creates:
    - analysis.json (raw data)
    - analysis.txt (formatted text)
    - narrative.md (just the narrative if present)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save raw JSON
    with open(out / "analysis.json", "w", encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)

    # Save formatted text
    with open(out / "analysis.txt", "w", encoding='utf-8') as f:
        f.write(format_analysis_text(analysis))

    # Save narrative separately if present
    if 'narrative' in analysis:
        with open(out / "narrative.md", "w", encoding='utf-8') as f:
            f.write(f"# {analysis.get('file', 'Song')}\n\n")
            f.write(analysis['narrative'])

    return str(out)
