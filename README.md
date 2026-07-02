<img width="1024" height="1024" alt="Ear" src="https://github.com/user-attachments/assets/3ac16dd0-194c-4263-9017-8bc40f0c8f44" />

# Ear

Audio perception for Claude and other language models.

Ear takes an audio file, analyzes the sound, optionally adds lyrics, and asks an LLM to write a grounded description of what the song feels like. It is not only metadata extraction. The point is to turn sound into an experience report a model can reason from.

## What It Does

Ear runs a multi-stage pipeline:

1. Load an audio file with `librosa`.
2. Analyze structure, harmony, timbre, rhythm, melody, and vocals.
3. Detect harsh vocals with a trained classifier.
4. Optionally separate stems with Demucs through Replicate.
5. Use pasted lyrics or transcribe with local Whisper.
6. Send the structured analysis to an LLM for narrative synthesis.
7. Save a `.ear` bundle with raw data, readable analysis, narrative, and field notes.

## Current Entrypoints

- `ear_gui.pyw` - desktop GUI.
- `run.py` - command line interface.
- `core.py` - programmatic pipeline.

Older experimental entrypoints have been removed so the repo points at the current app.

## Install

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

Install FFmpeg too. Whisper and some audio formats need it.

Windows:

```powershell
winget install ffmpeg
```

macOS:

```bash
brew install ffmpeg
```

Linux:

```bash
sudo apt install ffmpeg
```

## API Keys

Ear can read keys from `~/.keywallet.json`, environment variables, or a parent `.env` file.

Example `~/.keywallet.json`:

```json
{
  "anthropic": "sk-ant-...",
  "openai": "sk-...",
  "replicate": "r8_...",
  "dashscope": "sk-..."
}
```

What each key is for:

- Anthropic: Claude narrative synthesis.
- OpenAI: GPT narrative synthesis.
- Replicate: optional Demucs source separation.
- DashScope: optional Qwen model routing.
- Ollama: no key; run Ollama locally and use a model name like `ollama:llama3.2`.

Local Whisper transcription does not need an API key, but it downloads/runs a local Whisper model.

## Run The GUI

```bash
python ear_gui.pyw
```

Then:

1. Drop an audio file or click **Browse**.
2. Pick a synthesis model.
3. Choose options:
   - **Transcribe**: local Whisper lyrics.
   - **Separate**: stem separation through Replicate.
   - **Synthesize**: LLM narrative.
4. Optional: click **Lyrics** and paste lyrics manually. Pasted lyrics override Whisper.
5. Click **Copy** or **Save Bundle**.

Analysis auto-saves to a `.ear` folder next to the source audio.

## Run From Command Line

```bash
python run.py song.mp3
```

Useful options:

```bash
python run.py song.mp3 --no-synth
python run.py song.mp3 --no-transcribe
python run.py song.mp3 --separate
python run.py song.mp3 --model gpt-4o
python run.py song.mp3 --model ollama:llama3.2
python run.py song.mp3 --json
```

## Output Bundle

Each `.ear` folder can contain:

- `analysis.json` - raw structured analysis.
- `analysis.txt` - readable report.
- `narrative.md` - synthesized description.
- `field_notes.md` - human annotation template.

## Architecture

```text
ear/
  core.py
  ear_gui.pyw
  run.py
  analyzers/
    structure.py
    harmony.py
    timbre.py
    rhythm.py
    melody.py
    vocals.py
  harsh_classifier/
    models/harsh_classifier.joblib
    features.py
    predict.py
    train.py
```

The vocal analyzer is the most developed part. It combines pitch/range analysis, vocal mode detection, clean intensity detection, and a Random Forest harsh-vocal classifier. The classifier exists because simple energy/ZCR heuristics confused powerful clean belting with actual harsh vocals.

## Package

The PyInstaller spec is:

```bash
pyinstaller ear.spec
```

It bundles the GUI, splash screen, icon, and harsh vocal classifier model.

## Status

Beta. The core pipeline works, but this is still a working tool:

- Install size can be large because audio/ML dependencies are heavy.
- Local Whisper can be slow on CPU.
- Stem separation requires a Replicate key.
- Some analysis is approximate; the saved JSON is evidence, not final truth.

## License

Do what you want with it. If it helps you hear music through new ears, that is the point.
