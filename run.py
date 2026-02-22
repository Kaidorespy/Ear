#!/usr/bin/env python
"""
Ear CLI - Analyze audio from command line
"""

import sys
import argparse
from pathlib import Path

from core import run_full_analysis, format_analysis_text, save_bundle


def main():
    parser = argparse.ArgumentParser(
        description="Ear - Audio perception for Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py song.mp3                    # Basic analysis
  python run.py song.mp3 --no-synth         # Skip narrative synthesis
  python run.py song.mp3 --model gpt-4o     # Use GPT-4o for synthesis
  python run.py song.mp3 --separate         # Include source separation
  python run.py song.mp3 --json             # Output raw JSON
"""
    )

    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--model', '-m', default='claude-sonnet-4-20250514',
                        help='Model for synthesis (default: claude-sonnet-4-20250514)')
    parser.add_argument('--no-synth', action='store_true',
                        help='Skip narrative synthesis')
    parser.add_argument('--no-transcribe', action='store_true',
                        help='Skip lyric transcription')
    parser.add_argument('--separate', action='store_true',
                        help='Run source separation (requires Replicate API)')
    parser.add_argument('--json', action='store_true',
                        help='Output raw JSON instead of formatted text')
    parser.add_argument('--save', '-s', metavar='DIR',
                        help='Save bundle to directory')

    args = parser.parse_args()

    if not Path(args.audio_file).exists():
        print(f"Error: File not found: {args.audio_file}")
        sys.exit(1)

    def progress(msg):
        print(f"  {msg}")

    print(f"\nAnalyzing: {args.audio_file}\n")

    result = run_full_analysis(
        args.audio_file,
        do_separation=args.separate,
        do_transcription=not args.no_transcribe,
        do_synthesis=not args.no_synth,
        synthesis_model=args.model,
        progress_callback=progress
    )

    print()

    if args.json:
        import json
        print(json.dumps(result, indent=2))
    else:
        print(format_analysis_text(result))

    if args.save:
        bundle_path = save_bundle(result, args.save)
        print(f"\nBundle saved to: {bundle_path}")
    else:
        # Auto-save next to file
        ear_dir = Path(args.audio_file).with_suffix('.ear')
        save_bundle(result, str(ear_dir))
        print(f"\nBundle saved to: {ear_dir}")


if __name__ == "__main__":
    main()
