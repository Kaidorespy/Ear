"""
Ear GUI - Full audio perception for Claude
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Check dependencies
try:
    import numpy as np
    import librosa
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install numpy librosa")
    sys.exit(1)

from core import run_full_analysis, format_analysis_text, save_bundle


class EarApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ear")
        self.root.configure(bg='#1a1714')
        self.root.geometry("800x700")

        self.current_analysis = None
        self.dnd_initialized = False

        self.build_ui()

    def build_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg='#1a1714')
        header_frame.pack(fill='x', padx=20, pady=(20, 5))

        title = tk.Label(
            header_frame,
            text="EAR",
            font=('Consolas', 28, 'bold'),
            fg='#f59e0b',
            bg='#1a1714'
        )
        title.pack(side='left')

        subtitle = tk.Label(
            header_frame,
            text="audio perception for claude",
            font=('Consolas', 10),
            fg='#666',
            bg='#1a1714'
        )
        subtitle.pack(side='left', padx=(10, 0), pady=(12, 0))

        # Options frame
        options_frame = tk.Frame(self.root, bg='#1a1714')
        options_frame.pack(fill='x', padx=20, pady=10)

        # Model selection
        model_label = tk.Label(
            options_frame,
            text="Model:",
            font=('Consolas', 10),
            fg='#888',
            bg='#1a1714'
        )
        model_label.pack(side='left')

        self.model_var = tk.StringVar(value="claude-sonnet-4-20250514")
        model_dropdown = ttk.Combobox(
            options_frame,
            textvariable=self.model_var,
            values=[
                "claude-sonnet-4-20250514",
                "claude-opus-4-20250514",
                "claude-3-5-haiku-20241022",
                "gpt-4o",
                "gpt-4o-mini"
            ],
            state="readonly",
            width=25,
            font=('Consolas', 9)
        )
        model_dropdown.pack(side='left', padx=(5, 20))

        # Checkboxes
        self.do_transcription = tk.BooleanVar(value=True)
        self.do_separation = tk.BooleanVar(value=False)
        self.do_synthesis = tk.BooleanVar(value=True)

        trans_check = tk.Checkbutton(
            options_frame,
            text="Transcribe",
            variable=self.do_transcription,
            font=('Consolas', 9),
            fg='#888',
            bg='#1a1714',
            selectcolor='#2a2420',
            activebackground='#1a1714',
            activeforeground='#f59e0b'
        )
        trans_check.pack(side='left', padx=5)

        sep_check = tk.Checkbutton(
            options_frame,
            text="Separate",
            variable=self.do_separation,
            font=('Consolas', 9),
            fg='#888',
            bg='#1a1714',
            selectcolor='#2a2420',
            activebackground='#1a1714',
            activeforeground='#f59e0b'
        )
        sep_check.pack(side='left', padx=5)

        synth_check = tk.Checkbutton(
            options_frame,
            text="Synthesize",
            variable=self.do_synthesis,
            font=('Consolas', 9),
            fg='#888',
            bg='#1a1714',
            selectcolor='#2a2420',
            activebackground='#1a1714',
            activeforeground='#f59e0b'
        )
        synth_check.pack(side='left', padx=5)

        # Status
        self.status = tk.Label(
            self.root,
            text="Ready - drop audio or click Browse",
            font=('Consolas', 9),
            fg='#f59e0b',
            bg='#1a1714'
        )
        self.status.pack(pady=(0, 10))

        # Progress bar
        self.progress = ttk.Progressbar(
            self.root,
            mode='indeterminate',
            length=300
        )

        # Text output
        self.output = scrolledtext.ScrolledText(
            self.root,
            font=('Consolas', 10),
            bg='#12100e',
            fg='white',
            insertbackground='#f59e0b',
            relief='flat',
            wrap=tk.WORD
        )
        self.output.pack(fill='both', expand=True, padx=20, pady=(0, 10))

        # Welcome message
        self.output.insert('1.0', """
╔══════════════════════════════════════════════════════════╗
║                         EAR                              ║
║            Audio Perception for Claude                   ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Drop an audio file here or click Browse to select one.  ║
║                                                          ║
║  Ear will analyze:                                       ║
║    • Structure (tempo, sections, energy arc)             ║
║    • Harmony (key, chords, tension)                      ║
║    • Timbre (brightness, texture, space)                 ║
║    • Rhythm (groove, swing, syncopation)                 ║
║    • Melody (contour, range, movement)                   ║
║    • Vocals (delivery, pacing, emotion)                  ║
║                                                          ║
║  Options:                                                ║
║    • Transcribe: Get lyrics via Whisper                  ║
║    • Separate: Split into stems (vocals/drums/etc)       ║
║    • Synthesize: Generate narrative description          ║
║                                                          ║
║  Select a model for the synthesis step.                  ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

        # Buttons
        btn_frame = tk.Frame(self.root, bg='#1a1714')
        btn_frame.pack(pady=(0, 20))

        browse_btn = tk.Button(
            btn_frame,
            text="Browse",
            font=('Consolas', 10),
            bg='#2a2420',
            fg='white',
            activebackground='#f59e0b',
            activeforeground='#1a1714',
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=5,
            command=self.browse
        )
        browse_btn.pack(side='left', padx=5)

        copy_btn = tk.Button(
            btn_frame,
            text="Copy",
            font=('Consolas', 10),
            bg='#2a2420',
            fg='white',
            activebackground='#f59e0b',
            activeforeground='#1a1714',
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=5,
            command=self.copy_output
        )
        copy_btn.pack(side='left', padx=5)

        save_btn = tk.Button(
            btn_frame,
            text="Save Bundle",
            font=('Consolas', 10),
            bg='#2a2420',
            fg='white',
            activebackground='#f59e0b',
            activeforeground='#1a1714',
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=5,
            command=self.save_bundle
        )
        save_btn.pack(side='left', padx=5)

        # Try to setup drag and drop (only once)
        if not self.dnd_initialized:
            self.setup_drop()

    def setup_drop(self):
        """Setup drag and drop if tkinterdnd2 is available."""
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD

            # Need to recreate window with DnD support
            self.root.destroy()
            self.root = TkinterDnD.Tk()
            self.root.title("Ear")
            self.root.configure(bg='#1a1714')
            self.root.geometry("800x700")

            self.dnd_initialized = True
            self.build_ui()

            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)

        except ImportError:
            # No drag-drop support, that's fine
            pass

    def on_drop(self, event):
        """Handle dropped file."""
        filepath = event.data
        if filepath.startswith('{') and filepath.endswith('}'):
            filepath = filepath[1:-1]
        self.analyze(filepath)

    def browse(self):
        """Open file browser."""
        filepath = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.wma"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self.analyze(filepath)

    def analyze(self, filepath):
        """Run analysis in background thread."""
        self.output.delete('1.0', 'end')
        self.output.insert('1.0', f"Analyzing: {Path(filepath).name}\n\n")
        self.status.config(text="Starting analysis...")

        # Show progress bar
        self.progress.pack(pady=5)
        self.progress.start(10)

        def run():
            try:
                result = run_full_analysis(
                    filepath,
                    do_separation=self.do_separation.get(),
                    do_transcription=self.do_transcription.get(),
                    do_synthesis=self.do_synthesis.get(),
                    synthesis_model=self.model_var.get(),
                    progress_callback=self.update_status
                )
                self.root.after(0, lambda: self.show_result(result, filepath))
            except Exception as e:
                self.root.after(0, lambda: self.show_error(str(e)))

        threading.Thread(target=run, daemon=True).start()

    def update_status(self, msg):
        """Update status from background thread."""
        self.root.after(0, lambda: self.status.config(text=msg))

    def show_result(self, result, filepath):
        """Show analysis result."""
        self.progress.stop()
        self.progress.pack_forget()

        self.current_analysis = result

        if 'error' in result:
            self.output.delete('1.0', 'end')
            self.output.insert('1.0', f"Error: {result['error']}")
            self.status.config(text="Error")
            return

        formatted = format_analysis_text(result)
        self.output.delete('1.0', 'end')
        self.output.insert('1.0', formatted)
        self.status.config(text="Done!")

        # Auto-save to .ear directory next to file
        try:
            ear_dir = Path(filepath).with_suffix('.ear')
            save_bundle(result, str(ear_dir))
            self.status.config(text=f"Saved to {ear_dir.name}/")
        except Exception as e:
            self.status.config(text=f"Done (save failed: {e})")

    def show_error(self, error):
        """Show error message."""
        self.progress.stop()
        self.progress.pack_forget()
        self.output.delete('1.0', 'end')
        self.output.insert('1.0', f"Error: {error}")
        self.status.config(text="Error")

    def copy_output(self):
        """Copy output to clipboard."""
        content = self.output.get('1.0', 'end').strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.status.config(text="Copied!")

    def save_bundle(self):
        """Save analysis bundle to chosen location."""
        if not self.current_analysis:
            self.status.config(text="Nothing to save")
            return

        directory = filedialog.askdirectory(title="Save bundle to...")
        if directory:
            filename = self.current_analysis.get('file', 'analysis')
            bundle_dir = Path(directory) / f"{Path(filename).stem}.ear"
            save_bundle(self.current_analysis, str(bundle_dir))
            self.status.config(text=f"Saved to {bundle_dir}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # Handle command line
    if len(sys.argv) > 1:
        from core import run_full_analysis, format_analysis_text
        result = run_full_analysis(sys.argv[1], progress_callback=print)
        print(format_analysis_text(result))
    else:
        app = EarApp()
        app.run()
