"""
Ear GUI - Full audio perception for Claude
"""

import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, simpledialog
from pathlib import Path

# Config file for user settings
CONFIG_PATH = Path.home() / ".ear_config.json"

def load_config():
    """Load config from file."""
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except:
            pass
    return {}

def save_config(config):
    """Save config to file."""
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

def get_ffmpeg_path():
    """Get ffmpeg path from config or default locations."""
    config = load_config()
    if config.get('ffmpeg_path') and Path(config['ffmpeg_path']).exists():
        return config['ffmpeg_path']
    # Check default locations
    defaults = [
        r"C:\Users\Casey\Projects\filetriage\ffmpeg\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for p in defaults:
        if Path(p).exists():
            return p
    # Check PATH
    import shutil
    return shutil.which('ffmpeg')

def check_api_keys():
    """Check which API keys are configured."""
    # Check keywallet
    wallet_path = Path.home() / ".keywallet.json"
    keys = {'anthropic': False, 'openai': False}
    if wallet_path.exists():
        try:
            wallet = json.loads(wallet_path.read_text())
            keys['anthropic'] = bool(wallet.get('anthropic'))
            keys['openai'] = bool(wallet.get('openai'))
        except:
            pass
    # Check env vars
    if os.environ.get('ANTHROPIC_API_KEY'):
        keys['anthropic'] = True
    if os.environ.get('OPENAI_API_KEY'):
        keys['openai'] = True
    return keys

def check_ollama():
    """Check if Ollama is running locally."""
    import urllib.request
    try:
        req = urllib.request.urlopen('http://localhost:11434/api/tags', timeout=1)
        return req.status == 200
    except:
        return False

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
                "gpt-4o-mini",
                "qwen-plus",
                "qwen-turbo",
                "qwen2.5-72b-instruct",
                "ollama:llama3.2",
                "ollama:qwen2.5:14b",
            ],
            width=25,
            font=('Consolas', 9)
        )
        # Editable - type any model name (e.g., "ollama:your-model")
        # Enter key confirms selection and removes cursor
        model_dropdown.bind('<Return>', lambda e: self.root.focus_set())
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

        settings_btn = tk.Button(
            btn_frame,
            text="Settings",
            font=('Consolas', 10),
            bg='#2a2420',
            fg='white',
            activebackground='#f59e0b',
            activeforeground='#1a1714',
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=5,
            command=self.open_settings
        )
        settings_btn.pack(side='left', padx=5)

        # Status bar for connections
        status_bar = tk.Frame(self.root, bg='#1a1714')
        status_bar.pack(fill='x', padx=20, pady=(0, 10))

        # FFmpeg status
        ffmpeg_path = get_ffmpeg_path()
        ffmpeg_status = "FFmpeg: ✓" if ffmpeg_path else "FFmpeg: ✗"
        ffmpeg_color = "#4ade80" if ffmpeg_path else "#ef4444"
        self.ffmpeg_label = tk.Label(
            status_bar,
            text=ffmpeg_status,
            font=('Consolas', 8),
            fg=ffmpeg_color,
            bg='#1a1714'
        )
        self.ffmpeg_label.pack(side='left', padx=(0, 15))

        # API key status
        api_keys = check_api_keys()
        anthropic_status = "Anthropic: ✓" if api_keys['anthropic'] else "Anthropic: ✗"
        anthropic_color = "#4ade80" if api_keys['anthropic'] else "#ef4444"
        self.anthropic_label = tk.Label(
            status_bar,
            text=anthropic_status,
            font=('Consolas', 8),
            fg=anthropic_color,
            bg='#1a1714'
        )
        self.anthropic_label.pack(side='left', padx=(0, 15))

        openai_status = "OpenAI: ✓" if api_keys['openai'] else "OpenAI: ✗"
        openai_color = "#4ade80" if api_keys['openai'] else "#ef4444"
        self.openai_label = tk.Label(
            status_bar,
            text=openai_status,
            font=('Consolas', 8),
            fg=openai_color,
            bg='#1a1714'
        )
        self.openai_label.pack(side='left', padx=(0, 15))

        # Ollama status
        ollama_running = check_ollama()
        ollama_status = "Ollama: ✓" if ollama_running else "Ollama: ✗"
        ollama_color = "#4ade80" if ollama_running else "#888"  # Gray when not running (optional)
        self.ollama_label = tk.Label(
            status_bar,
            text=ollama_status,
            font=('Consolas', 8),
            fg=ollama_color,
            bg='#1a1714'
        )
        self.ollama_label.pack(side='left')

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

    def open_settings(self):
        """Open settings dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Ear Settings")
        dialog.configure(bg='#1a1714')
        dialog.geometry("500x350")
        dialog.transient(self.root)
        dialog.grab_set()

        # FFmpeg section
        ffmpeg_frame = tk.LabelFrame(
            dialog, text="FFmpeg", font=('Consolas', 10),
            fg='#f59e0b', bg='#1a1714', padx=10, pady=10
        )
        ffmpeg_frame.pack(fill='x', padx=20, pady=(20, 10))

        config = load_config()
        current_ffmpeg = get_ffmpeg_path() or "Not found"

        ffmpeg_var = tk.StringVar(value=current_ffmpeg)
        ffmpeg_entry = tk.Entry(
            ffmpeg_frame, textvariable=ffmpeg_var,
            font=('Consolas', 9), bg='#2a2420', fg='white',
            insertbackground='#f59e0b', width=50
        )
        ffmpeg_entry.pack(side='left', fill='x', expand=True)

        def browse_ffmpeg():
            path = filedialog.askopenfilename(
                title="Select ffmpeg executable",
                filetypes=[("Executable", "*.exe"), ("All files", "*.*")]
            )
            if path:
                ffmpeg_var.set(path)

        browse_btn = tk.Button(
            ffmpeg_frame, text="Browse", font=('Consolas', 9),
            bg='#2a2420', fg='white', command=browse_ffmpeg
        )
        browse_btn.pack(side='left', padx=(10, 0))

        # API Keys section
        api_frame = tk.LabelFrame(
            dialog, text="API Keys (saved to ~/.keywallet.json)",
            font=('Consolas', 10), fg='#f59e0b', bg='#1a1714', padx=10, pady=10
        )
        api_frame.pack(fill='x', padx=20, pady=10)

        # Load existing keys
        wallet_path = Path.home() / ".keywallet.json"
        wallet = {}
        if wallet_path.exists():
            try:
                wallet = json.loads(wallet_path.read_text())
            except:
                pass

        # Anthropic key
        tk.Label(
            api_frame, text="Anthropic:", font=('Consolas', 9),
            fg='#888', bg='#1a1714'
        ).grid(row=0, column=0, sticky='w', pady=5)

        anthropic_var = tk.StringVar(value=wallet.get('anthropic', ''))
        anthropic_entry = tk.Entry(
            api_frame, textvariable=anthropic_var, show='*',
            font=('Consolas', 9), bg='#2a2420', fg='white',
            insertbackground='#f59e0b', width=45
        )
        anthropic_entry.grid(row=0, column=1, pady=5, padx=(10, 0))

        # OpenAI key
        tk.Label(
            api_frame, text="OpenAI:", font=('Consolas', 9),
            fg='#888', bg='#1a1714'
        ).grid(row=1, column=0, sticky='w', pady=5)

        openai_var = tk.StringVar(value=wallet.get('openai', ''))
        openai_entry = tk.Entry(
            api_frame, textvariable=openai_var, show='*',
            font=('Consolas', 9), bg='#2a2420', fg='white',
            insertbackground='#f59e0b', width=45
        )
        openai_entry.grid(row=1, column=1, pady=5, padx=(10, 0))

        # Replicate key
        tk.Label(
            api_frame, text="Replicate:", font=('Consolas', 9),
            fg='#888', bg='#1a1714'
        ).grid(row=2, column=0, sticky='w', pady=5)

        replicate_var = tk.StringVar(value=wallet.get('replicate', ''))
        replicate_entry = tk.Entry(
            api_frame, textvariable=replicate_var, show='*',
            font=('Consolas', 9), bg='#2a2420', fg='white',
            insertbackground='#f59e0b', width=45
        )
        replicate_entry.grid(row=2, column=1, pady=5, padx=(10, 0))

        # Save button
        def save_settings():
            # Save ffmpeg path
            config = load_config()
            ffmpeg_path = ffmpeg_var.get()
            if ffmpeg_path and ffmpeg_path != "Not found":
                config['ffmpeg_path'] = ffmpeg_path
            save_config(config)

            # Save API keys to keywallet
            wallet = {}
            if wallet_path.exists():
                try:
                    wallet = json.loads(wallet_path.read_text())
                except:
                    pass

            if anthropic_var.get():
                wallet['anthropic'] = anthropic_var.get()
            if openai_var.get():
                wallet['openai'] = openai_var.get()
            if replicate_var.get():
                wallet['replicate'] = replicate_var.get()

            wallet_path.write_text(json.dumps(wallet, indent=2))

            # Update status indicators
            self.update_status_indicators()

            dialog.destroy()
            self.status.config(text="Settings saved!")

        save_btn = tk.Button(
            dialog, text="Save", font=('Consolas', 10),
            bg='#f59e0b', fg='#1a1714', padx=30, pady=5,
            command=save_settings
        )
        save_btn.pack(pady=20)

    def update_status_indicators(self):
        """Update the status bar indicators."""
        ffmpeg_path = get_ffmpeg_path()
        ffmpeg_status = "FFmpeg: ✓" if ffmpeg_path else "FFmpeg: ✗"
        ffmpeg_color = "#4ade80" if ffmpeg_path else "#ef4444"
        self.ffmpeg_label.config(text=ffmpeg_status, fg=ffmpeg_color)

        api_keys = check_api_keys()
        anthropic_status = "Anthropic: ✓" if api_keys['anthropic'] else "Anthropic: ✗"
        anthropic_color = "#4ade80" if api_keys['anthropic'] else "#ef4444"
        self.anthropic_label.config(text=anthropic_status, fg=anthropic_color)

        openai_status = "OpenAI: ✓" if api_keys['openai'] else "OpenAI: ✗"
        openai_color = "#4ade80" if api_keys['openai'] else "#ef4444"
        self.openai_label.config(text=openai_status, fg=openai_color)

    def run(self):
        self.root.mainloop()


def close_splash():
    """Close PyInstaller splash screen if running from frozen exe."""
    try:
        import pyi_splash
        pyi_splash.close()
    except ImportError:
        pass  # Not running from PyInstaller bundle


if __name__ == "__main__":
    # Handle command line
    if len(sys.argv) > 1:
        close_splash()
        from core import run_full_analysis, format_analysis_text
        result = run_full_analysis(sys.argv[1], progress_callback=print)
        print(format_analysis_text(result))
    else:
        app = EarApp()
        close_splash()  # Close splash once window is built
        app.run()
