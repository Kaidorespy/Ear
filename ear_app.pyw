"""
Ear - Audio description for Claude
Drag an audio file, get a description of what it sounds like.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext, filedialog
from pathlib import Path
import subprocess
import tempfile

import numpy as np

# Try to import librosa, guide user if missing
try:
    import librosa
except ImportError:
    print("Missing librosa. Run: pip install librosa")
    sys.exit(1)


def analyze_audio(filepath, progress_callback=None):
    """Analyze audio and return musical descriptions."""

    if progress_callback:
        progress_callback("Loading audio...")

    # Convert to wav if needed using ffmpeg
    ext = Path(filepath).suffix.lower()
    if ext not in ['.wav', '.mp3', '.flac', '.ogg']:
        if progress_callback:
            progress_callback(f"Converting {ext} to wav...")
        temp_wav = tempfile.mktemp(suffix='.wav')
        try:
            subprocess.run([
                'ffmpeg', '-i', filepath, '-ar', '22050', '-ac', '1',
                '-y', temp_wav
            ], capture_output=True, check=True)
            filepath = temp_wav
        except Exception as e:
            return f"Error converting audio: {e}\nMake sure ffmpeg is installed."

    try:
        y, sr = librosa.load(filepath, sr=22050)
    except Exception as e:
        return f"Error loading audio: {e}"

    duration = librosa.get_duration(y=y, sr=sr)

    if progress_callback:
        progress_callback("Analyzing tempo and beats...")

    # Get tempo and beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if hasattr(tempo, '__float__') else tempo[0]
    beat_times = librosa.frames_to_time(beats, sr=sr)

    if progress_callback:
        progress_callback("Analyzing energy...")

    # Energy over time (RMS)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    if progress_callback:
        progress_callback("Detecting onsets...")

    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    onset_times = librosa.frames_to_time(onsets, sr=sr)

    if progress_callback:
        progress_callback("Finding sections...")

    # Segment into sections
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    bounds = librosa.segment.agglomerative(chroma, k=min(8, max(2, int(duration / 30))))
    bound_times = librosa.frames_to_time(bounds, sr=sr)

    if progress_callback:
        progress_callback("Generating description...")

    # Build description
    lines = []
    lines.append("═" * 50)
    lines.append("EAR ANALYSIS")
    lines.append("═" * 50)
    lines.append("")
    lines.append(f"File: {Path(filepath).name}")
    lines.append(f"Duration: {int(duration // 60)}:{int(duration % 60):02d}")
    lines.append(f"Tempo: ~{tempo:.0f} BPM")
    lines.append("")
    lines.append("ENERGY ARC")
    lines.append("─" * 50)

    # Describe in chunks
    chunk_duration = 15
    num_chunks = int(np.ceil(duration / chunk_duration))

    for i in range(num_chunks):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, duration)

        start_idx = int(start_time / duration * len(rms))
        end_idx = int(end_time / duration * len(rms))

        if start_idx >= end_idx:
            continue

        chunk_rms = rms[start_idx:end_idx]
        chunk_centroid = spectral_centroid[start_idx:end_idx]

        # Energy level
        avg_energy = np.mean(chunk_rms)
        max_energy = np.max(rms)
        energy_pct = (avg_energy / max_energy) * 100 if max_energy > 0 else 0

        # Brightness
        avg_brightness = np.mean(chunk_centroid)
        max_brightness = np.max(spectral_centroid)
        brightness_pct = (avg_brightness / max_brightness) * 100 if max_brightness > 0 else 0

        # Trend
        if len(chunk_rms) > 1:
            trend = chunk_rms[-1] - chunk_rms[0]
            trend_word = "building" if trend > 0.01 else "dropping" if trend < -0.01 else "steady"
        else:
            trend_word = "steady"

        # Hit density
        hits = np.sum((onset_times >= start_time) & (onset_times < end_time))
        hits_per_sec = hits / (end_time - start_time)

        # Words
        if energy_pct > 80:
            energy_word = "LOUD"
        elif energy_pct > 60:
            energy_word = "high energy"
        elif energy_pct > 40:
            energy_word = "medium"
        elif energy_pct > 20:
            energy_word = "pulled back"
        else:
            energy_word = "quiet"

        if brightness_pct > 70:
            bright_word = "bright"
        elif brightness_pct > 40:
            bright_word = "full"
        else:
            bright_word = "warm"

        if hits_per_sec > 8:
            density_word = "dense"
        elif hits_per_sec > 4:
            density_word = "driving"
        elif hits_per_sec > 2:
            density_word = "moderate"
        else:
            density_word = "sparse"

        ts = f"{int(start_time // 60)}:{int(start_time % 60):02d}"
        lines.append(f"[{ts}] {energy_word}, {bright_word}, {density_word}, {trend_word}")

    lines.append("")
    lines.append("SECTIONS")
    lines.append("─" * 50)

    for i, t in enumerate(bound_times):
        ts = f"{int(t // 60)}:{int(t % 60):02d}"
        lines.append(f"  {ts} - Section {i+1}")

    lines.append("")
    lines.append("MOMENTS")
    lines.append("─" * 50)

    # Find energy changes
    rms_diff = np.diff(rms)
    threshold = np.std(rms_diff) * 2
    big_changes = np.where(np.abs(rms_diff) > threshold)[0]

    last_t = -5
    moment_count = 0
    for idx in big_changes:
        if moment_count >= 8:
            break
        t = times[idx]
        if t - last_t > 3:
            direction = "↑ spike" if rms_diff[idx] > 0 else "↓ drop"
            ts = f"{int(t // 60)}:{int(t % 60):02d}"
            lines.append(f"  {ts} - {direction}")
            last_t = t
            moment_count += 1

    lines.append("")
    lines.append("═" * 50)

    return "\n".join(lines)


class EarApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ear")
        self.root.configure(bg='#1a1a2e')
        self.root.geometry("600x500")

        # Header
        header = tk.Label(
            self.root,
            text="EAR",
            font=('Consolas', 24, 'bold'),
            fg='#00d9ff',
            bg='#1a1a2e'
        )
        header.pack(pady=(20, 5))

        subtitle = tk.Label(
            self.root,
            text="drop audio here or click to browse",
            font=('Consolas', 10),
            fg='#666',
            bg='#1a1a2e'
        )
        subtitle.pack(pady=(0, 10))

        # Status
        self.status = tk.Label(
            self.root,
            text="Ready",
            font=('Consolas', 9),
            fg='#00d9ff',
            bg='#1a1a2e'
        )
        self.status.pack(pady=(0, 10))

        # Text output
        self.output = scrolledtext.ScrolledText(
            self.root,
            font=('Consolas', 10),
            bg='#0d1117',
            fg='white',
            insertbackground='#00d9ff',
            relief='flat',
            wrap=tk.WORD
        )
        self.output.pack(fill='both', expand=True, padx=20, pady=(0, 10))

        # Buttons
        btn_frame = tk.Frame(self.root, bg='#1a1a2e')
        btn_frame.pack(pady=(0, 20))

        browse_btn = tk.Button(
            btn_frame,
            text="Browse",
            font=('Consolas', 10),
            bg='#16213e',
            fg='white',
            activebackground='#00d9ff',
            activeforeground='#1a1a2e',
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
            bg='#16213e',
            fg='white',
            activebackground='#00d9ff',
            activeforeground='#1a1a2e',
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=5,
            command=self.copy_output
        )
        copy_btn.pack(side='left', padx=5)

        # Enable drag and drop
        self.root.drop_target_register = lambda *a: None  # placeholder
        self.setup_drop()

        # Click anywhere to browse
        self.output.bind('<Button-1>', lambda e: self.browse() if not self.output.get('1.0', 'end').strip() else None)

    def setup_drop(self):
        """Setup drag and drop - works on Windows with tkinterdnd2 if available."""
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD
            # Re-init with DnD support
            self.root.destroy()
            self.root = TkinterDnD.Tk()
            self.__init__()
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)
        except ImportError:
            # No drag-drop, just use browse button
            pass

    def on_drop(self, event):
        """Handle dropped file."""
        filepath = event.data
        # Clean up path (remove braces on Windows)
        if filepath.startswith('{') and filepath.endswith('}'):
            filepath = filepath[1:-1]
        self.analyze(filepath)

    def browse(self):
        """Open file browser."""
        filepath = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.flac *.ogg *.m4a"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self.analyze(filepath)

    def analyze(self, filepath):
        """Run analysis in background thread."""
        self.output.delete('1.0', 'end')
        self.output.insert('1.0', f"Analyzing: {Path(filepath).name}\n\n")
        self.status.config(text="Analyzing...")

        def run():
            result = analyze_audio(filepath, self.update_status)
            self.root.after(0, lambda: self.show_result(result, filepath))

        threading.Thread(target=run, daemon=True).start()

    def update_status(self, msg):
        """Update status from background thread."""
        self.root.after(0, lambda: self.status.config(text=msg))

    def show_result(self, result, filepath):
        """Show analysis result."""
        self.output.delete('1.0', 'end')
        self.output.insert('1.0', result)
        self.status.config(text="Done")

        # Save to file
        out_path = Path(filepath).with_suffix('.ear.txt')
        try:
            with open(out_path, 'w') as f:
                f.write(result)
            self.status.config(text=f"Saved: {out_path.name}")
        except:
            pass

    def copy_output(self):
        """Copy output to clipboard."""
        content = self.output.get('1.0', 'end').strip()
        if content:
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.status.config(text="Copied!")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # Handle command line arg
    if len(sys.argv) > 1:
        result = analyze_audio(sys.argv[1], print)
        print(result)
    else:
        app = EarApp()
        app.run()
