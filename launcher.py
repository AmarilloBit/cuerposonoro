"""
Cuerpo Sonoro — GUI Launcher

A tkinter GUI to configure and launch main.py with the desired arguments.
"""

import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, ttk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(ROOT_DIR, "main.py")

MODES = ["osc", "midi"]
MIDI_MODES = ["classic", "rhythmical"]
BACKENDS = ["auto", "cpu", "metal", "tensorrt"]


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Cuerpo Sonoro — Launcher")
        self.root.resizable(False, False)

        self.process: subprocess.Popen | None = None

        self._build_ui()
        self._on_mode_change()  # set initial visibility

    # -- UI construction ----------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        frame = ttk.Frame(self.root, padding=16)
        frame.grid(sticky="nsew")

        row = 0

        # -- Output mode ----------------------------------------------------
        ttk.Label(frame, text="Output mode:").grid(row=row, column=0, sticky="w", **pad)
        self.mode_var = tk.StringVar(value="midi")
        mode_combo = ttk.Combobox(
            frame, textvariable=self.mode_var, values=MODES,
            state="readonly", width=22,
        )
        mode_combo.grid(row=row, column=1, sticky="w", **pad)
        mode_combo.bind("<<ComboboxSelected>>", lambda _: self._on_mode_change())
        row += 1

        # -- MIDI mode ------------------------------------------------------
        self.midi_mode_label = ttk.Label(frame, text="MIDI mode:")
        self.midi_mode_label.grid(row=row, column=0, sticky="w", **pad)
        self.midi_mode_var = tk.StringVar(value="classic")
        self.midi_mode_combo = ttk.Combobox(
            frame, textvariable=self.midi_mode_var, values=MIDI_MODES,
            state="readonly", width=22,
        )
        self.midi_mode_combo.grid(row=row, column=1, sticky="w", **pad)
        row += 1

        # -- Separator: General options -------------------------------------
        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8,
        )
        row += 1

        ttk.Label(frame, text="General options", font=("TkDefaultFont", 0, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", **pad,
        )
        row += 1

        # -- Backend --------------------------------------------------------
        ttk.Label(frame, text="Backend:").grid(row=row, column=0, sticky="w", **pad)
        self.backend_var = tk.StringVar(value="auto")
        ttk.Combobox(
            frame, textvariable=self.backend_var, values=BACKENDS,
            state="readonly", width=22,
        ).grid(row=row, column=1, sticky="w", **pad)
        row += 1

        # -- Video source ---------------------------------------------------
        ttk.Label(frame, text="Video source:").grid(row=row, column=0, sticky="w", **pad)
        source_frame = ttk.Frame(frame)
        source_frame.grid(row=row, column=1, sticky="w", **pad)
        self.source_var = tk.StringVar(value="")
        self.source_entry = ttk.Entry(source_frame, textvariable=self.source_var, width=20)
        self.source_entry.pack(side="left")
        ttk.Button(source_frame, text="Browse…", command=self._browse_source, width=8).pack(
            side="left", padx=(4, 0),
        )
        self.source_hint = ttk.Label(frame, text="Leave empty for webcam", foreground="gray")
        row += 1
        self.source_hint.grid(row=row, column=1, sticky="w", padx=8)
        row += 1

        # -- Debug ----------------------------------------------------------
        self.debug_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Debug overlay", variable=self.debug_var).grid(
            row=row, column=0, columnspan=2, sticky="w", **pad,
        )
        row += 1

        # -- Separator ------------------------------------------------------
        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8,
        )
        row += 1

        # -- Command preview ------------------------------------------------
        ttk.Label(frame, text="Command:").grid(row=row, column=0, sticky="nw", **pad)
        self.cmd_text = tk.Text(frame, height=3, width=50, state="disabled", wrap="word")
        self.cmd_text.grid(row=row, column=1, sticky="w", **pad)
        row += 1

        # -- Buttons --------------------------------------------------------
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=(12, 0))

        self.launch_btn = ttk.Button(btn_frame, text="Launch", command=self._launch)
        self.launch_btn.pack(side="left", padx=4)

        self.stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        # Trace variables to update command preview
        for var in (self.mode_var, self.midi_mode_var, self.backend_var,
                    self.source_var, self.debug_var):
            var.trace_add("write", lambda *_: self._update_cmd_preview())

        self._update_cmd_preview()

    # -- Callbacks ----------------------------------------------------------

    def _on_mode_change(self):
        is_midi = self.mode_var.get() == "midi"
        state = "readonly" if is_midi else "disabled"
        self.midi_mode_combo.configure(state=state)
        self._update_cmd_preview()

    def _browse_source(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if path:
            self.source_var.set(path)

    # -- Command building ---------------------------------------------------

    def _build_command(self) -> list[str]:
        cmd = [sys.executable, MAIN_SCRIPT]

        cmd.extend(["--mode", self.mode_var.get()])

        if self.mode_var.get() == "midi":
            cmd.extend(["--midi-mode", self.midi_mode_var.get()])

        backend = self.backend_var.get()
        if backend != "auto":
            cmd.extend(["--backend", backend])

        source = self.source_var.get().strip()
        if source:
            cmd.extend(["--source", source])

        if self.debug_var.get():
            cmd.append("--debug")

        return cmd

    def _update_cmd_preview(self):
        cmd = self._build_command()
        # Show a clean version (replace python path with just "python")
        display = cmd.copy()
        display[0] = "python"
        text = " ".join(
            f'"{a}"' if " " in a else a for a in display
        )
        self.cmd_text.configure(state="normal")
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("1.0", text)
        self.cmd_text.configure(state="disabled")

    # -- Launch / Stop ------------------------------------------------------

    def _launch(self):
        cmd = self._build_command()
        self.process = subprocess.Popen(cmd, cwd=ROOT_DIR)
        self.launch_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._poll_process()

    def _stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self._reset_buttons()

    def _poll_process(self):
        if self.process and self.process.poll() is not None:
            self._reset_buttons()
        else:
            self.root.after(500, self._poll_process)

    def _reset_buttons(self):
        self.launch_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.process = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
