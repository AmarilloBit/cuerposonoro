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
CALIBRATE_SCRIPT = os.path.join(ROOT_DIR, "calibrate.py")

MODES = ["osc", "midi"]
MIDI_MODES = ["classic", "musical"]
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

        # -- Separator: Calibration -----------------------------------------
        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=8,
        )
        row += 1

        ttk.Label(frame, text="Calibration", font=("TkDefaultFont", 0, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", **pad,
        )
        row += 1

        # Video list
        ttk.Label(frame, text="Videos:").grid(row=row, column=0, sticky="nw", **pad)
        cal_list_frame = ttk.Frame(frame)
        cal_list_frame.grid(row=row, column=1, sticky="w", **pad)

        self.cal_listbox = tk.Listbox(cal_list_frame, height=4, width=40, selectmode="extended")
        self.cal_listbox.pack(side="left")

        cal_btn_frame = ttk.Frame(cal_list_frame)
        cal_btn_frame.pack(side="left", padx=(4, 0))
        ttk.Button(cal_btn_frame, text="Add…", command=self._cal_add_videos, width=7).pack(pady=1)
        ttk.Button(cal_btn_frame, text="Remove", command=self._cal_remove_videos, width=7).pack(pady=1)
        row += 1

        # Label selector
        ttk.Label(frame, text="Label:").grid(row=row, column=0, sticky="w", **pad)
        label_frame = ttk.Frame(frame)
        label_frame.grid(row=row, column=1, sticky="w", **pad)
        self.cal_label_var = tk.StringVar(value="full_body")
        ttk.Combobox(
            label_frame, textvariable=self.cal_label_var,
            values=["still", "slow_arms", "fast_arms", "pelvis", "lean", "full_body"],
            state="readonly", width=14,
        ).pack(side="left")
        ttk.Button(label_frame, text="Set label", command=self._cal_set_label, width=8).pack(
            side="left", padx=(4, 0),
        )
        row += 1

        # Apply checkbox + Run button
        cal_run_frame = ttk.Frame(frame)
        cal_run_frame.grid(row=row, column=0, columnspan=2, sticky="w", **pad)
        self.cal_apply_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cal_run_frame, text="Apply to config.yaml", variable=self.cal_apply_var).pack(
            side="left",
        )
        self.cal_run_btn = ttk.Button(cal_run_frame, text="Run Calibration", command=self._cal_run)
        self.cal_run_btn.pack(side="left", padx=(12, 0))
        row += 1

        # Internal: label mapping {path: label}
        self._cal_labels: dict[str, str] = {}

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

    # -- Calibration -------------------------------------------------------

    def _cal_add_videos(self):
        paths = filedialog.askopenfilenames(
            title="Select calibration videos",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        for p in paths:
            if p not in self._cal_labels:
                label = self.cal_label_var.get()
                self._cal_labels[p] = label
                self.cal_listbox.insert("end", f"[{label}] {os.path.basename(p)}")

    def _cal_remove_videos(self):
        selected = list(self.cal_listbox.curselection())
        paths = list(self._cal_labels.keys())
        for idx in reversed(selected):
            if idx < len(paths):
                del self._cal_labels[paths[idx]]
            self.cal_listbox.delete(idx)

    def _cal_set_label(self):
        selected = list(self.cal_listbox.curselection())
        paths = list(self._cal_labels.keys())
        new_label = self.cal_label_var.get()
        for idx in selected:
            if idx < len(paths):
                path = paths[idx]
                self._cal_labels[path] = new_label
                self.cal_listbox.delete(idx)
                self.cal_listbox.insert(idx, f"[{new_label}] {os.path.basename(path)}")

    def _cal_run(self):
        if not self._cal_labels:
            return

        cmd = [sys.executable, CALIBRATE_SCRIPT]

        if self.cal_apply_var.get():
            cmd.append("--apply")

        for path, label in self._cal_labels.items():
            cmd.extend(["--label", label, path])

        self.cal_run_btn.configure(state="disabled")
        self.process = subprocess.Popen(cmd, cwd=ROOT_DIR)
        self._poll_calibration()

    def _poll_calibration(self):
        if self.process and self.process.poll() is not None:
            self.cal_run_btn.configure(state="normal")
            self.process = None
        else:
            self.root.after(500, self._poll_calibration)

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
