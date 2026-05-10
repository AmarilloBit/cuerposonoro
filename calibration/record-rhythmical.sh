#!/usr/bin/env bash
# =============================================================================
# record-rhythmical.sh — Calibration recording helper for Cuerpo Sonoro
# (rhythmical mode: percussive pentatonic, all 17 kinematic features mapped).
#
# WHAT IT DOES
#   1. Verifies that BlackHole and switchaudio-osx are installed.
#   2. Switches the macOS system audio output to the multi-output device that
#      sends sound to both your speakers and BlackHole (so the screen recorder
#      can capture it via BlackHole).
#   3. Waits for you to start a macOS Screen Recording (Cmd+Shift+5) with
#      BlackHole 2ch as the microphone source.
#   4. Runs `main.py` against the calibration video in MIDI/rhythmical/debug
#      mode, with --no-loop so it exits after a single pass.
#   5. Restores your previous audio output device on exit.
#
# WHAT YOU MUST DO ONCE, MANUALLY, BEFORE THE FIRST RUN
#   - Create a Multi-Output Device in Audio MIDI Setup named exactly
#       'CuerpoSonoro-Record'
#     containing BlackHole 2ch + your speakers.
#   - In Surge XT: load a percussive patch (mallet, koto, kalimba, plucked
#     synth — sustained pads like 'Bloom' don't fit; the sender emits ~90ms
#     notes and you'd only hear the attack envelope). Set Output =
#     CuerpoSonoro-Record. The patch should listen on channels 2-4 (bass,
#     right melody, left melody); a single patch in omni mode works.
#   - Grant Screen Recording permission to the screenshot tool: System Settings
#     > Privacy & Security > Screen Recording > enable for 'Screenshot' (or
#     'Captura de pantalla'). Required only the first time.
#   - The very first time you run main.py, tick `CuerpoSonoro` under
#     Surge XT > Options > Audio/MIDI Settings > Active MIDI inputs.
#     Surge XT should remember this for subsequent runs.
#
# USAGE
#   ./calibration/record-rhythmical.sh [path/to/video.mp4]
#
#   Defaults to calibration/clasx2.mp4 if no path is given.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VIDEO_PATH="${1:-${REPO_ROOT}/calibration/clasx2.mp4}"
AUDIO_DEVICE_NAME="CuerpoSonoro-Record"
VENV_ACTIVATE="${REPO_ROOT}/.venv/bin/activate"

# -----------------------------------------------------------------------------
# Pretty printing
# -----------------------------------------------------------------------------

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
note()  { printf '  %s\n' "$*"; }
ok()    { printf '\033[32m✓\033[0m %s\n' "$*"; }
warn()  { printf '\033[33m⚠\033[0m  %s\n' "$*"; }
die()   { printf '\033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------

bold "Pre-flight checks"

[[ -f "$VIDEO_PATH" ]] || die "Video not found: $VIDEO_PATH"
ok "Video: $VIDEO_PATH"

[[ -f "$VENV_ACTIVATE" ]] || die "Virtualenv not found at $VENV_ACTIVATE. Create one with 'python -m venv .venv'."
ok "Virtualenv present"

if ! command -v SwitchAudioSource >/dev/null 2>&1; then
    die "Missing 'SwitchAudioSource'. Install with: brew install switchaudio-osx"
fi
ok "SwitchAudioSource installed"

if ! SwitchAudioSource -a -t output | grep -q "BlackHole"; then
    die "BlackHole audio device not found. Install with: brew install blackhole-2ch"
fi
ok "BlackHole present"

if ! SwitchAudioSource -a -t output | grep -q "$AUDIO_DEVICE_NAME"; then
    die "Multi-Output Device '$AUDIO_DEVICE_NAME' not found. Create it in Audio MIDI Setup."
fi
ok "Multi-Output Device '$AUDIO_DEVICE_NAME' present"

# -----------------------------------------------------------------------------
# Audio routing
# -----------------------------------------------------------------------------

PREVIOUS_AUDIO="$(SwitchAudioSource -c -t output)"
note "Current audio output: $PREVIOUS_AUDIO"

cleanup() {
    if [[ -n "${PREVIOUS_AUDIO:-}" ]] && [[ "$PREVIOUS_AUDIO" != "$AUDIO_DEVICE_NAME" ]]; then
        SwitchAudioSource -s "$PREVIOUS_AUDIO" >/dev/null 2>&1 || true
        ok "Audio output restored to: $PREVIOUS_AUDIO"
    fi
}
trap cleanup EXIT INT TERM

SwitchAudioSource -s "$AUDIO_DEVICE_NAME" >/dev/null
ok "Audio output switched to: $AUDIO_DEVICE_NAME"

# -----------------------------------------------------------------------------
# Wait for the operator to start recording
# -----------------------------------------------------------------------------

cat <<EOF

$(bold "Now start a macOS Screen Recording:")
  1. Press Cmd+Shift+5.
  2. Click 'Record Selected Portion' (the dashed-square icon).
  3. Click 'Options' and set:
       - Microphone   → BlackHole 2ch
       - Save to      → wherever you want the .mov to land
       - Show Mouse Clicks → off
  4. Drag a selection over the area where the 'Cuerpo Sonoro' window will
     appear (top-left of your screen by default). Make it a bit larger than
     needed — you can crop later.
  5. Click 'Record'.

When the recording is rolling, come back here and press [Enter].
The Cuerpo Sonoro window will then open inside your selection.
EOF

read -r -p ">> Press Enter when the screen recording is rolling..." _

# -----------------------------------------------------------------------------
# Run the pipeline
# -----------------------------------------------------------------------------

bold "Starting Cuerpo Sonoro..."
note "Video will play once at source FPS, then main.py will exit."
note "If Surge XT does not produce sound after ~2 seconds:"
note "  Surge XT > Options > Audio/MIDI Settings > untick and re-tick 'CuerpoSonoro'."
echo

# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

cd "$REPO_ROOT"
python main.py \
    --source "$VIDEO_PATH" \
    --mode midi \
    --midi-mode rhythmical \
    --backend metal \
    --debug \
    --no-loop

bold "Pipeline finished."
echo "Now stop the screen recording: click the Stop button in the menu bar"
echo "(top-right, square-in-circle icon) or press Cmd+Ctrl+Esc."
echo "The .mov file will be saved to the location you chose in 'Options'."
