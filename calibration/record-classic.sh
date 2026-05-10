#!/usr/bin/env bash
# =============================================================================
# record-classic.sh — Calibration recording helper for Cuerpo Sonoro.
#
# WHAT IT DOES
#   1. Verifies that BlackHole and switchaudio-osx are installed.
#   2. Switches the macOS system audio output to the multi-output device that
#      sends sound to both your speakers and BlackHole (so OBS can capture it).
#   3. Opens OBS Studio if it isn't already running.
#   4. Waits for you to confirm that OBS is recording.
#   5. Runs `main.py` against the calibration video in MIDI/classic/debug mode,
#      with --no-loop so it exits after a single pass.
#   6. Restores your previous audio output device on exit.
#
# WHAT YOU MUST DO ONCE, MANUALLY, BEFORE THE FIRST RUN
#   - Create a Multi-Output Device in Audio MIDI Setup named exactly
#       'CuerpoSonoro-Record'
#     containing BlackHole 2ch + your speakers.
#   - In Surge XT: enable MPE, set Output = CuerpoSonoro-Record, load an MPE
#     patch (e.g. 'Bloom').
#   - In OBS: create a scene with two sources:
#       (a) Window Capture pointing at the OpenCV window 'Cuerpo Sonoro'
#       (b) Audio Input Capture using device 'BlackHole 2ch'
#   - The very first time you run main.py, tick `CuerpoSonoro` under
#     Surge XT > Options > Audio/MIDI Settings > Active MIDI inputs.
#     Surge XT should remember this for subsequent runs.
#
# USAGE
#   ./calibration/record-classic.sh [path/to/video.mp4]
#
#   Defaults to calibration/calibration-video.mp4 if no path is given.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VIDEO_PATH="${1:-${REPO_ROOT}/calibration/calibration-video.mp4}"
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

if ! pgrep -x "OBS" >/dev/null 2>&1; then
    note "OBS not running, opening it..."
    open -a OBS
fi
ok "OBS launched (or already running)"

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

$(bold "Now do this in OBS:")
  1. Bring OBS to the front.
  2. Make sure your scene with the 'Cuerpo Sonoro' Window Capture and the
     BlackHole 2ch audio source is selected.
  3. Click 'Start Recording'.

When OBS shows the red REC indicator, come back here and press [Enter].
EOF

read -r -p ">> Press Enter when OBS is recording..." _

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
    --midi-mode classic \
    --debug \
    --no-loop

bold "Pipeline finished."
echo "Now stop the recording in OBS (Cmd+Shift+R or the 'Stop Recording' button)."
echo "The recorded file will be in your OBS recordings folder."
