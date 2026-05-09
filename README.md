# Cuerpo Sonoro

**Interactive installation that transforms human movement into real-time music.**

Cuerpo Sonoro captures human body movement through computer vision and translates it into musical expression in real time. The system uses AI pose estimation to extract motion features — energy, symmetry, smoothness, joint angles — and maps them to sound synthesis parameters, creating a dialogue between performer and machine.

> Final Degree Project (TFG) · Software Engineering · Universidad Rey Juan Carlos · 2025/2026

**Live Demo:** [cuerposonoro.art](https://cuerposonoro.art)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [Pose Estimation Backends](#pose-estimation-backends)
- [MPE Features & Musical Mapping](#mpe-features--musical-mapping)
  - [Chords (Lower Body)](#chords-lower-body)
  - [Melody (Upper Body)](#melody-upper-body)
  - [Global Controls](#global-controls)
  - [Feature Reference Table](#feature-reference-table)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Installation](#local-installation)
  - [Running the System](#running-the-system)
- [Installation](#installation)
  - [macOS](#macos)
  - [Linux (Ubuntu / Debian)](#linux-ubuntu--debian)
  - [NVIDIA Jetson (Orin Nano)](#nvidia-jetson-orin-nano)
  - [Running modes](#running-modes)
  - [Running tests (no hardware required)](#running-tests-no-hardware-required)
- [Web Demo](#web-demo)
- [Performance Targets](#performance-targets)
- [Testing](#testing)
- [Benchmarks](#benchmarks)
- [Deployment](#deployment)
- [Future Work](#future-work)
- [Academic Context](#academic-context)
- [License](#license)
- [Author](#author)

---

## Overview

Cuerpo Sonoro explores how the human body can become a musical instrument. A camera captures the performer's movements, a pose estimation model detects body landmarks in real time, and a set of algorithms extract meaningful motion features. These features are then mapped to musical parameters and sent via OSC to SuperCollider for audio synthesis, or via MIDI/MPE to external synthesizers like Surge XT.

Two MIDI modes are available: **classic** (hand position selects note, jerk triggers it) and **musical** (tempo-quantized melody navigating chord tones based on movement direction).

The system supports two modes of operation:

- **Local installation** — Full pipeline with SuperCollider and/or MIDI/MPE output, designed for live performance and installation contexts.
- **Web demo** — Browser-based version deployed at [cuerposonoro.art](https://cuerposonoro.art), using MediaPipe.js and the Web Audio API so anyone can try it from their webcam.

### Pipeline

```
Camera → Pose Estimation → Feature Extraction → Mapping → Audio Synthesis
        (backend-dependent)    (Python)         (OSC/MIDI)  (SuperCollider / Surge XT / Web Audio)
```

---

## Architecture

The system follows a merged 2-service Docker architecture, chosen over a 3-service separation to minimize latency (saves 5-15ms of inter-container serialization overhead).

```
┌─────────────────────────────────────────┐
│         Vision Processor                │  ← Docker Service 1
│  ┌─────────────────────────────────┐    │
│  │  Camera Capture (OpenCV)        │    │
│  └──────────────┬──────────────────┘    │
│                 │                       │
│  ┌──────────────▼──────────────────┐    │
│  │  Pose Estimation (backend)      │    │
│  │  CPU / Metal / TensorRT         │    │
│  └──────────────┬──────────────────┘    │
│                 │                       │
│  ┌──────────────▼──────────────────┐    │
│  │  Feature Extraction             │    │
│  └──────────────┬──────────────────┘    │
│                 │                       │
│  ┌──────────────▼──────────────────┐    │
│  │  OSC Sender / MIDI Sender       │    │
│  └─────────────────────────────────┘    │
└──────────────────┬──────────────────────┘
                   │ OSC / MIDI messages
                   ▼
┌─────────────────────────────────────────┐
│         Audio Engine                    │  ← Docker Service 2
│  SuperCollider (scsynth + sclang)       │
└─────────────────────────────────────────┘
```

### Pose Estimation Backends

The pose estimation layer supports multiple hardware backends through a common `BasePoseEstimator` interface. The backend is selected automatically based on the hardware detected at startup, or can be forced with the `--backend` flag.

| Backend | Hardware | Model | Landmarks | Multi-person | GPU |
|---------|----------|-------|-----------|--------------|-----|
| `cpu` | Any machine | MediaPipe Full | 33 (BlazePose) | No | No |
| `metal` | Mac Apple Silicon | MediaPipe Full | 33 (BlazePose) | No | Yes (Metal) |

**Auto-detection priority:** `metal` (Darwin + arm64) → `cpu`.

**Override at runtime:**
```bash
python main.py --backend cpu       # force CPU on any machine
python main.py --backend metal     # force Metal (Mac only)
```

#### GPU backend investigation (Jetson)

The following approaches were attempted to enable GPU inference on the NVIDIA Jetson Orin Nano and all failed:

1. **MediaPipe GPU delegate (pip wheel):** The aarch64 wheel for JetPack 6 is compiled without GPU support. `INFO: Created TensorFlow Lite XNNPACK delegate for CPU` is always printed regardless of the delegate requested.

2. **ONNX Runtime GPU on Jetson:** The `onnxruntime-gpu` package for JetPack 6 / CUDA 12.6 is broken as of March 2026 (known NVIDIA forum issue, no fix available).

3. **TFLite → ONNX conversion with `tflite2onnx`:** MediaPipe models use the `DENSIFY` operator (opcode 124), which `tflite2onnx` does not support. Conversion fails immediately.

4. **TensorRT native TFLite parser:** Removed in TensorRT 8. Not available in TensorRT 10.3.

5. **MediaPipe CPU on Jetson with `jetson_clocks`:** Viable at 55.9ms mean / 56.8ms P95 with clocks fixed. Current production path for the Jetson.

The `tensorrt.py` backend stub remains in the repository for reference but is not active.

---

## MPE Features & Musical Mapping

The system extracts a set of kinematic and postural descriptors from body landmarks and maps them to musical parameters using the MPE (MIDI Polyphonic Expression) paradigm. The body is divided into three zones: lower body controls harmony, upper body controls melody, and head/global motion controls effects.

### Chords (Lower Body)

**1. Chord Selection — Foot position on X axis (`feetCenterX`)**

The feet determine which chord is playing. The camera's capture space is divided into 4 horizontal zones. Walking left or right changes the chord. The midpoint between both ankles is used to calculate position. C major scale with degrees I, IV, V and VI in triads.

**2. Chord Expression — Hip lateral tilt (`hipTilt`)**

The lateral tilt of the hips controls two things: pitch bend (subtle detuning that creates tension) and the addition of extension notes. When the tilt is moderate, pitch bend is applied. When extreme, 6th or 7th notes are added to the chord to enrich it harmonically.

**3. Chord Volume — Knee flexion angle (`kneeAngle`)**

The knee flexion angle controls volume. Extended legs mean maximum volume. Bending the knees progressively reduces volume. This allows natural crescendos and diminuendos with the body.

### Melody (Upper Body)

**4. Right hand melodic note — Right hand Y height (`rightHandY`)**

The vertical height of the right hand in space determines which note plays in the lower octave (C3 to B3). Hand down plays low notes, hand up plays high notes within that octave.

**5. Left hand melodic note — Left hand Y height (`leftHandY`)**

Same as the right hand but in the upper octave (C5 to B5). This enables playing two-voice melodies separated by two octaves.

**6. Note trigger — Sudden hand/wrist movement (`rightHandJerk`, `leftHandJerk`)**

Notes do not sound continuously. They are triggered only when a sudden movement (high jerk) is detected in the hand or wrist. This gives percussive, expressive control over when notes sound.

**7. Note intensity and duration — Arm speed (`rightArmVelocity`, `leftArmVelocity`)**

The speed at which the arm moves during a gesture determines two things: MIDI velocity (how loud the note sounds) and note duration. Fast movement produces loud, short notes (staccato). Slow movement produces soft, longer notes.

**8. Glissando and vibrato — Elbow-to-hip angle (`rightElbowHipAngle`, `leftElbowHipAngle`)**

The angle formed by the arm relative to the torso (measured between elbow and hip) controls pitch bend on melodic notes. Arm close to the body means a stable note. Extended arm applies glissando. Oscillating elbow movement generates vibrato.

### Global Controls

**9. Global frequency filter — Head tilt (`headTilt`)**

The lateral tilt of the head controls a global frequency filter that affects all sound. Head straight is neutral sound. Tilting to one side darkens the sound, tilting to the other makes it brighter.

**10. Textures and drones — Global energy (`energy`)**

The overall motion energy of the body is sent to SuperCollider to control background textures and drones that complement the melodic and harmonic MPE output.

### Feature Reference Table

| Feature | Landmarks Used | Output Range |
|---------|---------------|-------------|
| `feetCenterX` | Ankles (27, 28) | 0.0 – 1.0 |
| `hipTilt` | Hips (23, 24) | -1.0 – 1.0 |
| `kneeAngle` | Hip, knee, ankle (23/24, 25/26, 27/28) | 0.0 – 1.0 |
| `rightHandY` | Right wrist (16) | 0.0 – 1.0 |
| `leftHandY` | Left wrist (15) | 0.0 – 1.0 |
| `rightHandJerk` | Right wrist (16) velocity | 0.0 – 1.0 |
| `leftHandJerk` | Left wrist (15) velocity | 0.0 – 1.0 |
| `rightArmVelocity` | Right wrist (16) | 0.0 – 1.0 |
| `leftArmVelocity` | Left wrist (15) | 0.0 – 1.0 |
| `rightElbowHipAngle` | Shoulder, elbow, hip (12, 14, 24) | 0.0 – 1.0 |
| `leftElbowHipAngle` | Shoulder, elbow, hip (11, 13, 23) | 0.0 – 1.0 |
| `headTilt` | Ears (7, 8) | -1.0 – 1.0 |

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Pose Estimation (Mac, GPU) | [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html) + Metal | Real-time body landmark detection (33 points, GPU) |
| Pose Estimation (fallback) | [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html) CPU | Real-time body landmark detection (33 points, CPU) |
| Audio Synthesis | [SuperCollider](https://supercollider.github.io/) | Algorithmic sound synthesis via OSC |
| MIDI/MPE Output | [python-rtmidi](https://github.com/SpotlightKid/python-rtmidi) | MIDI Polyphonic Expression for external synths |
| External Synth | [Surge XT](https://surge-synthesizer.github.io/) | Open-source MPE-compatible synthesizer |
| Video Capture | [OpenCV](https://opencv.org/) | Camera access and frame processing |
| Communication | [python-osc](https://github.com/attwad/python-osc) | OSC protocol for real-time control |
| Web Frontend | MediaPipe.js + Web Audio API | Browser-based pose detection and synthesis |
| Web Backend | [FastAPI](https://fastapi.tiangolo.com/) + WebSocket | Real-time feature extraction server |
| Containerization | [Docker](https://www.docker.com/) + Docker Compose | Portable, reproducible deployment |
| Infrastructure | Hetzner VPS + Nginx + Let's Encrypt | Cloud deployment with HTTPS |
| Language | Python 3.10+ | Core application logic |
| Dependencies | pip-tools | Reproducible dependency management |

---

## Project Structure

```
cuerposonoro/
├── vision_processor/           # Perception + feature extraction module
│   ├── __init__.py
│   ├── capture.py              # Camera abstraction layer
│   ├── pose.py                 # BasePoseEstimator interface + PoseEstimator() factory
│   ├── backends/               # Hardware-specific pose estimation backends
│   │   ├── __init__.py
│   │   ├── cpu.py              # CPUPoseEstimator — MediaPipe on CPU (any machine)
│   │   ├── metal.py            # MetalPoseEstimator — MediaPipe + Metal GPU (Mac Apple Silicon)
│   │   └── tensorrt.py         # TensorRTPoseEstimator — stub (GPU path not viable, see docs)
│   ├── features.py             # Motion feature extraction (19 features)
│   ├── osc_sender.py           # OSC communication to SuperCollider
│   ├── midi/                   # MIDI sender strategy pattern
│   │   ├── base.py             # BaseMidiSender abstract interface
│   │   ├── classic.py          # Classic mode: hand Y position → note, jerk triggers
│   │   └── musical.py          # Musical mode: tempo-quantized, chord-tone navigation
│   ├── midi_sender.py          # Backward-compatible alias → ClassicMidiSender
│   ├── config.py               # Centralized config loader with factory methods
│   └── latency_logger.py       # Per-stage latency instrumentation
├── audio_engine/               # SuperCollider audio synthesis
│   └── ...
├── supercollider/              # SuperCollider SynthDef files
│   └── ...
├── tests/
│   ├── unit/                   # Automated unit tests (pytest)
│   │   ├── test_features.py
│   │   └── test_backends.py    # CPU & Metal backends, BasePoseEstimator, _detect_backend
│   ├── integration/            # Automated integration tests (pytest)
│   │   └── test_integration.py
│   └── manual/                 # Interactive scripts (require hardware)
│       ├── README.md
│       ├── manual_camera.py
│       ├── manual_pose.py
│       ├── manual_osc.py
│       ├── manual_midi_sender.py
│       ├── manual_e2e_osc.py
│       └── manual_e2e_midi_debug.py
├── benchmarks/                 # Latency benchmarking system
│   ├── README.md               # Full documentation with results
│   ├── run_benchmark.py        # Automated benchmark runner
│   ├── analyze_results.py      # Analysis with pandas + matplotlib
│   ├── results/                # Session-organized CSV data (gitignored)
│   └── charts/                 # Generated charts and screenshots (gitignored)
├── logs/                       # Session logs (CSV) for analysis
├── main.py                     # Application entry point
├── config.yaml                 # Centralized configuration
├── requirements.in             # Top-level dependencies
├── requirements.txt            # Pinned dependencies (pip-tools)
├── LICENSE
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **SuperCollider** — [Download](https://supercollider.github.io/downloads)
- **Webcam** (Logitech C922 recommended, any USB webcam works)
- **Docker** (optional, for containerized deployment)

For MIDI/MPE mode:
- **Surge XT** — [Download](https://surge-synthesizer.github.io/) (free, open-source, GPL3)

### Local Installation

1. **Clone the repository:**

```bash
git clone https://github.com/maramotto/cuerposonoro.git
cd cuerposonoro
```

2. **Create a virtual environment and install dependencies:**

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
```

3. **Configure the system:**

Edit `config.yaml` to adjust camera settings, OSC ports, feature parameters, and more.

### Running the System

**With SuperCollider (OSC mode):**

```bash
python main.py
```

**With Surge XT (MIDI/MPE mode):**

```bash
python main.py --mode midi
```

**CLI flags:**

| Flag | Description |
|------|-------------|
| *(none)* | Live webcam, output mode from `config.yaml` |
| `--backend cpu\|metal` | Override pose estimation backend |
| `--source PATH` | Use a video file instead of webcam |
| `--debug` | Show feature values and skeleton overlay |
| `--mode osc\|midi` | Override output mode |
| `--midi-mode classic\|musical` | Override MIDI mode |

---

## Installation

### macOS

```bash
git clone https://github.com/maramotto/cuerposonoro.git
cd cuerposonoro
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The Metal backend is selected automatically on Apple Silicon. On Intel Macs, CPU is used.

> **Camera permissions:** macOS requires explicit camera access. On first run a system dialog will appear. If you accidentally denied it: System Settings → Privacy & Security → Camera → enable your terminal.

---

### Linux (Ubuntu / Debian)

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip libportaudio2
git clone https://github.com/maramotto/cuerposonoro.git
cd cuerposonoro
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### NVIDIA Jetson (Orin Nano)

MediaPipe GPU inference is not viable on the Jetson via pip (see [GPU backend investigation](#gpu-backend-investigation-jetson)). The Jetson runs the CPU backend with clocks fixed for stable latency.

```bash
sudo jetson_clocks
git clone https://github.com/maramotto/cuerposonoro.git
cd cuerposonoro
pip3 install -r requirements.txt
python3 main.py --backend cpu --debug
```

---

### Running modes

| Mode | Synthesizer | Command |
|------|------------|---------|
| OSC (default) | SuperCollider | `python main.py` |
| MIDI classic | Surge XT | `python main.py --mode midi --midi-mode classic` |
| MIDI musical | Surge XT | `python main.py --mode midi --midi-mode musical` |
| Video file (debug) | any | `python main.py --source path/to/video.mp4 --debug` |

---

### Running tests (no hardware required)

```bash
pytest tests/unit/ tests/integration/ -v
```

All tests run without a camera or synthesizer connected.

---

## Web Demo

A browser-based version is deployed at **[cuerposonoro.art](https://cuerposonoro.art)**, allowing anyone with a webcam to experience the installation without any software installation.

```
┌─────────────────────────────────────────────────┐
│                   Browser                       │
│  Camera → MediaPipe.js → WebSocket → Web Audio  │
└────────────────────────┬────────────────────────┘
                         │ landmarks (WebSocket)
                         ▼
┌─────────────────────────────────────────────────┐
│              Server (Docker)                    │
│  Nginx (static + reverse proxy)                 │
│  FastAPI (feature extraction via WebSocket)     │
└─────────────────────────────────────────────────┘
```

---

## Testing

```bash
pytest tests/ -v                          # all automated (168 tests)
pytest tests/unit/ -v                     # unit tests only (119 tests)
pytest tests/integration/ -v              # integration tests only (~50 tests)
```

Manual tests (require hardware):

```bash
python tests/manual/manual_camera.py
python tests/manual/manual_pose.py
python tests/manual/manual_e2e_osc.py
python tests/manual/manual_e2e_midi_debug.py
```

See [`tests/manual/README.md`](tests/manual/README.md) for prerequisites and usage.

---

## Benchmarks

The project includes a dedicated benchmarking system that measures end-to-end latency across every pipeline stage (capture, pose estimation, feature extraction, send). All benchmarks use 300 frames per configuration.

### Hardware Tested

| Machine | CPU | GPU | Backend | Camera |
|---------|-----|-----|---------|--------|
| MacBook Air M4 (2025) | Apple M4 | Apple M4 GPU (Metal) | `metal` | Built-in + Logitech C922 |
| MacBook Pro (2020) | Intel Core i7-1068NG7 | Intel Iris Plus (not used) | `cpu` | Logitech C922 |

### Results: Metal (MacBook Air M4)

Metal GPU acceleration makes model complexity irrelevant for latency. All three model sizes perform identically.

| Resolution | Pose Model | Pose Mean | Total Mean | P95 | FPS | Under 80ms |
|------------|------------|-----------|------------|-----|-----|------------|
| 640x480 | Lite (0) | 9.3ms | 33.3ms | 36.2ms | 30.3 | 100% |
| 640x480 | Full (1) | 10.4ms | 33.3ms | 36.2ms | 30.3 | 100% |
| 640x480 | Heavy (2) | 10.0ms | 33.4ms | 36.6ms | 30.4 | 99.9% |
| 1280x720 | Lite (0) | 10.4ms | 33.3ms | 36.8ms | 30.4 | 100% |
| 1280x720 | Full (1) | 10.5ms | 33.3ms | 36.6ms | 30.4 | 100% |
| 1280x720 | Heavy (2) | 10.6ms | 33.3ms | 36.4ms | 30.4 | 100% |

*Values averaged across OSC, OSC bundle, and MIDI output modes with the Logitech C922. Built-in camera results are equivalent (~11ms pose mean).*

### Results: CPU (MacBook Pro 2020 Intel i7)

CPU backend handles Lite and Full models at 30 FPS. The Heavy model exceeds the 80ms latency budget.

| Resolution | Pose Model | Pose Mean | Total Mean | P95 | FPS | Under 80ms |
|------------|------------|-----------|------------|-----|-----|------------|
| 640x480 | Lite (0) | 15.5ms | 33.3ms | 39.1ms | 30.5 | 100% |
| 640x480 | Full (1) | 19.4ms | 33.4ms | 38.8ms | 30.5 | 99.9% |
| 640x480 | Heavy (2) | 60.8ms | 69.6ms | 98.9ms | 14.6 | 89.5% |
| 1280x720 | Lite (0) | 15.2ms | 33.5ms | 39.4ms | 30.4 | 99.9% |
| 1280x720 | Full (1) | 19.9ms | 33.4ms | 37.5ms | 30.4 | 99.9% |
| 1280x720 | Heavy (2) | 68.2ms | 93.4ms | 113.2ms | 11.1 | 20.7% |

### Results: Jetson Orin Nano (MediaPipe CPU)

| Configuration | Mean | P95 | Max |
|---------------|------|-----|-----|
| Without `jetson_clocks` | 89.0ms | 98.9ms | 99.4ms |
| With `jetson_clocks` | 55.9ms | 56.8ms | 57.1ms |

### Key Findings

1. **Metal eliminates model complexity cost.** Pose inference stays at ~10ms for all models (Lite/Full/Heavy), meaning users can select the Heavy model for maximum accuracy with no latency penalty.
2. **CPU is viable for Lite and Full.** Both achieve 30 FPS with 100% of frames under 80ms. Heavy is not recommended for real-time use on CPU.
3. **Resolution has minimal impact.** 720p adds negligible latency on both backends — the camera's 30 FPS cap is the bottleneck, not pose estimation.
4. **Output mode is negligible.** OSC, OSC bundle, and MIDI all add <1ms and do not affect the overall latency profile.
5. **Camera choice is neutral.** Built-in and Logitech C922 perform equivalently on Metal; capture latency differences are within noise.

### Running Benchmarks

```bash
python benchmarks/run_benchmark.py --preview --session-name my-session
python benchmarks/analyze_results.py --save
```

For full methodology, see [`benchmarks/README.md`](benchmarks/README.md). Raw data is in [`benchmarks/benchmark_results_tfg/`](benchmarks/benchmark_results_tfg/).

---

## Deployment

### Docker (local)

```bash
docker compose up --build
```

### Cloud (Hetzner VPS)

```bash
ssh user@your-server
cd ~/cuerposonoro-webdemo
docker compose up -d --build
```

---

## Academic Context

This project is a Final Degree Project (Trabajo de Fin de Grado, TFG) for the **Software Engineering degree** at **Universidad Rey Juan Carlos (URJC)**, Madrid, Spain.

### Evaluation Criteria

- **Quantitative:** FPS, end-to-end latency, pose detection robustness.
- **Qualitative:** User testing with 3-5 participants measuring perceived responsiveness, expressiveness, intuitiveness, engagement, and sense of control (Likert scale questionnaire).

### Ethical Considerations

- Only pose landmark coordinates are processed — no images or video are stored.
- GDPR-compliant: no personal or identifiable data is retained.
- User testing participants sign informed consent forms.

---

## License

- **Code:** [Unlicense](LICENSE)
- **Documentation:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

---

## Author

**Ana María J Crespo**

- Web: [maramotto/cuerposonoro](https://maramotto.com/cuerposonoro.html)
- Github: [@maramotto](https://github.com/maramotto)
- Email: am.juradoc@alumnos.urjc.es
- University: ETSII, Universidad Rey Juan Carlos
