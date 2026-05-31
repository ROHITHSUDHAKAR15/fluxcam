# Setup & Troubleshooting

FluxCam is a single Python script with three dependencies. From a clean machine to a
running window is about two minutes.

---

## 1. Requirements

| | |
|---|---|
| **Python** | 3.9 – 3.12 (MediaPipe wheels exist for these; 3.13 may not have one yet) |
| **OS** | macOS, Linux, or Windows |
| **Hardware** | Any laptop CPU. No GPU required. A webcam for live mode (not needed for `--selftest`). |
| **Disk** | ~250 MB for the virtualenv (MediaPipe + OpenCV + their deps), plus the bundled MediaPipe models (face ~3.8 MB, hand ~7.8 MB) already in the repo. |

---

## 2. Install

```bash
git clone https://github.com/ROHITHSUDHAKAR15/fluxcam.git
cd fluxcam

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt`:

```
opencv-python>=4.8
numpy>=1.24
mediapipe>=0.10        # hand control; optional — run --no-hands to skip
```

Both MediaPipe models — `models/face_landmarker.task` (~3.8 MB, the AR filters) and
`models/hand_landmarker.task` (~7.8 MB, the particle grab/push) — are committed to the repo,
so there is **nothing else to download**.

---

## 3. Run

```bash
python fluxcam.py                 # clean camera + AR face filter (press n to swap)
python fluxcam.py --filter dog    # start on a specific filter
python fluxcam.py --mode flow     # dense-flow rainbow mode
python fluxcam.py --no-hands      # lighter: no MediaPipe (no filters, no hand control)
python fluxcam.py --input clip.mp4 --particles 12000
python fluxcam.py --selftest      # headless smoke test → writes PNGs, no camera/GUI
```

Focus the window and press **`h`** for the on-screen key list. Press **`n`** to swap face
filters, **`m`** to switch into the motion-art modes.

---

## 4. Verify it works without a camera

`--selftest` drives synthetic motion (a sweeping blob) through the full engine and writes
one PNG per mode. It needs no webcam, no GUI, and exercises every render path — this is the
command CI or a reviewer can run to confirm the install is healthy:

```bash
python fluxcam.py --selftest
```

Expected output:

```
particles -> selftest_particles.png  960x540  lit= 6.22%  mean=6.7
flow      -> selftest_flow.png  960x540  lit= 5.33%  mean=11.3
ink       -> selftest_ink.png  960x540  lit= 7.74%  mean=8.6
selftest OK
```

`lit%` is the fraction of pixels brighter than a threshold — a non-zero value proves the
particles actually responded to motion and got splatted onto the canvas.

---

## 5. Command-line reference

| Flag | Default | Meaning |
|---|---|---|
| `--input` | `0` | Camera index (`0`, `1`, …) or a path to a video file. |
| `--mode` | `photo` | Start mode: `photo`, `particles`, `flow`, or `ink`. |
| `--filter` | `sunglasses` | Starting AR face filter (cycle live with `n`). Static props: `sunglasses`, `mustache`, `dog`, `crown`, `clown`. Interactive: `fire`, `bubbles` (open mouth), `hearts` (smile), `lasers` (raise brows). Plus `none`. |
| `--particles` | `6000` | Particle count (live-tunable with `[` `]`, 500–40000). |
| `--width` / `--height` | `960` / `540` | Output window size in pixels. |
| `--no-mirror` | off | Don't horizontally flip the camera (default mirrors, like a selfie). |
| `--no-hands` | off | Disable MediaPipe entirely (no face filters, no hand control). |
| `--face-model` | `models/face_landmarker.task` | Path to an alternate face-landmark model. |
| `--hand-model` | `models/hand_landmarker.task` | Path to an alternate hand-landmark model. |
| `--selftest` | off | Run headless, write PNGs, exit. |

---

## 6. Troubleshooting

### macOS: "camera permission" / a black window on first run
macOS gates the camera per-app. The **first** run pops a permission request for your
**terminal** (Terminal.app, iTerm, or VS Code). Allow it, then rerun. If you denied it by
accident: *System Settings → Privacy & Security → Camera → enable your terminal.*

### "could not open camera/video"
The default camera index `0` isn't always the built-in one (virtual cameras, OBS, and
Continuity Camera can take it). Try:

```bash
python fluxcam.py --input 1
python fluxcam.py --input 2
```

### MediaPipe install fails with `incomplete-download` / network error
The MediaPipe wheel pulls a large `opencv-contrib-python` dependency; a flaky connection can
truncate it. Resume the download:

```bash
pip install --resume-retries 5 mediapipe
```

If you don't need gesture control at all, you can skip MediaPipe completely and run with
`--no-hands`.

### `ModuleNotFoundError: No module named 'mediapipe.python'` (or `mp.solutions` missing)
Newer MediaPipe builds ship **only** the Tasks API, not the legacy `mp.solutions.hands`.
FluxCam already uses the Tasks API (`HandLandmarker`), so this is expected — there's nothing
to fix. If your own code needs `solutions`, pin an older MediaPipe (`mediapipe==0.10.9`).

### Face filter "isn't showing up"
- It only appears in **photo** mode (the default). If you pressed `m`, you're in an art mode —
  press `m` until the status line reads `photo`, or just restart.
- Get your **whole face** in frame and reasonably well lit; MediaPipe needs to detect the face
  before any prop is drawn.
- Press **`n`** to make sure you're not on the `none` filter (the status line shows the current
  filter name).
- The startup log prints whether face tracking loaded. If it says it's unavailable, MediaPipe or
  `models/face_landmarker.task` is missing — reinstall (`pip install -r requirements.txt`).

### Hand control "isn't doing anything"
- Make sure your **whole hand** is in frame and reasonably well lit.
- The gestures are deliberate: a *pinch* (thumb tip touching index tip) grabs; a flat *open
  palm* pushes. A relaxed or half-closed hand is treated as idle so it doesn't fight your
  body motion.
- Watch for the feedback ring: **green = grab**, **blue = push**. No ring means the gesture
  didn't cross the threshold yet.
- Press **`g`** to confirm hand control is on (the startup log also prints whether it loaded).

### Low frame rate
- Drop the particle count with `[` (down to 500).
- Use `--no-hands` — MediaPipe inference is the single most expensive part of a frame.
- Optical flow is already computed at a small 320 px width; lowering the **window** size
  (`--width/--height`) mostly affects splatting/compositing cost, which is cheap.

### It runs but the screen stays dark
Optical flow measures *apparent motion of texture*. A blank wall, a static scene, or a very
even background barely moves, so there's nothing to paint. **Move** — wave a hand, and wear
or stand in front of something with a bit of pattern. This is expected behaviour, not a bug
(see [CONCEPTS.md](CONCEPTS.md)).

---

## 7. Project layout

```
fluxcam/
├── fluxcam.py                    # the whole app (~500 lines)
├── requirements.txt
├── models/
│   ├── face_landmarker.task      # MediaPipe face mesh — AR filters (bundled, ~3.8 MB)
│   └── hand_landmarker.task      # MediaPipe hand model (bundled, ~7.8 MB)
├── docs/
│   ├── modes.png                 # montage of the three modes
│   ├── SETUP.md                  # this file
│   ├── CONCEPTS.md               # how it works, in depth
│   └── INTERVIEW.md              # presentation & Q&A guide
└── README.md
```
