# 🌈 FluxCam — AR face filters + motion art

Point your webcam at yourself. By default you get a clean, sharp camera with a **Snapchat-style
AR face filter** locked onto your face — sunglasses, a mustache, dog ears, a crown, a clown nose
— tracked live with MediaPipe's 468-point face mesh. Press **`n`** to swap filters; they rotate,
scale, and follow your head in real time.

Press **`m`** and the same webcam turns into **generative motion art**: the program measures how
the picture is *flowing* every frame and sweeps thousands of glowing particles along with your
movement, leaving neon trails. **Reach in and touch it** — pinch to *grab and fling* the
particles, open palm to *push* them away. Live hand tracking, no controller.

It's a real-time, interactive computer-vision toy in ~500 lines of Python.

![three modes](docs/modes.png)
<!-- run `python fluxcam.py --selftest` to regenerate sample frames -->

**Docs:** [Setup & troubleshooting](docs/SETUP.md) · [How it works (concepts)](docs/CONCEPTS.md) · [Interview / presentation guide](docs/INTERVIEW.md)

---

## How it works (the ideas doing the work)

1. **AR face filters** — MediaPipe's `FaceLandmarker` returns **468 face landmarks** per frame. We
   derive a small *face frame* from a couple of them (centre between the eyes, scale = eye-corner
   distance, roll = angle of the eye line) and draw props anchored to it — sunglasses over the eyes,
   a mustache under the nose, ears above the forehead. Because everything is positioned in that
   frame, the props **track head tilt, distance, and movement** automatically. The props are plain
   OpenCV shapes (no image assets), so the repo stays tiny and there's nothing to download.
2. **Interactive (expression-driven) filters** — from the same landmarks we read a few continuous
   signals: **mouth-open** (inner-lip gap), **smile** (mouth width), and **eyebrow-raise** (brow-to-
   eye gap), all normalised by face size so they work near or far. Those drive a tiny particle
   emitter: open your mouth and *fire* or *bubbles* stream from it, smile and *hearts* float up with
   heart-eyes, raise your brows and *laser eyes* fire — intensity scales with how big the expression
   is. Same vectorised, no-asset discipline; the glow is an additive blurred overlay.
3. **Dense optical flow** — `cv2.calcOpticalFlowFarneback` compares two consecutive frames and
   returns, for *every pixel*, how far and which way it moved. That velocity field **is** your
   motion, captured as data.
4. **A particle system rides the flow** — ~6,000 particles sample the field at their position and
   get advected along it, so they literally stream with your hand/body. They're coloured by the
   **direction** they travel (angle → hue), so a wave of your hand paints an arc of rainbow.
5. **A fading trail buffer** — each frame the canvas is dimmed by a decay factor and new particle
   splats are added on top. That's what turns flickering dots into smooth glowing trails.
6. **Hand control** — MediaPipe's `HandLandmarker` returns 21 landmarks per hand. From them we read
   two gestures: thumb–index distance (a **pinch**) and finger spread (an **open palm**). A pinch
   becomes an attractor that grabs nearby particles and flings them in the direction your hand is
   moving; an open palm becomes a repeller that shoves them away. The forces are applied as a single
   vectorized displacement over all particles — same no-Python-loop discipline as the rest.

The whole thing is **vectorized in NumPy + OpenCV** — no per-particle Python loop. Optical flow is
computed at a small 320-px width and the particles are splatted with `np.add.at`, so it comfortably
hits real-time framerates on a laptop CPU — no GPU. The two MediaPipe models (face ≈3.8 MB, hand
≈7.8 MB) are small and bundled; all MediaPipe tracking is optional (`--no-hands`).

```
webcam ─► resize 320px ─► Farneback flow ─┬─► advect particles ─► colour by direction
   │                                      │                          │
   └─► MediaPipe hands ─► grab / push ────┤        additive splat ◄──┤
                          fade trail ◄────┘                          └─► glow ─► window
```

## Modes (cycle with `m`)

| Mode | What you see |
|---|---|
| **photo** *(default)* | A clean, clear live camera with an **AR face filter** locked onto your face. Press **`n`** to cycle. *Static props* track your head: sunglasses · mustache · dog ears · crown · clown nose. *Interactive filters* react to your face: **fire** (open your mouth to breathe flames) · **bubbles** (open mouth to blow bubbles) · **hearts** (smile for floating hearts + heart-eyes) · **lasers** (raise your eyebrows to fire laser eyes). |
| **particles** | Glowing particles swept by your motion, trailing light. The signature art look. **Pinch** to grab/fling, open palm to push. |
| **flow** | The raw motion field as colour — direction → hue, speed → brightness. Your movement *is* the rainbow. |
| **ink** | Like particles, but the trail is blurred each frame so colour bleeds and diffuses like dye in water. |

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python fluxcam.py                     # clean camera + AR face filter (press n to swap)
python fluxcam.py --filter fire       # start on an interactive filter (open your mouth!)
python fluxcam.py --filter dog        # ...or a static prop
python fluxcam.py --mode particles    # the glowing motion-particle art
python fluxcam.py --mode flow         # dense-flow rainbow mode
python fluxcam.py --input clip.mp4 --particles 12000
python fluxcam.py --no-hands          # skip MediaPipe (lighter, optical flow only)
python fluxcam.py --selftest          # headless: render synthetic frames to PNGs (no camera/GUI)
```

> **macOS:** the first run will ask for camera permission for your terminal app. Allow it, then
> rerun. If `--input 0` can't open, try `--input 1`.

## Controls (window focused)

| Key | Action | Key | Action |
|---|---|---|---|
| `m` | cycle mode | `n` | next face filter (photo mode) |
| `c` | particle colour (direction / camera / ember) | `[` `]` | fewer / more particles |
| `x` | toggle faint camera "ghost" | `-` `=` | shorter / longer trails |
| `g` | toggle MediaPipe tracking | `f` | mirror |
| `r` | clear the trail | `s` | save a PNG |
| `space` | pause | `h` | toggle help · `q`/`Esc` quit |

**Face filters** (photo mode): press **`n`** to cycle. *Static props* follow your face — tilt your
head and the sunglasses tilt with you, lean in and they scale up. *Interactive filters* react to
your expression and the on-screen status line tells you the cue: **open your mouth** for fire or
bubbles, **smile** for hearts (you also get heart-eyes), **raise your eyebrows** to fire laser
eyes — the bigger the expression, the stronger the effect. Press **`s`** to save a clean PNG with
the filter baked in. **Hand gestures** (art modes): **pinch** thumb to index to grab and *fling*
the particles in the direction you sweep; **hold an open palm** to push them away. A green ring
means grab, blue means push.

## Design notes & honest trade-offs

- **CPU-only by design.** Farneback dense flow at 320-px width is the sweet spot — full-resolution
  flow would look marginally crisper but tank the framerate. The particle math then runs in
  flow-space and is scaled up to the display, so particle count is decoupled from flow cost.
- **`np.add.at` for the splat.** It's the clean vectorized way to accumulate many particles into one
  canvas. For *huge* counts you'd move the splat to a shader (OpenGL/moderngl) and easily do millions
  — that's the natural next step if you want it GPU-fast.
- **Optical flow ≠ object tracking.** It measures apparent motion of texture, so a plain wall barely
  moves (nothing to track) while a patterned shirt lights up. That's expected, and part of the charm.
- **Direction-coloured particles** make motion *legible*: you can see at a glance which way each part
  of the scene is moving. Switch to `camera` colour to paint with your actual colours instead.
- **Gestures, not classification.** Hand control reads two cheap geometric features from the
  landmarks (thumb–index distance, finger spread) rather than a gesture classifier. It's robust,
  has zero training, and the thresholds are easy to reason about — but it only knows *pinch* and
  *open palm*, by design.

## Possible extensions
GPU splatting (moderngl) for millions of particles · two-handed pinch to stretch/rotate the field ·
audio-reactive brightness · record to video (`cv2.VideoWriter`) · attractors you place with the mouse.

---

**Stack:** Python 3 · OpenCV · NumPy · MediaPipe

### Credits / inspiration
Built after studying common OpenCV optical-flow and webcam creative-coding patterns:
- [OpenCV optical-flow tutorial](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html)
- [LearnOpenCV — Optical Flow](https://learnopencv.com/optical-flow-in-opencv/)
- [daisukelab/cv_opt_flow](https://github.com/daisukelab/cv_opt_flow) (dense-flow HSV showcase)
- [RomalaMishra/Air_Canvas](https://github.com/RomalaMishra/Air_Canvas) (webcam interactive-art pattern)
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (21-point hand tracking)
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (468-point face mesh, used for the AR filters)
