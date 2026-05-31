# Concepts — how FluxCam actually works

FluxCam has two halves that share one webcam pipeline:

* **photo mode (default)** — a clean camera with a **Snapchat-style AR face filter** tracked by
  MediaPipe's 468-point face mesh (section 6 below).
* **art modes (`m`)** — your motion turned into glowing particle art, driven by dense optical
  flow, with MediaPipe **hand** tracking to grab and push the particles (sections 1–5, 7).

Nothing is pre-recorded: every frame is computed from the two most recent camera frames plus
(optionally) MediaPipe's face/hand landmarks. This doc explains each idea, why it was chosen, and
where it lives in the code.

The whole pipeline:

```
              ┌─────────────── per frame ───────────────┐
 webcam ─► resize to 320px ─► Farneback dense flow ─┐
   │                                                 │
   │                              ┌──────────────────▼─────────────────┐
   └─► MediaPipe HandLandmarker ─►│ advect ~6k particles by the flow   │
                                  │ + apply hand grab / push forces    │
                                  └──────────────────┬─────────────────┘
                                                     │ colour by travel direction
                                  fade the trail ◄───┤
                                  buffer (×decay)    │ additive splat (np.add.at)
                                                     ▼
                                             glow + compose ─► window
```

---

## 1. Dense optical flow — capturing motion as data

**The idea.** Optical flow answers, for the image as a whole, *"which way and how far did each
point move between these two frames?"* **Dense** flow answers it for **every pixel**, producing
a 2-channel field `flow[y, x] = (dx, dy)`. That field *is* your motion, expressed as numbers we
can push particles around with.

**In the code** (`Engine.step`):

```python
flow = cv2.calcOpticalFlowFarneback(
    self.prev_gray, gray, None,
    pyr_scale=0.5, levels=3, winsize=15, iterations=3,
    poly_n=5, poly_sigma=1.2, flags=0)
```

Gunnar Farnebäck's algorithm fits a local quadratic polynomial to the image brightness around
each pixel in both frames and solves for the displacement that maps one to the other. The
parameters trade speed for smoothness:

| Param | Value | Effect |
|---|---|---|
| `pyr_scale=0.5`, `levels=3` | image pyramid, each level half-size | lets it catch **large** motions cheaply by solving coarse-to-fine |
| `winsize=15` | averaging window | bigger = smoother but blurrier flow; 15 is a good real-time middle |
| `iterations=3` | refinement passes per level | more = more accurate, slower |
| `poly_n=5`, `poly_sigma=1.2` | polynomial neighbourhood | the basis used for the local fit |

**Why dense (not sparse) flow?** Sparse methods (Lucas–Kanade) track a handful of *corner*
points — great for "follow this object", useless for "paint the whole moving field". We want
the second thing.

**Why 320 px wide?** Flow cost scales with pixel count, and it's the most expensive op in the
frame. At `FLOW_W = 320` it's fast enough for real time on a CPU; the particles then live in
this small flow-space and are *scaled up* to the display, so visual resolution is decoupled
from flow cost. Full-res flow would look marginally crisper and tank the frame rate.

**The honest limitation.** Optical flow measures *apparent motion of texture*, not objects. A
blank wall barely moves (nothing to lock onto); a patterned shirt lights up. That's inherent to
the method and part of the look.

---

## 2. A particle system that rides the flow

**The idea.** Scatter ~6,000 particles over the flow field. Each frame, every particle reads the
flow vector *at its own location* and steps along it — so the particles literally stream with
your motion.

**In the code** (`Particles.update`):

```python
xi = np.clip(self.pos[:, 0].astype(np.int32), 0, fw - 1)
yi = np.clip(self.pos[:, 1].astype(np.int32), 0, fh - 1)
vel = flow[yi, xi]            # the flow vector under each particle
self.pos += vel * speed       # advect
self.pos[:, 1] += 0.15        # a touch of ambient downward drift
```

This is **advection** — moving a particle through a velocity field, the same idea fluid sims use
for smoke and dye.

**Lifecycle.** Each particle has an `age` and a random `life`. A particle is respawned at a fresh
random position when it ages out *or* leaves the frame. This keeps the field full and
continuously refreshed, so trails don't permanently "burn in":

```python
self.age += dt / self.life
oob = (out of bounds) | (self.age >= 1.0)
self.pos[oob] = self._rand_pos(oob.sum())
```

**Everything is vectorized.** `pos`, `age`, and `life` are parallel NumPy arrays of shape
`(n, …)`. There is **no per-particle Python loop** anywhere — advection, ageing, respawn, and
colouring are all whole-array operations. That single decision is what makes 6,000 (or 40,000)
particles real-time in pure Python.

---

## 3. Colour as a readout of motion

Particles are tinted so the *image tells you something* about the motion, not just "pretty
lights" (`colors_for`). Three schemes (`c` cycles them):

- **direction** — the angle the particle is travelling maps to hue: `arctan2(dy, dx) → hue`.
  Now you can *see* which way every part of the scene is moving; a wave of the hand paints an arc
  of rainbow. This is the signature look.
- **camera** — sample the actual webcam colour under each particle, so you paint with your real
  colours.
- **ember** — a fixed warm orange, scaled by speed, for a fire/sparks feel.

In every scheme, **brightness encodes speed**: `bright = clip(speed*0.5 + 0.25, 0.25, 1.6)`.
Fast motion glows; slow motion stays dim. Colour and brightness together make the visual a
genuine *visualisation* of the flow, not just decoration.

---

## 4. The fading trail buffer — why it glows

**The idea.** Without memory, particles are flickering dots. The trail buffer gives them a
long-exposure glow.

**In the code** (`Engine.step`):

```python
self.trail *= cfg.decay          # dim last frame's image a little
splat(self.trail, pos, colors)   # add this frame's particles on top
```

`trail` is a `float32` canvas that **persists across frames**. Each frame we multiply it by a
`decay` factor (default `0.86`) and then *add* the new particles. A bright splat therefore fades
over many frames instead of vanishing — that exponential decay is the visible "tail" of light.
`decay` is the trail-length knob (`-`/`=`): higher = longer, dreamier trails.

**Additive splatting** (`splat`):

```python
np.add.at(canvas, (ys[m], xs[m]), colors[m])
```

`np.add.at` accumulates many particles into the same canvas pixels **correctly** — if two
particles land on one pixel, both contribute (a plain `canvas[ys, xs] = colors` would let one
silently overwrite the other). Additive blending is also why overlapping trails bloom into
bright white hotspots, like real light.

**Float canvas, clamp at the end.** Accumulating in `float32` lets values exceed 255 (that's the
"over-bright" glow); we only clip to `uint8` at display time (`compose`).

---

## 5. The three modes

`m` cycles them; all three share the pipeline above.

| Mode | What changes | Look |
|---|---|---|
| **particles** | the default path | glowing particles swept by motion, trailing light |
| **flow** | skip particles; render the raw flow field as HSV (direction→hue, magnitude→brightness) | your movement *is* the rainbow — a direct view of the data |
| **ink** | blur the trail buffer a little each frame + boost splat brightness | colour bleeds and diffuses like dye dropped in water |

`flow` mode (`render_flow`) is the "show me the raw signal" view: it's literally the optical-flow
field colour-coded, blended over a dim camera image. `ink` mode adds one line —
`cv2.GaussianBlur(self.trail, (0,0), 1.1)` — so each frame's light spreads slightly before the
next splat, giving the diffusing-dye effect.

---

## 6. AR face filters — the photo-mode headline

**The idea.** In photo mode the camera stays clean and sharp, and we lock a Snapchat-style prop
onto the face — sunglasses, a mustache, dog ears, a crown, a clown nose. `n` cycles them.

**Tracking** (`FaceTracker`). We use MediaPipe's **FaceLandmarker** (the Tasks API), which returns
**468 face landmarks** per face in normalized `[0,1]` image coordinates. The model is a small
bundled `.task` file; inference runs on CPU in VIDEO mode (temporal tracking, so each call needs a
monotonically increasing timestamp — exactly like the hand tracker).

**A face frame, not a per-landmark hack.** Drawing a prop "at landmark 5" would ignore head tilt
and distance. Instead we derive a tiny coordinate frame from two stable landmarks (the outer eye
corners, 33 and 263) and express every prop in it (`_face_frame`):

```python
eL, eR = landmark[33], landmark[263]      # outer eye corners, in pixels
centre = (eL + eR) / 2                     # midpoint between the eyes
scale  = ‖eR − eL‖                         # eye-corner distance = a natural unit
angle  = atan2((eR − eL).y, (eR − eL).x)   # head roll
ux = (cos angle, sin angle)                # unit vector along the eye line
uy = (−sin angle, cos angle)               # unit vector down the face
```

Every prop is then placed as `anchor + (a·ux + b·uy)·scale` for some landmark anchor and offsets
`a, b` *in eye-distance units*. Because the offsets are scaled by `scale` and rotated by `angle`,
the props **automatically follow head tilt and distance** — lean in and the sunglasses grow; tilt
your head and they tilt with you. Ellipse props are drawn with `angle` as their rotation so they
bank correctly too.

**Props are pure OpenCV, no assets.** Each filter is a handful of `cv2.ellipse` / `cv2.fillPoly`
calls (`_glasses`, `_mustache`, `_dog`, `_crown`, `_clown`). There are **no PNG/sprite files** to
ship, load, or alpha-composite — the repo stays tiny and there's nothing to download. Trade-off:
the look is geometric/cartoonish rather than photoreal, which suits the playful intent.

**Anchors per prop.** Sunglasses sit on the two eye centres (averaged from a ring of eye
landmarks); the mustache hangs off the philtrum (164); dog ears rise above the forehead (10), the
nose sits on the nose tip (1); the crown is a zig-zag polygon above the forehead; the clown's
translucent cheeks blend over landmarks 50/280 with a red nose on the tip.

**Interactive (expression-driven) filters.** Beyond static props, four filters *react to your
face*. From the same landmarks we read three continuous signals in `face_expression()`, each
divided by the eye-corner distance so they're scale-invariant, and computed in **pixel** space so
the x/y aspect ratio is correct:

```python
jaw   = ‖inner_top_lip(13) − inner_bottom_lip(14)‖ / scale   # mouth open
smile = ‖mouth_left(61)    − mouth_right(291)‖     / scale    # mouth widen
brow  = mean(‖eyelid − eyebrow‖ for both eyes)     / scale    # eyebrow raise
```

Each is squashed to `0..1` with a calibrated `(value − rest)/range` clip. Those amounts drive a
tiny fixed-capacity particle pool (`class Emitter`) and a stateful `FaceFX`:

- **fire / bubbles** — while `jaw` is open, spawn particles at the mouth each frame. Fire shoots
  out along the face's down-axis, ages white→yellow→red, and is drawn as an **additive, blurred
  overlay** so it glows; bubbles rise, drawn as translucent rings.
- **hearts** — while you `smile`, spawn hearts that float up (drawn from two circles + a triangle),
  plus a heart over each eye.
- **lasers** — beams from each eye along an outward/down direction, thickness and brightness scaled
  by `brow`, rendered as a blurred red glow with a white core.

`FaceFX` deliberately splits `update()` (advance the sim once per captured frame) from `draw()`
(render current state), so a frame can be re-drawn — e.g. when you press `s` to save — without
double-stepping the animation. The on-screen status line shows the cue for the active filter
("open your mouth!", "smile :)", "raise your eyebrows!").

**Graceful degradation.** Same discipline as hands: if MediaPipe or the face model is missing,
`FaceTracker` construction is caught and photo mode simply shows the clean camera with no prop.
`--no-hands` disables all MediaPipe; `g` toggles tracking live.

---

## 7. Hand control — reaching into the field

**The idea.** Let the user *touch* the particles. Track the hands, read two simple gestures, and
turn them into forces on the particle field.

**Tracking** (`HandTracker`). We use MediaPipe's **HandLandmarker** (the Tasks API), which returns
**21 landmarks** per hand — knuckles and fingertips in normalized `[0,1]` image coordinates. The
model is a small bundled `.task` file; inference runs on CPU in VIDEO mode (it uses temporal
tracking across frames, which is why each call needs a monotonically increasing timestamp).

**Gestures from geometry, not a classifier.** Rather than train or load a gesture model, we read
two cheap geometric features off the landmarks:

```python
scale     = ‖wrist − middle_mcp‖                  # hand size, for scale-invariance
pinch_d   = ‖thumb_tip − index_tip‖ / scale       # small when pinching
spread    = mean(‖tip − wrist‖ for the 4 fingers) / scale   # large when palm is open

pinch_amt = clip(1 − (pinch_d − 0.20)/0.70, 0, 1) # ~1 when tips touch
openness  = clip((spread − 1.10)/0.90,    0, 1)   # ~1 when palm is spread
```

Dividing by `scale` makes the thresholds work whether your hand is near or far from the camera.
The decision is then just:

- `pinch_amt > 0.6` → **grab**
- else `openness > 0.5` → **push**
- else → **idle** (a relaxed/closed hand does nothing, so gestures stay intentional)

**Forces** (`Engine.apply_hands`). The gesture's action point is mapped into flow-space and turned
into a whole-array displacement of every particle, weighted by a radial falloff (full strength at
the hand, zero past radius `R = 0.42 × flow_width`):

```python
d       = particle_positions − hand_point     # vector from hand to each particle
falloff = clip(1 − ‖d‖/R, 0, 1)               # 1 at the hand → 0 at the edge

grab:  Δ = −d · (0.20·falloff)  +  hand_velocity · (1.1·falloff)
push:  Δ = (d/‖d‖) · (4.0·falloff)
particle_positions += Δ
```

- **grab** pulls particles toward the pinch (`−d`, a spring) *and* adds your hand's velocity
  (`hand_velocity`) so a quick pinch-and-sweep **flings** them.
- **push** drives particles radially outward along the unit vector `d/‖d‖`.

**Making grabbed particles glow.** The displacement (`kick`) is folded back into the velocity used
for colouring — `vel = vel + kick·5` — so particles you grab and fling light up bright and take
their hue from the direction you threw them, exactly like fast camera motion does.

**Same discipline as the rest.** Note there's still no Python loop over particles: the force is one
vectorized NumPy expression applied to all 6,000 at once, per hand.

**Graceful degradation.** If MediaPipe or the model file is missing, `HandTracker` construction is
caught, hand control switches off, and the rest of FluxCam runs unchanged. `--no-hands` skips it on
purpose, and `g` toggles it live.

---

## 8. Performance summary

| Stage | Cost driver | Mitigation |
|---|---|---|
| Optical flow | pixel count | computed at 320 px width, not full res |
| Particle update | particle count | fully vectorized NumPy, no Python loop |
| Splatting | particle count | single `np.add.at` call |
| Hand tracking | the ML model | optional (`--no-hands`); the only heavy ML in the app |
| Compositing | window size | cheap resize + weighted add |

The architecture deliberately **decouples** the expensive stage (flow) from the tunable one
(particle count): you can push the particle count way up without making flow any slower, because
particles live in the small flow-space and are only scaled up when splatted.

---

## 9. Where to look in the code

| Concept | Code |
|---|---|
| Face tracking | `class FaceTracker` |
| Face frame (centre/scale/roll) | `_face_frame` |
| AR filter props | `draw_filter`, `_glasses`, `_mustache`, `_dog`, `_crown`, `_clown` |
| Expression signals (mouth/smile/brow) | `face_expression` |
| Interactive filter particles | `class Emitter`, `class FaceFX` (`_draw_fire`, `_draw_bubbles`, `_draw_hearts`, `_draw_lasers`) |
| Optical flow | `Engine.step` → `cv2.calcOpticalFlowFarneback` |
| Particle advection & lifecycle | `class Particles` (`update`, `resize`) |
| Colour-by-direction | `colors_for` |
| Additive splat | `splat` (`np.add.at`) |
| Fading trail / glow | `Engine.step` (`self.trail *= cfg.decay`) |
| Flow & ink renderers | `render_flow`, the `ink` branch of `Engine.step` |
| Hand tracking | `class HandTracker` |
| Hand forces (grab/push/fling) | `Engine.apply_hands` |
| Live loop, keys, overlay | `run_live`, `handle_key`, `overlay`, `draw_hands` |
| Headless test | `run_selftest` |
