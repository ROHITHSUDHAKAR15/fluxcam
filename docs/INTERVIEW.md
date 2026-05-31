# Interview & Presentation Guide

A cheat sheet for demoing FluxCam and talking about it confidently. Use it to rehearse — the
goal is that you can explain every line you'd be asked about and defend every design choice.

---

## 1. The 30-second pitch

> "FluxCam is a real-time computer-vision toy in about 500 lines of Python with two halves. By
> default it's **AR face filters** — I use MediaPipe's 468-point face mesh to lock Snapchat-style
> props onto your face: sunglasses, dog ears, a crown. They track head tilt and distance because
> I build a little coordinate frame from the eye landmarks and draw everything relative to it.
> Press a key and it flips into **generative motion art** — I compute **dense optical flow**
> between consecutive frames, a per-pixel motion field, and push thousands of particles through
> it so they stream with your movement and leave glowing trails. I also wired **MediaPipe hand
> tracking** so you can pinch to grab and fling the particles. The whole thing runs on a laptop
> CPU with no GPU, because every particle operation is vectorized in NumPy — no per-particle
> Python loop anywhere."

That hits: real-time, CV fundamentals (optical flow + face mesh), real ML integration (two
MediaPipe models), performance engineering (vectorization), and product polish (interactive,
no special hardware, nothing to download).

---

## 2. The live demo script (≈90 seconds)

1. **Launch** `python fluxcam.py`. You appear with sunglasses on — *"clean camera, and this prop
   is anchored to MediaPipe's 468-point face mesh."* Tilt your head — *"it banks with me because I
   draw it in a coordinate frame built from the eye landmarks."* Lean in — *"and it scales with
   face size."*
2. **Press `n` a few times** — *"sunglasses, mustache, dog ears, crown, clown nose. Every prop is
   just OpenCV shapes anchored to landmarks — no image assets, nothing to download."*
3. **Press `m`** to particles — *"now the same webcam drives motion art. Each glowing dot is a
   particle carried by the optical-flow field; colour is the direction it's travelling."* Wave.
4. **Pinch** — *"MediaPipe also gives me 21 hand landmarks; thumb-to-index close = grab, and I add
   my hand's velocity so I can fling them."* (Green ring.) **Open palm** — *"open hand pushes."*
5. **Press `m`** to flow mode — *"this is the raw optical-flow field, colour-coded — direction is
   hue, magnitude is brightness. The actual data driving the particles."*
6. **Press `]` a few times** — *"I can push the particle count up live without slowing the flow
   computation, because particles are decoupled from flow cost."*

**No camera handy?** Run `python fluxcam.py --selftest` and show the generated PNGs — it proves
the engine works headlessly and is exactly what CI runs.

---

## 3. The architecture in one breath

```
photo mode:  webcam → MediaPipe FaceLandmarker (468 pts) → face frame → draw AR prop → window
art modes:   webcam → resize 320px → Farneback dense flow ┐
                                                          ├→ advect ~6k particles + hand forces
             MediaPipe HandLandmarker → grab/push forces ─┘   → colour by direction
                                  fade trail buffer (×decay) ← additive splat → glow → window
```

Ideas: **(1)** a face mesh + a derived coordinate frame anchor AR props that track head pose,
**(2)** dense optical flow captures motion as data, **(3)** a particle system is advected by that
field, **(4)** a fading trail buffer turns dots into glowing streaks, **(5)** hand tracking adds
direct manipulation.

---

## 4. Design decisions & trade-offs (the part interviewers probe)

| Decision | Why | Trade-off I accepted |
|---|---|---|
| **Props in a face coordinate frame, not at raw landmarks** | Building a frame from the eye corners (centre/scale/roll) makes every prop track head tilt and distance for free. | A little vector math up front instead of hard-coding pixel offsets. |
| **AR props as OpenCV shapes, no image assets** | Zero sprite files to ship/load/alpha-composite; the repo stays tiny and there's nothing to download. | The look is geometric/cartoonish, not photoreal. |
| **Dense flow, not sparse (Lucas–Kanade)** | I want the *whole* moving field to paint, not a few tracked corners. | More compute than sparse tracking. |
| **Flow at 320 px, not full resolution** | Flow is the most expensive op; cost scales with pixels. | Slightly softer flow; particles are scaled up to the display to hide it. |
| **Particles in flow-space, scaled at splat time** | Decouples particle count from flow cost — I can 5× the particles without touching flow. | A little coordinate bookkeeping (`sx`, `sy`). |
| **Vectorized NumPy, no per-particle loop** | Pure-Python per-particle loops would cap me at a few hundred particles; vectorized I do tens of thousands in real time. | Logic must be expressed as whole-array ops, which is less obvious to read. |
| **`np.add.at` for splatting** | Correctly *accumulates* overlapping particles; plain indexed assignment silently drops collisions. | Slightly slower than unsafe assignment — worth it for correct additive glow. |
| **Float trail buffer, clamp at the end** | Lets brightness exceed 255 so overlapping trails bloom. | A few extra MB of memory. |
| **Gestures from geometry, not a classifier** | Two distances (pinch, spread) are robust, need zero training, and the thresholds are explainable. | Only knows *pinch* and *open palm* — no rich vocabulary. |
| **Hand control optional + graceful fallback** | MediaPipe is a big dependency; the art should still run without it. | A branch and a try/except around tracker setup. |
| **CPU-only, no GPU** | Runs anywhere, no driver/setup friction — great for a portfolio piece. | A GPU shader splat would do millions of particles; that's the stated next step. |

If you only memorize one row, memorize **"particles decoupled from flow cost via flow-space"** —
it's the cleanest piece of systems thinking in the project.

---

## 5. Likely questions & strong answers

**Q: How do the AR face filters stay aligned when I tilt or move?**
I don't draw props at raw landmark pixels. I take two stable landmarks — the outer eye corners —
and build a small coordinate frame from them: the midpoint is the origin, the distance between
them is my scale unit, and the angle of the line between them is the head roll. Every prop is then
expressed as offsets *in eye-distance units*, rotated by that angle. So when I lean in, `scale`
grows and the prop grows; when I tilt my head, `angle` changes and the prop banks with me. It's the
same trick a 2D rigging system uses — a local space attached to the face.

**Q: Why draw the props with OpenCV shapes instead of PNG overlays?**
Two reasons. It keeps the repo self-contained — no sprite assets to ship, version, or load, and
nothing for a user to download. And it avoids per-frame alpha compositing of transparent images,
which is fiddly to get right under rotation. The cost is that the props look geometric rather than
photoreal, which fits the playful intent. Swapping in textured PNGs later is a contained change —
the anchor points and face frame are already computed.

**Q: Why two MediaPipe models (face *and* hand) instead of one?**
They serve the two different halves: FaceLandmarker (468 points) drives the AR filters in photo
mode, HandLandmarker (21 points) drives grab/push in the art modes. I only run one per frame —
whichever the current mode needs — so I never pay for both at once.

**Q: What *is* optical flow, concretely?**
A 2-channel image the same size as the frame, where each pixel holds `(dx, dy)` — how far that
point appears to have moved since the previous frame. Farnebäck computes it by fitting a local
quadratic to the brightness around each pixel in both frames and solving for the displacement,
coarse-to-fine over an image pyramid so it catches large motions.

**Q: Why does a blank wall not light up?**
Optical flow measures apparent motion of *texture/brightness*, not objects. A featureless region
has no gradient to track, so its flow is ~0. A patterned shirt has lots of structure, so it
lights up. It's an inherent property of the method, and I lean into it as part of the look.

**Q: How do you hit real-time with thousands of particles in Python?**
Two things. First, the particle state is parallel NumPy arrays and every operation — advection,
ageing, respawn, colouring, splatting, hand forces — is a whole-array vectorized expression, so
there's no Python-level loop over particles; it's all in C under NumPy/OpenCV. Second, flow runs
at a small 320 px width and particles live in that space, so particle count doesn't affect flow
cost.

**Q: How does the splat handle two particles on the same pixel?**
`np.add.at(canvas, (ys, xs), colors)` *accumulates* — both contribute. A naive
`canvas[ys, xs] = colors` would let one overwrite the other non-deterministically. Accumulation
is also what makes overlapping trails bloom to white, which is the realistic light behaviour I
want.

**Q: Where does the "glow / trail" come from?**
A persistent float canvas. Each frame I multiply it by a decay factor (~0.86) and add the new
particles on top. So any splat fades exponentially over many frames instead of disappearing —
that decaying tail is the trail. The decay factor is the trail-length knob.

**Q: How does the pinch detection work?**
MediaPipe returns 21 landmarks per hand. I compute the thumb-tip-to-index-tip distance, divide by
the hand size (wrist→middle-knuckle distance) so it's scale-invariant, and threshold it. Open
palm is the average fingertip-to-wrist spread, again normalized. No ML classifier — just two
geometric ratios with explainable thresholds.

**Q: Why MediaPipe's Tasks API and not `mp.solutions.hands`?**
The current MediaPipe wheels ship only the Tasks API — the legacy `solutions` module isn't in the
build I have. So I use `HandLandmarker` with a bundled `.task` model and run it in VIDEO mode,
which does temporal tracking and needs monotonically increasing timestamps per frame.

**Q: How do you "fling" particles if they have no velocity state?**
Particles are advected fresh from the flow each frame, so there's no stored velocity. For a fling
I add the *hand's* per-frame velocity directly to the grabbed particles' positions, scaled by the
radial falloff. They get a real directional shove, and because I fold that displacement back into
the colour-velocity, they also light up and take their hue from the throw direction.

**Q: How would you scale to millions of particles?**
Move the splat to the GPU — a fragment/compute shader with additive blending (moderngl/OpenGL) —
and keep particle state in a GPU buffer. The CPU stays responsible only for flow and hand
tracking. The architecture already isolates the splat, so this is a contained change.

**Q: How is it tested without a camera?**
`--selftest` runs the engine headlessly on synthetic motion (a programmatically swept blob) and
writes one PNG per mode, printing a `lit%` metric (fraction of bright pixels) to confirm
particles actually responded. It needs no webcam or GUI, so it's CI-friendly. I also unit-checked
the hand forces directly: assert that "grab" reduces mean particle distance to the hand point,
"push" increases it, and a sideways fling produces a positive sideways displacement. The AR filters
I exercised by feeding a synthetic 468-landmark face into `draw_filter` for every filter and
asserting it draws without error and actually changes pixels — no webcam needed.

**Q: What's the biggest weakness?**
Optical flow is content-dependent (no texture → no motion signal), and the gesture vocabulary is
intentionally tiny (pinch + palm). Both are conscious trade-offs for robustness and real-time
performance, but they're the honest limits.

---

## 6. Complexity & numbers to know

- **Per frame, photo mode:** one MediaPipe FaceLandmarker inference + a few OpenCV draw calls.
- **Per frame, art modes:** O(pixels) for flow at 320×~180, plus O(N) for `N` particles
  (advection, splat, hand forces), plus one MediaPipe HandLandmarker inference if hands are on.
- **Defaults:** photo mode + sunglasses filter, 6,000 particles (tunable 500–40,000), trail decay
  0.86 (0.50–0.98), flow width 320 px, output 960×540.
- **Models:** face mesh ~3.8 MB (468 landmarks), hand ~7.8 MB (21 landmarks, up to 2 hands).
- **No GPU, no network, no per-particle Python loop.**

---

## 7. What this project demonstrates (map to a JD)

- **Computer-vision fundamentals** — dense optical flow, the difference from sparse tracking,
  HSV motion encoding, and a face-landmark coordinate frame for pose-stable AR overlays.
- **ML integration** — wiring two real MediaPipe models (face mesh + hands) into a real-time
  loop, including their API quirks (Tasks vs solutions, VIDEO-mode timestamps) and graceful
  fallback.
- **Performance engineering** — vectorization over interpreted loops, decoupling cost centres,
  picking the right resolution for the right stage.
- **Clean design** — a pure-function `Engine.step` reused by both the live loop and the headless
  test; optional dependencies that fail gracefully; one file, no framework ceremony.
- **Product sense** — interactive, runs on any laptop, self-documenting on-screen help, and a
  test mode anyone can run to verify it.

---

## 8. One-line closer

> "It's a small codebase, but it touches the whole stack of a real-time CV app — capture,
> motion estimation, a simulated particle system, an ML model, and the performance work to make
> all of it run together at frame rate on a CPU."
