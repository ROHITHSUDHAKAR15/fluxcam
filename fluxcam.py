#!/usr/bin/env python3
"""
FluxCam — AR face filters + motion-painted particles.

Point your webcam at yourself. By default you get a clean, sharp camera with a
Snapchat-style AR face filter locked onto your face (sunglasses, mustache, dog ears,
a crown, a clown nose) — press n to swap filters. Press m and your motion turns into
glowing particle art instead.

The two halves:
  * photo mode (default): a clean live camera with an AR face filter tracked by
    MediaPipe FaceLandmarker (468 face points). The props rotate, scale, and follow your
    head in real time. n cycles filters, s saves a PNG. Face always sharp and centered.
  * particles / flow / ink modes (press m): your motion painted as glowing light.
    Dense optical flow (Farneback) measures how the image is *flowing* every frame and
    advects thousands of particles through it; a fading trail buffer gives the glow.
    Pinch (MediaPipe hands) to grab and fling the particles, open palm to push them.

Why the art looks alive (the three ideas doing the work):
  1. Dense optical flow turns two consecutive frames into a per-pixel velocity field:
     which way and how fast did each point move? That field *is* your motion.
  2. A particle system samples that field and advects ~6000 points through it, coloured
     by the direction they travel, so a wave of your hand paints an arc of rainbow.
  3. A fading trail buffer (each frame dimmed, new splats added) gives the long-exposure
     glow instead of flickering dots.

Tracking is MediaPipe (on by default). Everything else is vectorized NumPy + OpenCV —
no per-particle Python loop — so it runs in real time on a laptop CPU.

Run it:
    python fluxcam.py                   # clean camera + AR face filter (press n to swap)
    python fluxcam.py --mode particles  # the glowing motion-particle art
    python fluxcam.py --mode flow       # dense-flow rainbow mode
    python fluxcam.py --input clip.mp4
    python fluxcam.py --no-hands        # disable all MediaPipe tracking
    python fluxcam.py --selftest        # headless: render synthetic frames to PNGs

Keys (window focused):
    q/Esc quit   space pause   m mode   n next face filter   c particle colour
    x camera ghost   [ ] fewer/more particles   - = shorter/longer trails
    f mirror   g tracking on/off   r clear trail   s save PNG   h help
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

MODES = ["photo", "particles", "flow", "ink"]
PARTICLE_COLORS = ["direction", "camera", "ember"]
FILTERS = ["none", "sunglasses", "mustache", "dog", "crown", "clown"]

_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_HAND_MODEL = os.path.join(_MODELS, "hand_landmarker.task")
DEFAULT_FACE_MODEL = os.path.join(_MODELS, "face_landmarker.task")


# --------------------------------------------------------------------------------------
# Particle system — all state is parallel NumPy arrays, updated without a Python loop.
# --------------------------------------------------------------------------------------
class Particles:
    def __init__(self, n: int, w: int, h: int, rng: np.random.Generator):
        self.rng = rng
        self.w, self.h = w, h
        self.pos = self._rand_pos(n)                         # (n, 2) float x,y
        self.age = rng.uniform(0, 1, n).astype(np.float32)   # 0..1, respawn near 1
        self.life = rng.uniform(0.6, 1.6, n).astype(np.float32)

    def _rand_pos(self, n: int) -> np.ndarray:
        p = np.empty((n, 2), np.float32)
        p[:, 0] = self.rng.uniform(0, self.w, n)
        p[:, 1] = self.rng.uniform(0, self.h, n)
        return p

    def resize(self, n: int):
        cur = len(self.pos)
        if n == cur:
            return
        if n < cur:
            self.pos, self.age, self.life = self.pos[:n], self.age[:n], self.life[:n]
        else:
            extra = n - cur
            self.pos = np.vstack([self.pos, self._rand_pos(extra)])
            self.age = np.concatenate([self.age, self.rng.uniform(0, 1, extra).astype(np.float32)])
            self.life = np.concatenate([self.life, self.rng.uniform(0.6, 1.6, extra).astype(np.float32)])

    def update(self, flow: np.ndarray, speed: float, dt: float):
        """Advect every particle by the flow at its location (+ a little drift)."""
        fh, fw = flow.shape[:2]
        xi = np.clip(self.pos[:, 0].astype(np.int32), 0, fw - 1)
        yi = np.clip(self.pos[:, 1].astype(np.int32), 0, fh - 1)
        vel = flow[yi, xi]                                   # (n, 2) dx,dy
        self.prev = self.pos.copy()
        self.pos += vel * speed
        # gentle ambient drift so the field stays gently alive when you hold still
        self.pos[:, 1] += 0.15
        # age, and respawn the dead or the escaped
        self.age += dt / self.life
        oob = ((self.pos[:, 0] < 0) | (self.pos[:, 0] >= fw) |
               (self.pos[:, 1] < 0) | (self.pos[:, 1] >= fh) | (self.age >= 1.0))
        k = int(oob.sum())
        if k:
            self.pos[oob] = self._rand_pos(k)
            self.prev[oob] = self.pos[oob]
            self.age[oob] = 0.0
        return vel


# --------------------------------------------------------------------------------------
# Hand control — MediaPipe Tasks HandLandmarker boiled down to per-hand "action points".
# Pinch (thumb-index tips together) = grab/fling, open palm = push. mediapipe is an
# optional dependency; if it (or the model) is missing, FluxCam still runs without hands.
# --------------------------------------------------------------------------------------
class HandTracker:
    TIPS = [8, 12, 16, 20]                       # index/middle/ring/pinky finger tips

    def __init__(self, model_path: str, max_hands: int = 2):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.mp = mp
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self.prev: dict[str, tuple[float, float]] = {}   # handedness label -> last pinch xy
        self.t0 = time.time()
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray) -> list[dict]:
        """Return a list of hands as dicts: pos (normalized xy), vel, act, amt."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = max(self._last_ts + 1, int((time.time() - self.t0) * 1000))  # must increase
        self._last_ts = ts
        res = self.landmarker.detect_for_video(image, ts)

        hands = []
        for i, lms in enumerate(res.hand_landmarks):
            label = (res.handedness[i][0].category_name
                     if i < len(res.handedness) else str(i))
            P = np.array([[p.x, p.y] for p in lms], np.float32)
            wrist, mmcp = P[0], P[9]
            scale = float(np.linalg.norm(wrist - mmcp)) + 1e-6
            pinch_d = float(np.linalg.norm(P[4] - P[8])) / scale
            pinch_amt = float(np.clip(1 - (pinch_d - 0.20) / 0.70, 0, 1))
            spread = float(np.mean(np.linalg.norm(P[self.TIPS] - wrist, axis=1))) / scale
            openness = float(np.clip((spread - 1.10) / 0.90, 0, 1))

            px, py = ((P[4] + P[8]) / 2.0).tolist()        # pinch midpoint
            prev = self.prev.get(label, (px, py))
            vel = (px - prev[0], py - prev[1])
            self.prev[label] = (px, py)

            if pinch_amt > 0.6:
                act, amt = "grab", pinch_amt
            elif openness > 0.5:
                act, amt = "push", openness
            else:
                act, amt = "idle", 0.0
            hands.append({"pos": (px, py), "vel": vel, "act": act, "amt": amt})
        return hands

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


def draw_hands(img: np.ndarray, hands: list[dict]):
    """Feedback markers: green ring = grab, blue ring = push."""
    H, W = img.shape[:2]
    for h in hands:
        if h["act"] == "idle":
            continue
        x, y = int(h["pos"][0] * W), int(h["pos"][1] * H)
        color = (120, 255, 120) if h["act"] == "grab" else (255, 180, 80)
        r = int(18 + 34 * h["amt"])
        cv2.circle(img, (x, y), r, color, 2, cv2.LINE_AA)
        cv2.circle(img, (x, y), 3, color, -1, cv2.LINE_AA)


# --------------------------------------------------------------------------------------
# Face tracking — MediaPipe Tasks FaceLandmarker (468 landmarks). We keep the raw
# normalized landmark array per face; the filter renderers below derive a small face
# frame (centre, scale, roll) from a couple of landmarks and draw props onto it. Like
# HandTracker, it degrades gracefully if mediapipe or the model file is missing.
# --------------------------------------------------------------------------------------
class FaceTracker:
    def __init__(self, model_path: str, max_faces: int = 1):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.mp = mp
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=max_faces,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        self.t0 = time.time()
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray) -> list[dict]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = max(self._last_ts + 1, int((time.time() - self.t0) * 1000))   # must increase
        self._last_ts = ts
        res = self.landmarker.detect_for_video(image, ts)
        faces = []
        for lms in res.face_landmarks:
            lm = np.array([[p.x, p.y] for p in lms], np.float32)            # (468, 2) norm
            faces.append({"lm": lm})
        return faces

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# AR face filters — every prop is drawn with plain OpenCV primitives (no image assets),
# anchored to MediaPipe landmarks and oriented to a face frame (centre between the eyes,
# scale = eye-corner distance, roll = angle of the eye line) so they track head tilt.
# --------------------------------------------------------------------------------------
# A few canonical MediaPipe FaceMesh indices we anchor to.
L_EYE_OUT, R_EYE_OUT = 33, 263                 # outer eye corners (define the face frame)
L_EYE_RING = [33, 133, 159, 145]               # points around the left-image eye
R_EYE_RING = [362, 263, 386, 374]              # points around the right-image eye
NOSE_TIP, PHILTRUM, LOWER_LIP = 1, 164, 14
FOREHEAD, L_CHEEK, R_CHEEK = 10, 50, 280


def _px(lm: np.ndarray, idx, W: int, H: int) -> np.ndarray:
    p = lm[idx]
    if p.ndim == 2:
        p = p.mean(0)
    return np.array([p[0] * W, p[1] * H], np.float32)


def _face_frame(lm: np.ndarray, W: int, H: int):
    eL, eR = _px(lm, L_EYE_OUT, W, H), _px(lm, R_EYE_OUT, W, H)
    vec = eR - eL
    scale = float(np.linalg.norm(vec)) + 1e-6
    angle = float(np.arctan2(vec[1], vec[0]))
    ux = np.array([np.cos(angle), np.sin(angle)], np.float32)   # along the eye line
    uy = np.array([-np.sin(angle), np.cos(angle)], np.float32)  # down the face
    return (eL + eR) / 2.0, scale, np.degrees(angle), ux, uy


def _glasses(img, lm, W, H, scale, deg, ux):
    cL = _px(lm, L_EYE_RING, W, H)
    cR = _px(lm, R_EYE_RING, W, H)
    ax = (int(scale * 0.46), int(scale * 0.34))
    for c in (cL, cR):
        cv2.ellipse(img, tuple(c.astype(int)), ax, deg, 0, 360, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.ellipse(img, tuple(c.astype(int)), ax, deg, 0, 360, (210, 210, 210), 2, cv2.LINE_AA)
        cv2.ellipse(img, tuple((c - ux * scale * 0.12).astype(int)),
                    (int(scale * 0.16), int(scale * 0.10)), deg, 0, 360, (70, 70, 70), -1, cv2.LINE_AA)
    cv2.line(img, tuple((cL + ux * scale * 0.46).astype(int)),
             tuple((cR - ux * scale * 0.46).astype(int)), (210, 210, 210), 3, cv2.LINE_AA)
    for c, s in ((cL, -1), (cR, 1)):
        a = c + ux * (scale * 0.46 * s)
        cv2.line(img, tuple(a.astype(int)), tuple((a + ux * scale * 0.5 * s).astype(int)),
                 (210, 210, 210), 3, cv2.LINE_AA)


def _mustache(img, lm, W, H, scale, deg, ux, uy):
    base = _px(lm, PHILTRUM, W, H) + uy * scale * 0.08
    off = ux * scale * 0.30
    ax = (int(scale * 0.36), int(scale * 0.20))
    cv2.ellipse(img, tuple((base - off).astype(int)), ax, deg - 12, 0, 360, (28, 24, 20), -1, cv2.LINE_AA)
    cv2.ellipse(img, tuple((base + off).astype(int)), ax, deg + 12, 0, 360, (28, 24, 20), -1, cv2.LINE_AA)


def _dog(img, lm, W, H, scale, deg, ux, uy):
    top = _px(lm, FOREHEAD, W, H)
    ax = (int(scale * 0.34), int(scale * 0.66))
    brown, pink = (40, 70, 120), (150, 150, 235)
    for s, tilt in ((-1, -22), (1, 22)):
        c = top + ux * scale * 0.95 * s - uy * scale * 0.55
        cv2.ellipse(img, tuple(c.astype(int)), ax, deg + tilt, 0, 360, brown, -1, cv2.LINE_AA)
        cv2.ellipse(img, tuple(c.astype(int)), (int(ax[0] * 0.55), int(ax[1] * 0.6)),
                    deg + tilt, 0, 360, pink, -1, cv2.LINE_AA)
    nose = _px(lm, NOSE_TIP, W, H)
    cv2.ellipse(img, tuple(nose.astype(int)), (int(scale * 0.22), int(scale * 0.16)),
                deg, 0, 360, (30, 25, 25), -1, cv2.LINE_AA)
    cv2.circle(img, tuple((nose - ux * scale * 0.06 - uy * scale * 0.05).astype(int)),
               max(1, int(scale * 0.05)), (190, 190, 190), -1, cv2.LINE_AA)
    tongue = _px(lm, LOWER_LIP, W, H) + uy * scale * 0.28
    cv2.ellipse(img, tuple(tongue.astype(int)), (int(scale * 0.17), int(scale * 0.30)),
                deg, 0, 360, (120, 120, 240), -1, cv2.LINE_AA)


def _crown(img, lm, W, H, scale, ux, uy):
    base = _px(lm, FOREHEAD, W, H) - uy * scale * 0.45
    bl, br = base - ux * scale * 0.9, base + ux * scale * 0.9
    up = -uy * scale * 0.75
    n = 5
    pts = [bl]
    peaks = []
    for i in range(n):
        valley = bl + (br - bl) * (i / n)
        peak = bl + (br - bl) * ((i + 0.5) / n) + up
        peaks.append(peak)
        pts += [valley, peak]
    pts.append(br)
    poly = np.array([p.astype(int) for p in pts], np.int32)
    cv2.fillPoly(img, [poly], (60, 200, 255), cv2.LINE_AA)
    cv2.polylines(img, [poly], True, (30, 120, 200), 2, cv2.LINE_AA)
    for peak in peaks:
        cv2.circle(img, tuple(peak.astype(int)), max(2, int(scale * 0.06)), (80, 80, 230), -1, cv2.LINE_AA)


def _clown(img, lm, W, H, scale):
    ov = img.copy()
    for idx in (L_CHEEK, R_CHEEK):
        cv2.circle(ov, tuple(_px(lm, idx, W, H).astype(int)), int(scale * 0.25), (130, 130, 240), -1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.35, img, 0.65, 0, img)
    nose = _px(lm, NOSE_TIP, W, H)
    r = int(scale * 0.32)
    cv2.circle(img, tuple(nose.astype(int)), r, (60, 60, 235), -1, cv2.LINE_AA)
    cv2.circle(img, tuple((nose - [r * 0.3, r * 0.35]).astype(int)),
               max(2, int(r * 0.32)), (210, 210, 255), -1, cv2.LINE_AA)


def draw_filter(img: np.ndarray, faces: list[dict], fi: int):
    """Draw the selected AR prop onto each detected face."""
    name = FILTERS[fi]
    if name == "none" or not faces:
        return
    H, W = img.shape[:2]
    for face in faces:
        lm = face["lm"]
        _, scale, deg, ux, uy = _face_frame(lm, W, H)
        if name == "sunglasses":
            _glasses(img, lm, W, H, scale, deg, ux)
        elif name == "mustache":
            _mustache(img, lm, W, H, scale, deg, ux, uy)
        elif name == "dog":
            _dog(img, lm, W, H, scale, deg, ux, uy)
        elif name == "crown":
            _crown(img, lm, W, H, scale, ux, uy)
        elif name == "clown":
            _clown(img, lm, W, H, scale)


# --------------------------------------------------------------------------------------
# App state
# --------------------------------------------------------------------------------------
@dataclass
class Cfg:
    n: int = 6000
    mode: int = 0
    pcolor: int = 0
    decay: float = 0.86          # trail persistence (higher = longer trails)
    mirror: bool = True
    ghost: bool = True           # faint camera image under the particles
    paused: bool = False
    show_help: bool = True
    hands: bool = True           # MediaPipe tracking (face filters + hand control)
    filter: int = 1              # AR face filter index (photo mode); 0 = none


FLOW_W = 320                     # optical flow is computed at this width (fast); scaled up


def colors_for(cfg: Cfg, vel: np.ndarray, pos: np.ndarray, small_bgr: np.ndarray) -> np.ndarray:
    """Per-particle BGR colour (float) used as additive brightness when splatted."""
    mode = PARTICLE_COLORS[cfg.pcolor]
    speed = np.linalg.norm(vel, axis=1)
    bright = np.clip(speed * 0.5 + 0.25, 0.25, 1.6)[:, None]
    if mode == "direction":
        ang = (np.arctan2(vel[:, 1], vel[:, 0]) + np.pi) / (2 * np.pi)  # 0..1
        hsv = np.zeros((len(vel), 1, 3), np.uint8)
        hsv[:, 0, 0] = (ang * 180).astype(np.uint8)
        hsv[:, 0, 1] = 255
        hsv[:, 0, 2] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3).astype(np.float32)
        return bgr * bright
    if mode == "camera":
        fh, fw = small_bgr.shape[:2]
        xi = np.clip(pos[:, 0].astype(np.int32), 0, fw - 1)
        yi = np.clip(pos[:, 1].astype(np.int32), 0, fh - 1)
        return small_bgr[yi, xi].astype(np.float32) * bright
    # ember: hot orange/white scaled by speed
    base = np.array([40, 140, 255], np.float32)              # BGR -> warm orange
    return base[None, :] * bright


def splat(canvas: np.ndarray, pos: np.ndarray, colors: np.ndarray, scale_x: float, scale_y: float):
    """Additively stamp particles onto the display-resolution canvas."""
    H, W = canvas.shape[:2]
    xs = (pos[:, 0] * scale_x).astype(np.int32)
    ys = (pos[:, 1] * scale_y).astype(np.int32)
    m = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    np.add.at(canvas, (ys[m], xs[m]), colors[m])


# --------------------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------------------
def render_flow(flow: np.ndarray, small_bgr: np.ndarray, out_size) -> np.ndarray:
    """Dense-flow rainbow: motion direction -> hue, motion strength -> brightness."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 18, 0, 255).astype(np.uint8)
    rgbflow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    base = cv2.addWeighted(small_bgr, 0.25, rgbflow, 1.0, 0)
    return cv2.resize(base, out_size, interpolation=cv2.INTER_LINEAR)


def compose(trail: np.ndarray, cfg: Cfg, small_bgr: np.ndarray, out_size) -> np.ndarray:
    disp = np.clip(trail, 0, 255).astype(np.uint8)
    if cfg.ghost:
        ghost = cv2.resize(small_bgr, out_size, interpolation=cv2.INTER_LINEAR)
        disp = cv2.addWeighted(disp, 1.0, ghost, 0.18, 0)
    return disp


# --------------------------------------------------------------------------------------
# Core step (pure function of frames -> image; reused by live loop and self-test)
# --------------------------------------------------------------------------------------
class Engine:
    def __init__(self, cfg: Cfg, out_w: int, out_h: int, seed: int = 7):
        self.cfg = cfg
        self.out_size = (out_w, out_h)
        self.flow_w = FLOW_W
        self.flow_h = None
        self.prev_gray = None
        self.rng = np.random.default_rng(seed)
        self.particles: Particles | None = None
        self.trail = np.zeros((out_h, out_w, 3), np.float32)

    def _ensure(self, fh, fw):
        if self.flow_h is None:
            self.flow_h = int(self.flow_w * fh / fw)
            self.particles = Particles(self.cfg.n, self.flow_w, self.flow_h, self.rng)

    def apply_hands(self, hands: list[dict] | None):
        """Push particle positions around in flow-space from hand gestures.

        Returns the per-particle displacement (the 'kick') so callers can fold it into
        the colour velocity — grabbed/flung particles then light up and take their hue
        from the direction the hand threw them.
        """
        p = self.particles
        if not hands or p is None:
            return None
        fw, fh = self.flow_w, self.flow_h
        R = 0.42 * fw                                # influence radius in flow pixels
        kick = np.zeros_like(p.pos)
        for h in hands:
            act = h["act"]
            if act == "idle":
                continue
            cx, cy = h["pos"][0] * fw, h["pos"][1] * fh
            d = p.pos - np.array([cx, cy], np.float32)        # centre -> particle
            dist = np.sqrt((d * d).sum(1)) + 1e-3
            falloff = np.clip(1 - dist / R, 0, 1)[:, None]    # 1 at centre, 0 at edge
            if act == "grab":
                hv = np.array([h["vel"][0] * fw, h["vel"][1] * fh], np.float32)
                kick += -d * (0.20 * falloff) + hv[None, :] * (1.1 * falloff)
            else:                                             # push
                kick += (d / dist[:, None]) * (4.0 * falloff)
        p.pos += kick
        return kick

    def step(self, frame_bgr: np.ndarray, dt: float = 1 / 30,
             hands: list[dict] | None = None) -> np.ndarray:
        fh, fw = frame_bgr.shape[:2]
        self._ensure(fh, fw)
        # photo mode: a clean, clear camera image — no flow, no particles. The AR face
        # filter is drawn on top by the caller (run_live) using FaceTracker landmarks.
        if MODES[self.cfg.mode] == "photo":
            return cv2.resize(frame_bgr, self.out_size, interpolation=cv2.INTER_LINEAR)
        small = cv2.resize(frame_bgr, (self.flow_w, self.flow_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0)
        self.prev_gray = gray

        cfg = self.cfg
        sx, sy = self.out_size[0] / self.flow_w, self.out_size[1] / self.flow_h

        if MODES[cfg.mode] == "flow":
            return render_flow(flow, small, self.out_size)

        # particle + ink modes both advect particles into the fading trail buffer
        self.particles.resize(cfg.n)
        vel = self.particles.update(flow, speed=2.2, dt=dt)
        kick = self.apply_hands(hands)
        if kick is not None:
            vel = vel + kick * 5.0            # grabbed/flung particles glow + take hand's hue
        colors = colors_for(cfg, vel, self.particles.pos, small)

        self.trail *= cfg.decay
        if MODES[cfg.mode] == "ink":
            # ink: blur the trail a touch each frame so splats bleed like dye in water
            self.trail = cv2.GaussianBlur(self.trail, (0, 0), 1.1)
            colors *= 1.4
        splat(self.trail, self.particles.pos, colors, sx, sy)
        return compose(self.trail, cfg, small, self.out_size)


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
def overlay(img, cfg: Cfg, fps: float):
    if MODES[cfg.mode] == "photo":
        s = f"photo | filter: {FILTERS[cfg.filter]} (n = next) | {fps:4.1f} fps"
    else:
        s = f"{MODES[cfg.mode]} | {PARTICLE_COLORS[cfg.pcolor]} | {cfg.n} particles | trail {cfg.decay:.2f} | {fps:4.1f} fps"
    cv2.putText(img, s, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if cfg.show_help:
        lines = ["m mode   n next filter   c colour   x ghost   f mirror   g tracking",
                 "[ ] particles    - = trails    r clear trail    s save    space pause    q quit",
                 "photo: AR face filter follows you   art modes: pinch to grab, open palm to push"]
        for i, line in enumerate(lines):
            y = img.shape[0] - 16 - (len(lines) - 1 - i) * 22
            cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 220, 255), 1, cv2.LINE_AA)


def handle_key(k: int, cfg: Cfg, eng: "Engine") -> bool:
    if k in (ord("q"), 27):
        return False
    elif k == ord(" "):
        cfg.paused = not cfg.paused
    elif k == ord("m"):
        cfg.mode = (cfg.mode + 1) % len(MODES)
    elif k == ord("n"):
        cfg.filter = (cfg.filter + 1) % len(FILTERS)
    elif k == ord("c"):
        cfg.pcolor = (cfg.pcolor + 1) % len(PARTICLE_COLORS)
    elif k == ord("x"):
        cfg.ghost = not cfg.ghost
    elif k == ord("f"):
        cfg.mirror = not cfg.mirror
    elif k == ord("h"):
        cfg.show_help = not cfg.show_help
    elif k == ord("g"):
        cfg.hands = not cfg.hands
    elif k == ord("["):
        cfg.n = max(500, cfg.n - 1000)
    elif k == ord("]"):
        cfg.n = min(40000, cfg.n + 1000)
    elif k in (ord("-"), ord("_")):
        cfg.decay = max(0.5, round(cfg.decay - 0.02, 2))
    elif k in (ord("="), ord("+")):
        cfg.decay = min(0.98, round(cfg.decay + 0.02, 2))
    elif k == ord("r"):
        eng.trail[:] = 0
    return True


def save_png(img) -> str:
    name = datetime.now().strftime("fluxcam_%Y%m%d_%H%M%S.png")
    cv2.imwrite(name, img)
    print("saved", name)
    return name


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------
def run_live(src: str, cfg: Cfg, out_w: int, out_h: int, hand_model: str, face_model: str) -> int:
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("could not open camera/video. Try --input <index|file> or --selftest.", file=sys.stderr)
        return 1

    # Photo mode uses the face tracker (AR filters); the art modes use the hand tracker
    # (grab/push). We build whichever is available and pick per-frame by mode.
    hand_tracker = face_tracker = None
    if cfg.hands:
        try:
            face_tracker = FaceTracker(face_model)
        except Exception as e:                       # missing mediapipe or model file
            print(f"face filters unavailable ({type(e).__name__}: {e}).", file=sys.stderr)
        try:
            hand_tracker = HandTracker(hand_model)
        except Exception as e:
            print(f"hand control unavailable ({type(e).__name__}: {e}).", file=sys.stderr)
        if hand_tracker is None and face_tracker is None:
            cfg.hands = False
        else:
            print("tracking ON: photo mode = AR face filters (n = swap),"
                  " art modes = pinch to grab / open palm to push (g toggles).")

    eng = Engine(cfg, out_w, out_h)
    win = "FluxCam - AR face filters + motion art (h = help)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last, hands, faces = None, [], []
    fps, t = 0.0, time.time()
    print("FluxCam running - press n to swap filters, m for art modes, h for keys, q to quit.")
    while True:
        if not cfg.paused:
            ok, frame = cap.read()
            if not ok:
                if src.isdigit():
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if cfg.mirror:
                frame = cv2.flip(frame, 1)
            hands, faces = [], []
            if cfg.hands:
                if MODES[cfg.mode] == "photo":
                    faces = face_tracker.process(frame) if face_tracker else []
                else:
                    hands = hand_tracker.process(frame) if hand_tracker else []
            now = time.time()
            dt = max(1e-3, now - t)
            last = eng.step(frame, dt=dt, hands=hands)
            inst = 1.0 / dt
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            t = now
        if last is not None:
            shown = last.copy()
            if MODES[cfg.mode] == "photo":
                draw_filter(shown, faces, cfg.filter)
            elif cfg.hands:
                draw_hands(shown, hands)
            overlay(shown, cfg, fps)
            cv2.imshow(win, shown)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s") and last is not None:
            snap = last.copy()                       # save the clean frame + filter, no HUD
            if MODES[cfg.mode] == "photo":
                draw_filter(snap, faces, cfg.filter)
            save_png(snap)
        elif k != 255 and not handle_key(k, cfg, eng):
            break
    for tr in (hand_tracker, face_tracker):
        if tr is not None:
            tr.close()
    cap.release()
    cv2.destroyAllWindows()
    return 0


def run_selftest(cfg: Cfg, out_w: int, out_h: int) -> int:
    """Headless smoke test: drive synthetic motion through the engine and save outputs."""
    eng = Engine(cfg, out_w, out_h)
    H, W = 360, 640
    last = None
    for i in range(40):                       # a blob sweeping across + a bouncing dot
        f = np.full((H, W, 3), 12, np.uint8)
        x = int(40 + (W - 80) * (i / 39))
        cv2.circle(f, (x, 180), 60, (60, 200, 255), -1)
        cv2.circle(f, (320, int(80 + 200 * abs(np.sin(i / 5)))), 30, (255, 120, 60), -1)
        last = eng.step(f, dt=1 / 30)
    for mode in range(len(MODES)):
        cfg.mode = mode
        eng.trail[:] = 0
        for i in range(20):
            f = np.full((H, W, 3), 12, np.uint8)
            x = int(40 + (W - 80) * (i / 19))
            cv2.circle(f, (x, 180), 60, (60, 200, 255), -1)
            last = eng.step(f, dt=1 / 30)
        name = f"selftest_{MODES[mode]}.png"
        cv2.imwrite(name, last)
        bright = float((last > 25).mean()) * 100
        print(f"{MODES[mode]:9s} -> {name}  {last.shape[1]}x{last.shape[0]}  lit={bright:5.2f}%  mean={last.mean():.1f}")
    print("selftest OK" if last is not None else "selftest FAILED")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="FluxCam - AR face filters + motion-painted particles.")
    p.add_argument("--input", default="0", help="camera index (default 0) or video file")
    p.add_argument("--mode", choices=MODES, default="photo")
    p.add_argument("--filter", choices=FILTERS, default="sunglasses",
                   help="starting AR face filter (photo mode); cycle with n")
    p.add_argument("--particles", type=int, default=6000)
    p.add_argument("--width", type=int, default=960, help="output window width")
    p.add_argument("--height", type=int, default=540, help="output window height")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--no-hands", action="store_true", help="disable all MediaPipe tracking")
    p.add_argument("--hand-model", default=DEFAULT_HAND_MODEL,
                   help="path to MediaPipe hand_landmarker.task")
    p.add_argument("--face-model", default=DEFAULT_FACE_MODEL,
                   help="path to MediaPipe face_landmarker.task")
    p.add_argument("--selftest", action="store_true", help="run headless, write PNGs, exit")
    args = p.parse_args(argv)

    cfg = Cfg(n=args.particles, mode=MODES.index(args.mode), mirror=not args.no_mirror,
              hands=not args.no_hands, filter=FILTERS.index(args.filter))
    if args.selftest:
        return run_selftest(cfg, args.width, args.height)
    return run_live(args.input, cfg, args.width, args.height, args.hand_model, args.face_model)


if __name__ == "__main__":
    raise SystemExit(main())
