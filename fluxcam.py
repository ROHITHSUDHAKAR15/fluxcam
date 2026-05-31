#!/usr/bin/env python3
"""
FluxCam — paint with motion.

Point your webcam at yourself and move: thousands of glowing particles get swept along
by your movement, leaving neon trails. Nothing is pre-recorded — every frame the program
measures how the image is *flowing* (dense optical flow) and pushes the particles with it.

Why it looks alive (the three ideas doing the work):
  1. Dense optical flow (Farneback) turns two consecutive frames into a velocity field:
     for every point, which way and how fast did it move? That field *is* your motion.
  2. A particle system samples that field and advects ~6000 points through it, so the
     particles literally ride your movement. They're colored by the direction they travel,
     so a wave of your hand paints an arc of rainbow.
  3. A fading trail buffer (each frame dimmed, new splats added) gives the glowing,
     long-exposure look instead of flickering dots.

All vectorized in NumPy + OpenCV — no per-particle Python loop — so it runs in real time.

Run it:
    python fluxcam.py                 # webcam, particle mode
    python fluxcam.py --mode flow     # start in dense-flow rainbow mode
    python fluxcam.py --input clip.mp4
    python fluxcam.py --selftest      # headless: render synthetic motion to PNGs

Keys (window focused):
    q/Esc quit   space pause   m mode   c particle colour   x camera ghost
    [ ] fewer/more particles   - = shorter/longer trails    f mirror
    r reset      s save PNG    h help
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

MODES = ["particles", "flow", "ink"]
PARTICLE_COLORS = ["direction", "camera", "ember"]


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
        disp = cv2.addWeighted(disp, 1.0, ghost, 0.12, 0)
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

    def step(self, frame_bgr: np.ndarray, dt: float = 1 / 30) -> np.ndarray:
        fh, fw = frame_bgr.shape[:2]
        self._ensure(fh, fw)
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
    s = f"{MODES[cfg.mode]} | {PARTICLE_COLORS[cfg.pcolor]} | {cfg.n} particles | trail {cfg.decay:.2f} | {fps:4.1f} fps"
    cv2.putText(img, s, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if cfg.show_help:
        for i, line in enumerate(["m mode  c colour  x ghost  f mirror  space pause",
                                  "[ ] particles   - = trails   r reset  s save  q quit"]):
            y = img.shape[0] - 16 - (1 - i) * 22
            cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 220, 255), 1, cv2.LINE_AA)


def handle_key(k: int, cfg: Cfg, eng: "Engine") -> bool:
    if k in (ord("q"), 27):
        return False
    elif k == ord(" "):
        cfg.paused = not cfg.paused
    elif k == ord("m"):
        cfg.mode = (cfg.mode + 1) % len(MODES)
    elif k == ord("c"):
        cfg.pcolor = (cfg.pcolor + 1) % len(PARTICLE_COLORS)
    elif k == ord("x"):
        cfg.ghost = not cfg.ghost
    elif k == ord("f"):
        cfg.mirror = not cfg.mirror
    elif k == ord("h"):
        cfg.show_help = not cfg.show_help
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
def run_live(src: str, cfg: Cfg, out_w: int, out_h: int) -> int:
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("could not open camera/video. Try --input <index|file> or --selftest.", file=sys.stderr)
        return 1

    eng = Engine(cfg, out_w, out_h)
    win = "FluxCam — paint with motion (h = help)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last = None
    fps, t = 0.0, time.time()
    print("FluxCam running — move around! Focus the window, press h for keys, q to quit.")
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
            now = time.time()
            last = eng.step(frame, dt=max(1e-3, now - t))
            inst = 1.0 / max(1e-6, now - t)
            fps = 0.9 * fps + 0.1 * inst if fps else inst
            t = now
        if last is not None:
            shown = last.copy()
            overlay(shown, cfg, fps)
            cv2.imshow(win, shown)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s") and last is not None:
            save_png(last)
        elif k != 255 and not handle_key(k, cfg, eng):
            break
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
    p = argparse.ArgumentParser(description="FluxCam — webcam motion painted as glowing particles.")
    p.add_argument("--input", default="0", help="camera index (default 0) or video file")
    p.add_argument("--mode", choices=MODES, default="particles")
    p.add_argument("--particles", type=int, default=6000)
    p.add_argument("--width", type=int, default=960, help="output window width")
    p.add_argument("--height", type=int, default=540, help="output window height")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--selftest", action="store_true", help="run headless, write PNGs, exit")
    args = p.parse_args(argv)

    cfg = Cfg(n=args.particles, mode=MODES.index(args.mode), mirror=not args.no_mirror)
    if args.selftest:
        return run_selftest(cfg, args.width, args.height)
    return run_live(args.input, cfg, args.width, args.height)


if __name__ == "__main__":
    raise SystemExit(main())
