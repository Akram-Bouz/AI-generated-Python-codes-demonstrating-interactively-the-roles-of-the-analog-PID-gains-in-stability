"""
DRONE HOVER LAB  -  your analog PID (TIPE) keeps a 2D drone hovering
=====================================================================

Run:   python drone_pid_tuner.py          (needs only pygame, which you already have)

WHAT IT SIMULATES
  * The drone's roll angle phi follows the master equation of your dossier:
        phi'' = k * u(t)              with   k = k_u / I_x
        u = Kp*e + Ki*Integral(e) + Kd*de/dt ,   e = phi_ref - phi
    so that, for phi_ref = 0:   phi'' + k*Kd*phi' + k*Kp*phi + k*Ki*Integral(phi) = 0
  * Real-circuit effects that your limitations study talked about:
        - op-amp output saturation (+-V_SAT)         - integrator that can saturate
        - differentiator roll-off (TAU_D)            - sensor noise (amplified by Kd)
        - motor + propeller lag (TAU_MOTOR)
  * "Stationary flight" needs more than the attitude loop, so two simple FIXED
    outer loops (not your circuit) turn position errors into phi_ref and thrust:
        x error -> phi_ref   (so the drone does not drift sideways)
        z error -> total thrust (so it keeps its altitude)
    Your PID circuit is the inner loop that makes phi follow phi_ref.

EVERY TRY (same scenario, so tries are comparable)
  start with the drone tilted 20 deg  ->  recover  ->  a wind gust hits at 3 s
  ->  recover again.  A small constant torque (unbalanced motors) is always present,
  which is what the integral term fights.

THE SCORE (lower is better) is made of: angle error (weighted by how late it is),
position drift, control effort, control jitter (noise through Kd), time spent in
saturation.  Change the weights in section 1 if you want a different "perfect".

All the numbers in section 1 are assumptions: replace them with the values from
your dossier (k, saturation voltage, ...).
"""
import json
import math
import os
import random

import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_FILE = os.path.join(HERE, "best_gains.json")

# =====================================================================
# 1. MODEL PARAMETERS  (edit me)
# =====================================================================
G = 9.81            # m/s^2
MASS = 0.6          # kg
ARM = 0.22          # m, centre -> motor
I_X = 0.012         # kg.m^2, roll inertia
K_PLANT = 6.0       # k = k_u / I_x   [rad/s^2 per volt]
V_SAT = 10.0        # V, saturation of the summing amplifier / integrator
TAU_MOTOR = 0.06    # s, motor + propeller lag
TAU_D = 0.015       # s, roll-off of the real (analog) differentiator
NOISE_STD = 0.003   # rad, sensor noise on phi (about 0.17 deg)
BIAS_ACC = -3.0     # rad/s^2, constant disturbance (unbalanced motors)

# fixed outer loops (NOT your circuit)
X_REF, Z_REF = 0.0, 1.05
KXP, KXD = 1.2, 1.8          # lateral position -> phi_ref
KZP, KZD = 9.0, 5.0          # altitude -> collective thrust
PHI_REF_MAX = 0.35           # rad, outer loop never asks for more than 20 deg
VIEW_WIDTH_M = 2.6           # width of the animation window in metres
VIEW_HALF_WIDTH = 1.2        # beyond this the drone counts as 'flew away'

# scenario
T_END = 8.0
DT = 0.004
PHI0 = 0.35                  # rad, initial tilt (20 deg)
GUST_T, GUST_DUR, GUST_ACC = 3.0, 0.25, -14.0   # s, s, rad/s^2

# score weights
W_ANG, W_POS, W_EFF, W_JIT, W_SAT = 60.0, 12.0, 20.0, 2.0, 100.0
CRASH_BASE, CRASH_PER_S = 300.0, 40.0
SETTLE_BAND = math.radians(2.0)

# gain ranges (sliders and tuner use the same ones)
KP_RANGE, KI_RANGE, KD_RANGE = (0.0, 25.0), (0.0, 60.0), (0.0, 12.0)
START_GAINS = (2.0, 0.0, 0.2)      # deliberately poor: the "before" picture
MAX_TRIES = 45


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# =====================================================================
# 2. SIMULATION
# =====================================================================
class Flight:
    """One simulated try: time series plus metrics."""


def simulate(kp, ki, kd, noise=True, gust=True):
    rng = random.Random(7)               # same noise every try -> fair comparison
    n = int(T_END / DT)
    a_f = 1.0 - math.exp(-DT / TAU_D)
    a_m = 1.0 - math.exp(-DT / TAU_MOTOR)

    x, z, vx, vz = X_REF, Z_REF, 0.0, 0.0
    phi, om = PHI0, 0.0
    integ, xf, u_act, u_prev = 0.0, None, 0.0, 0.0
    f = Flight()
    f.gains = (kp, ki, kd)
    f.x, f.z, f.phi, f.u, f.tl, f.tr = [], [], [], [], [], []
    j_ang = j_pos = j_eff = j_jit = sat = 0.0
    crashed, why, t_crash = False, "", T_END

    for i in range(n):
        t = i * DT
        # ---- sensor + fixed outer loops
        phi_meas = phi + (rng.gauss(0.0, NOISE_STD) if noise else 0.0)
        phi_ref = clamp(-(KXP * (X_REF - x) - KXD * vx) / G, -PHI_REF_MAX, PHI_REF_MAX)
        thrust = MASS * (G + KZP * (Z_REF - z) - KZD * vz) / max(math.cos(phi), 0.5)
        thrust = clamp(thrust, 0.0, 2.2 * MASS * G)

        # ---- YOUR ANALOG PID: subtractor -> P, I, D branches -> summing amp
        e = phi_ref - phi_meas
        if xf is None:
            xf = e
        xf += a_f * (e - xf)
        d = (e - xf) / TAU_D                       # filtered derivative
        integ = clamp(integ + ki * e * DT, -V_SAT, V_SAT)
        u = clamp(kp * e + integ + kd * d, -V_SAT, V_SAT)

        # ---- actuator + plant (phi'' = k u + disturbances)
        u_act += a_m * (u - u_act)
        g_acc = GUST_ACC if (gust and GUST_T <= t < GUST_T + GUST_DUR) else 0.0
        om += (K_PLANT * u_act + BIAS_ACC + g_acc) * DT
        phi += om * DT
        vx += -thrust * math.sin(phi) / MASS * DT
        vz += (thrust * math.cos(phi) / MASS - G) * DT
        x += vx * DT
        z += vz * DT

        # ---- log
        dT = I_X * K_PLANT * u_act / ARM           # T_right - T_left
        f.x.append(x); f.z.append(z); f.phi.append(phi); f.u.append(u)
        f.tl.append(max(0.0, thrust / 2 - dT / 2)); f.tr.append(max(0.0, thrust / 2 + dT / 2))
        t_ev = t - GUST_T if (gust and t >= GUST_T) else t
        j_ang += (1.0 + 2.0 * t_ev) * abs(phi) * DT
        j_pos += (abs(x - X_REF) + abs(z - Z_REF)) * DT
        j_eff += (u / V_SAT) ** 2 * DT
        j_jit += abs(u - u_prev) / V_SAT
        sat += DT if abs(u) >= V_SAT - 1e-9 else 0.0
        u_prev = u

        if z <= 0.0:
            crashed, why, t_crash = True, "CRASHED", t
            break
        if abs(x - X_REF) > VIEW_HALF_WIDTH:
            crashed, why, t_crash = True, "FLEW AWAY", t
            break

    f.n = len(f.phi)
    f.duration = f.n * DT
    f.crashed, f.why = crashed, why
    if crashed:
        f.score = CRASH_BASE + CRASH_PER_S * (T_END - t_crash)
    else:
        f.score = (W_ANG * j_ang + W_POS * j_pos + W_EFF * j_eff
                   + W_JIT * (j_jit / T_END) + W_SAT * sat / T_END * 10.0)
    f.parts = dict(ang=W_ANG * j_ang, pos=W_POS * j_pos, eff=W_EFF * j_eff,
                   jit=W_JIT * j_jit / T_END, sat=W_SAT * sat / T_END * 10.0)

    # ---- readable metrics
    t_gust = GUST_T if gust else T_END
    k_gust = int(t_gust / DT)
    last_bad1 = 0
    for i in range(min(k_gust, f.n)):
        if abs(f.phi[i]) > SETTLE_BAND:
            last_bad1 = i
    f.settle = last_bad1 * DT
    f.recover = None
    if gust and not crashed:
        last_bad2 = k_gust
        for i in range(k_gust, f.n):
            if abs(f.phi[i]) > SETTLE_BAND:
                last_bad2 = i
        f.recover = (last_bad2 - k_gust) * DT
    f.drift = max(abs(v - X_REF) for v in f.x) if f.x else 0.0
    tail = f.x[int(f.n * 0.85):] or f.x
    f.final_err = max(abs(v - X_REF) for v in tail) if tail else 0.0
    f.peak_u = max(abs(v) for v in f.u) if f.u else 0.0
    f.perfect = (not crashed and f.settle <= 1.8 and (f.recover is None or f.recover <= 1.8)
                 and f.final_err <= 0.05 and sat <= 0.05)
    # decimated curve for plotting (every 8 steps = 32 ms)
    f.plot_pts = [(i * DT, math.degrees(f.phi[i])) for i in range(0, f.n, 8)]
    return f


# =====================================================================
# 3. AUTOMATIC GAIN SEARCH
# =====================================================================
class Tuner:
    """Try 1 = the current gains, then random exploration, then a (1+1)
    evolution strategy that mutates the best gains found so far."""
    N_EXPLORE = 9
    SIGMA0, UP, DOWN = 0.12, 1.3, 0.85

    def __init__(self, start, seed=11):
        self.rng = random.Random(seed)
        self.n = 0
        self.best_s, self.best_score = None, float("inf")
        self.sigma = self.SIGMA0
        self.history = []                      # best score after each try
        self.start_s = self.to_s(start)
        # Latin hypercube for the exploration phase
        cuts = []
        for _ in range(3):
            c = [(k + self.rng.random()) / self.N_EXPLORE for k in range(self.N_EXPLORE)]
            self.rng.shuffle(c)
            cuts.append(c)
        self.explore = [tuple(cuts[a][k] for a in range(3)) for k in range(self.N_EXPLORE)]
        self.last_s = None
        self.last_phase = "explore"

    # normalised coordinates s in [0,1]^3  <->  gains
    @staticmethod
    def to_gains(s):
        kp = max(0.3, KP_RANGE[1]) ** s[0] * 0.3 ** (1 - s[0])
        return (kp, KI_RANGE[1] * s[1] ** 2, KD_RANGE[1] * s[2] ** 2)

    @staticmethod
    def to_s(g):
        kp = clamp(g[0], 0.3, KP_RANGE[1])
        return (math.log(kp / 0.3) / math.log(KP_RANGE[1] / 0.3),
                math.sqrt(clamp(g[1] / KI_RANGE[1], 0, 1)),
                math.sqrt(clamp(g[2] / KD_RANGE[1], 0, 1)))

    def ask(self):
        """-> (gains, phase) or None when the search is finished."""
        if self.n >= MAX_TRIES or self.converged():
            return None
        if self.n == 0:
            s, phase = self.start_s, "explore"
        elif self.n <= self.N_EXPLORE:
            s, phase = self.explore[self.n - 1], "explore"
        else:
            s = tuple(clamp(b + self.sigma * self.rng.gauss(0, 1), 0.0, 1.0)
                      for b in self.best_s)
            phase = "refine"
        self.last_s, self.last_phase = s, phase
        return self.to_gains(s), phase

    def tell(self, score):
        improved = score < self.best_score * (1 - 1e-4)
        if self.last_phase == "refine":
            self.sigma = clamp(self.sigma * (self.UP if improved else self.DOWN), 0.012, 0.35)
        if improved:
            self.best_score, self.best_s = score, self.last_s
        self.history.append(self.best_score)
        self.n += 1

    def converged(self):
        h = self.history
        if len(h) < self.N_EXPLORE + 12:
            return False
        stalled = h[-1] > h[-13] * 0.995          # < 0.5 % better over 12 tries
        return stalled and self.sigma < 0.08


# =====================================================================
# 4. SMALL UI TOOLKIT
# =====================================================================
C_BG = (13, 17, 23)
C_PANEL = (21, 27, 35)
C_PANEL2 = (30, 38, 49)
C_LINE = (50, 62, 78)
C_GRID = (36, 46, 59)
C_TEXT = (224, 231, 239)
C_MUTED = (128, 141, 157)
C_AMBER = (255, 179, 71)
C_GREEN = (61, 220, 151)
C_BLUE = (90, 169, 255)
C_RED = (255, 93, 115)
C_VIOLET = (178, 140, 255)
C_CYAN = (80, 220, 230)

KIND_COLOR = {"explore": C_VIOLET, "refine": C_CYAN, "manual": C_BLUE}


def lerp_color(a, b, t):
    t = clamp(t, 0.0, 1.0)
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def make_font(size, bold=False):
    try:
        return pygame.font.SysFont("segoeui,arial,helvetica,dejavusans", size, bold=bold)
    except Exception:
        return pygame.font.Font(None, size + 6)


def text(surf, s, font, color, pos, anchor="topleft"):
    img = font.render(s, True, color)
    r = img.get_rect()
    setattr(r, anchor, pos)
    surf.blit(img, r)
    return r


def panel(surf, rect, title=None, font=None):
    pygame.draw.rect(surf, C_PANEL, rect, border_radius=10)
    pygame.draw.rect(surf, C_LINE, rect, width=1, border_radius=10)
    if title:
        text(surf, title, font, C_MUTED, (rect.x + 14, rect.y + 8))


class Slider:
    STEP = 0.05

    def __init__(self, track, name, hint, units, vmin, vmax, value, color):
        self.track = pygame.Rect(track)
        self.name, self.hint, self.units = name, hint, units
        self.vmin, self.vmax, self.value, self.color = vmin, vmax, value, color
        self.drag = False

    def frac(self):
        return (self.value - self.vmin) / (self.vmax - self.vmin)

    def hit(self, pos):
        return self.track.inflate(26, 30).collidepoint(pos)

    def set_from_x(self, x):
        f = clamp((x - self.track.x) / self.track.w, 0.0, 1.0)
        v = self.vmin + f * (self.vmax - self.vmin)
        self.value = clamp(round(v / self.STEP) * self.STEP, self.vmin, self.vmax)

    def draw(self, surf, fonts, enabled):
        t = self.track
        col = self.color if enabled else C_MUTED
        text(surf, self.name, fonts["l"], C_TEXT if enabled else C_MUTED, (t.x - 80, t.centery - 16))
        text(surf, self.hint, fonts["xs"], C_MUTED, (t.x - 80, t.centery + 5))
        pygame.draw.rect(surf, C_PANEL2, t, border_radius=3)
        fill = pygame.Rect(t.x, t.y, int(t.w * self.frac()), t.h)
        pygame.draw.rect(surf, col, fill, border_radius=3)
        kx = int(t.x + t.w * self.frac())
        pygame.draw.circle(surf, C_BG, (kx, t.centery), 11)
        pygame.draw.circle(surf, col, (kx, t.centery), 9)
        pygame.draw.circle(surf, C_TEXT, (kx, t.centery), 4)
        text(surf, f"{self.value:5.2f}", fonts["l"], C_TEXT if enabled else C_MUTED,
             (t.right + 22, t.centery - 16))
        text(surf, self.units, fonts["xs"], C_MUTED, (t.right + 22, t.centery + 5))


class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label

    def draw(self, surf, font, mouse, active=False, enabled=True, accent=C_BLUE, label=None):
        hover = enabled and self.rect.collidepoint(mouse)
        if active:
            bg, fg, br = accent, C_BG, accent
        else:
            bg = C_PANEL2 if not hover else (40, 51, 66)
            fg = C_TEXT if enabled else C_MUTED
            br = C_LINE if not hover else accent
        pygame.draw.rect(surf, bg, self.rect, border_radius=7)
        pygame.draw.rect(surf, br, self.rect, width=1, border_radius=7)
        text(surf, label or self.label, font, fg, self.rect.center, "center")


class Axes:
    """A plot area with a frame, grid and tick labels."""

    def __init__(self, rect, title, xlim, ylim, ylog=False):
        self.rect = pygame.Rect(rect)
        self.title, self.xlim, self.ylim, self.ylog = title, xlim, ylim, ylog
        r = self.rect
        self.inner = pygame.Rect(r.x + 50, r.y + 30, r.w - 50 - 16, r.h - 30 - 30)

    def _ty(self, y):
        return math.log10(max(y, 1e-9)) if self.ylog else y

    def map(self, x, y, local=False):
        fx = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        y0, y1 = self._ty(self.ylim[0]), self._ty(self.ylim[1])
        fy = (self._ty(y) - y0) / (y1 - y0)
        px = fx * self.inner.w
        py = self.inner.h - fy * self.inner.h
        if local:
            return (px, py)
        return (self.inner.x + px, self.inner.y + py)

    def frame(self, surf, fonts, xticks, yticks, xlabel="", ylabel=""):
        panel(surf, self.rect, self.title, fonts["m"])
        for v, lab in yticks:
            _, py = self.map(self.xlim[0], v)
            pygame.draw.line(surf, C_GRID, (self.inner.x, py), (self.inner.right, py), 1)
            text(surf, lab, fonts["xs"], C_MUTED, (self.inner.x - 6, py), "midright")
        for v, lab in xticks:
            px, _ = self.map(v, self.ylim[0])
            pygame.draw.line(surf, C_GRID, (px, self.inner.y), (px, self.inner.bottom), 1)
            text(surf, lab, fonts["xs"], C_MUTED, (px, self.inner.bottom + 4), "midtop")
        pygame.draw.rect(surf, C_LINE, self.inner, width=1)
        if xlabel:
            text(surf, xlabel, fonts["xs"], C_MUTED, (self.inner.centerx, self.rect.bottom - 4), "midbottom")
        if ylabel:
            img = fonts["xs"].render(ylabel, True, C_MUTED)
            img = pygame.transform.rotate(img, 90)
            surf.blit(img, img.get_rect(midleft=(self.rect.x + 6, self.inner.centery)))


class Try:
    def __init__(self, idx, kind, flight):
        self.idx, self.kind, self.flight = idx, kind, flight
        self.gains, self.score = flight.gains, flight.score


# =====================================================================
# 5. THE APPLICATION
# =====================================================================
W, H = 1200, 700
VIEW = pygame.Rect(16, 16, 540, 400)
CTRL = pygame.Rect(16, 428, 540, 256)
PLOT_A = pygame.Rect(572, 16, 612, 236)
PLOT_B = pygame.Rect(572, 260, 612, 204)
PLOT_C = pygame.Rect(572, 472, 612, 212)


class App:
    def __init__(self, screen):
        self.screen = screen
        self.F = {"xs": make_font(12), "s": make_font(13), "m": make_font(15),
                  "l": make_font(19, True), "xl": make_font(28, True)}
        self.PHI = "\u03c6" if all((f.metrics("\u03c6") or [None])[0] is not None for f in self.F.values()) else "phi"
        self.t_anim = 0.0
        self.speed = 4
        self.opts = {"noise": True, "gust": True}
        self.auto = False
        self.tuner = None
        self.tries, self.best = [], None
        self.saved_best = self.load_best()
        self.banner = False
        self.final_pending = False
        self.hold = 0.0
        self.status = "Move a slider (release to fly), or press AUTO-TUNE."
        self.flight, self.cur, self.play_t, self.playing = None, None, 0.0, False
        self.wheel_timer = None

        tx = CTRL.x + 96
        ys = (CTRL.y + 52, CTRL.y + 94, CTRL.y + 136)
        self.sliders = [
            Slider((tx, ys[0] - 3, 300, 6), "Kp", "proportional", "V / rad", *KP_RANGE, START_GAINS[0], C_BLUE),
            Slider((tx, ys[1] - 3, 300, 6), "Ki", "integral", "V / (rad.s)", *KI_RANGE, START_GAINS[1], C_AMBER),
            Slider((tx, ys[2] - 3, 300, 6), "Kd", "derivative", "V.s / rad", *KD_RANGE, START_GAINS[2], C_GREEN),
        ]
        by = CTRL.y + 164
        bx = CTRL.x + 16
        self.btn = {
            "auto": Button((bx, by, 160, 34), "AUTO-TUNE"),
            "fly": Button((bx + 170, by, 84, 34), "FLY"),
            "best": Button((bx + 264, by, 84, 34), "BEST"),
            "reset": Button((bx + 358, by, 84, 34), "RESET"),
        }
        by2 = CTRL.y + 210
        self.btn["s1"] = Button((bx + 62, by2, 40, 28), "x1")
        self.btn["s2"] = Button((bx + 108, by2, 40, 28), "x2")
        self.btn["s4"] = Button((bx + 154, by2, 40, 28), "x4")
        self.btn["s8"] = Button((bx + 200, by2, 40, 28), "x8")
        self.btn["noise"] = Button((bx + 262, by2, 112, 28), "Noise")
        self.btn["gust"] = Button((bx + 384, by2, 112, 28), "Wind gust")

        self.axA = Axes(PLOT_A, f"Roll angle {self.PHI}(t) of every try   (green = better flights, amber = flying now)",
                        (0, T_END), (-40, 40))
        self.axB = Axes(PLOT_B, "Score of every try   (lower = better, log scale)", (0, MAX_TRIES + 1), (10, 1000), ylog=True)
        self.axC = Axes(PLOT_C, "Gains of every try   (% of slider range)", (0, MAX_TRIES + 1), (0, 100))
        self.layerA = None
        self.sky = self.make_sky()
        self.rebuild_layers()
        self.start_try(self.gains(), "manual")

    # ------------------------------------------------------------ helpers
    def gains(self):
        return tuple(s.value for s in self.sliders)

    def set_gains(self, g):
        for s, v in zip(self.sliders, g):
            s.value = clamp(v, s.vmin, s.vmax)

    def load_best(self):
        try:
            with open(BEST_FILE) as fh:
                d = json.load(fh)
            return (float(d["Kp"]), float(d["Ki"]), float(d["Kd"]))
        except Exception:
            return None

    def save_best(self, g, score):
        try:
            with open(BEST_FILE, "w") as fh:
                json.dump({"Kp": round(g[0], 3), "Ki": round(g[1], 3),
                           "Kd": round(g[2], 3), "score": round(score, 2)}, fh, indent=2)
        except Exception:
            pass

    def make_sky(self):
        s = pygame.Surface(VIEW.size)
        for y in range(VIEW.h):
            s.fill(lerp_color((14, 22, 36), (31, 47, 68), y / VIEW.h), (0, y, VIEW.w, 1))
        return s

    # ------------------------------------------------------------ flights
    def start_try(self, gains, kind):
        f = simulate(*gains, noise=self.opts["noise"], gust=self.opts["gust"])
        self.flight = f
        self.cur = Try(len(self.tries) + 1, kind, f)
        self.play_t, self.playing = 0.0, True

    def finish_try(self):
        self.playing = False
        cur = self.cur
        if cur.kind == "final":
            self.auto, self.final_pending, self.banner = False, False, True
            self.status = "Done. Play with the sliders, or press AUTO-TUNE to search again."
            self.saved_best = cur.gains
            self.save_best(cur.gains, cur.score)
            print(f"Best gains: Kp={cur.gains[0]:.2f}  Ki={cur.gains[1]:.2f}  Kd={cur.gains[2]:.2f}"
                  f"   (score {cur.score:.1f}, saved to best_gains.json)")
            return
        self.tries.append(cur)
        if self.best is None or cur.score < self.best.score:
            self.best = cur
        self.rebuild_layers()
        if self.auto and self.tuner is not None and cur.kind != "manual":
            self.tuner.tell(cur.score)
            self.hold = 0.45

    def next_auto(self):
        ask = self.tuner.ask()
        if ask is None:
            self.final_pending = True
            self.set_gains(self.best.gains)
            self.status = "Search finished. Flying the best gains..."
            self.start_try(self.best.gains, "final")
            return
        gains, phase = ask
        self.set_gains(gains)
        n = self.tuner.n + 1
        self.status = (f"Try {n}/{MAX_TRIES}: " +
                       ("exploring random gain sets..." if phase == "explore"
                        else "refining around the best gains..."))
        self.start_try(gains, phase)

    def start_auto(self):
        self.tuner = Tuner(self.gains())
        self.tries, self.best, self.banner = [], None, False
        self.rebuild_layers()
        self.auto, self.final_pending = True, False
        self.playing, self.hold, self.wheel_timer = False, 0.0, None

    def stop_auto(self):
        self.auto, self.final_pending, self.tuner = False, False, None
        self.status = "Auto-tune stopped."

    # ------------------------------------------------------------ events
    def handle(self, ev):
        locked = self.auto
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for s in self.sliders:
                if not locked and s.hit(ev.pos):
                    s.drag = True
                    s.set_from_x(ev.pos[0])
                    self.banner = False
            for name, b in self.btn.items():
                if b.rect.collidepoint(ev.pos):
                    self.click(name)
        elif ev.type == pygame.MOUSEMOTION:
            for s in self.sliders:
                if s.drag:
                    s.set_from_x(ev.pos[0])
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            if any(s.drag for s in self.sliders):
                for s in self.sliders:
                    s.drag = False
                self.fly_manual()
        elif ev.type == pygame.MOUSEWHEEL and not locked:
            pos = pygame.mouse.get_pos()
            for s in self.sliders:
                if s.hit(pos):
                    s.value = clamp(s.value + ev.y * (s.vmax - s.vmin) / 100.0, s.vmin, s.vmax)
                    self.wheel_timer = 0.6
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE and self.playing:
                self.play_t = self.flight.duration
            elif ev.key == pygame.K_a:
                self.click("auto")
            elif ev.key == pygame.K_f:
                self.click("fly")

    def fly_manual(self):
        self.banner = False
        self.status = "Manual flight with your gains."
        self.start_try(self.gains(), "manual")

    def click(self, name):
        if name == "auto":
            if self.auto:
                self.stop_auto()
            else:
                self.start_auto()
        elif name == "fly" and not self.auto:
            self.fly_manual()
        elif name == "best" and not self.auto:
            g = self.saved_best or (self.best.gains if self.best else None)
            if g:
                self.set_gains(g)
                self.fly_manual()
        elif name == "reset":
            self.auto, self.tuner, self.final_pending, self.banner = False, None, False, False
            self.tries, self.best = [], None
            self.set_gains(START_GAINS)
            self.rebuild_layers()
            self.fly_manual()
            self.status = "Reset."
        elif name in ("s1", "s2", "s4", "s8"):
            self.speed = int(name[1:])
        elif name in ("noise", "gust") and not self.auto:
            self.opts[name] = not self.opts[name]
            self.fly_manual()

    # ------------------------------------------------------------ update
    def update(self, dt):
        self.t_anim += dt
        if self.wheel_timer is not None and not self.auto:
            self.wheel_timer -= dt
            if self.wheel_timer <= 0:
                self.wheel_timer = None
                self.fly_manual()
        if self.playing:
            spd = 1.0 if self.cur.kind == "final" else float(self.speed)
            self.play_t += dt * spd
            if self.play_t >= self.flight.duration:
                self.play_t = self.flight.duration
                self.finish_try()
        elif self.auto and not self.final_pending:
            self.hold -= dt
            if self.hold <= 0:
                self.next_auto()

    # ------------------------------------------------------------ layers
    def rebuild_layers(self):
        ax = self.axA
        layer = pygame.Surface(ax.inner.size, pygame.SRCALPHA)
        n = len(self.tries)
        order = sorted(range(n), key=lambda i: -self.tries[i].score)   # worst first
        rank = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: self.tries[i].score))}
        for i in order:
            tr = self.tries[i]
            q = 1.0 if n < 2 else 1.0 - rank[i] / (n - 1)
            col = lerp_color(C_RED, C_GREEN, q) + (110,)
            pts = [ax.map(t, d, local=True) for t, d in tr.flight.plot_pts]
            if len(pts) > 1:
                pygame.draw.lines(layer, col, False, pts, 1)
        if self.best is not None:
            pts = [ax.map(t, d, local=True) for t, d in self.best.flight.plot_pts]
            pygame.draw.lines(layer, C_GREEN + (255,), False, pts, 3)
        self.layerA = layer

    # ------------------------------------------------------------ drawing
    def draw(self):
        s = self.screen
        s.fill(C_BG)
        self.draw_view()
        self.draw_controls()
        self.draw_plot_a()
        self.draw_plot_b()
        self.draw_plot_c()

    # ---- drone view
    def world(self, X, Z):
        sc = VIEW.w / VIEW_WIDTH_M
        return (VIEW.centerx + (X - X_REF) * sc, VIEW.bottom - 40 - Z * sc)

    def draw_view(self):
        s, F = self.screen, self.F
        s.blit(self.sky, VIEW)
        s.set_clip(VIEW)
        sc = VIEW.w / VIEW_WIDTH_M
        gy = VIEW.bottom - 40
        for k in range(-3, 4):                         # grid every 0.5 m
            gx = VIEW.centerx + k * 0.5 * sc
            pygame.draw.line(s, (36, 52, 74), (gx, VIEW.y), (gx, gy), 1)
        for k in range(0, 5):
            yy = gy - k * 0.5 * sc
            pygame.draw.line(s, (36, 52, 74), (VIEW.x, yy), (VIEW.right, yy), 1)
            if 0 < k <= 2:
                text(s, f"{k * 0.5:.1f} m", F["xs"], (86, 106, 132), (VIEW.x + 8, yy - 2), "bottomleft")
        pygame.draw.rect(s, (38, 46, 56), (VIEW.x, gy, VIEW.w, 40))
        pygame.draw.line(s, (92, 108, 126), (VIEW.x, gy), (VIEW.right, gy), 2)
        for k in range(0, VIEW.w, 22):
            pygame.draw.line(s, (52, 62, 74), (VIEW.x + k, gy + 2), (VIEW.x + k - 12, gy + 40), 1)

        f = self.flight
        idx = min(int(self.play_t / DT), f.n - 1)
        # target
        tx, ty = self.world(X_REF, Z_REF)
        tcol = (140, 110, 60)
        pygame.draw.circle(s, tcol, (tx, ty), int(0.06 * sc), 1)
        pygame.draw.line(s, tcol, (tx - 0.45 * sc, ty), (tx + 0.45 * sc, ty), 1)
        pygame.draw.line(s, tcol, (tx, ty - 0.35 * sc), (tx, ty + 0.35 * sc), 1)
        text(s, "hover target", F["xs"], (170, 135, 72), (tx - 0.47 * sc, ty - 3), "bottomright")
        # trail
        for j in range(0, idx, 10):
            px, py = self.world(f.x[j], f.z[j])
            a = j / max(idx, 1)
            pygame.draw.circle(s, lerp_color((40, 56, 80), C_AMBER, a * a), (px, py), 2)
        # drone
        cx, cy = self.world(f.x[idx], f.z[idx])
        self.draw_drone(s, cx, cy, sc, f.phi[idx], f.tl[idx], f.tr[idx], self.t_anim)
        # gust marker
        t_now = idx * DT
        if self.opts["gust"] and GUST_T <= t_now < GUST_T + GUST_DUR + 0.15:
            ax0, ax1 = cx + 0.40 * sc, cx + 0.70 * sc
            for k in range(3):
                yy = cy - 20 + k * 20
                pygame.draw.line(s, C_AMBER, (ax0 + 10, yy), (ax1, yy), 3)
                pygame.draw.polygon(s, C_AMBER, [(ax0, yy), (ax0 + 12, yy - 6), (ax0 + 12, yy + 6)])
            text(s, "GUST", F["m"], C_AMBER, ((ax0 + ax1) / 2, cy - 38), "midbottom")
        s.set_clip(None)
        pygame.draw.rect(s, C_LINE, VIEW, width=1, border_radius=4)

        # HUD
        label = self.cur.kind if self.cur else ""
        n = self.cur.idx if self.cur else 0
        title = {"explore": f"TRY {n}  -  exploring", "refine": f"TRY {n}  -  refining",
                 "manual": "MANUAL FLIGHT", "final": "BEST GAINS - FINAL FLIGHT"}.get(label, "")
        text(s, title, F["l"], C_TEXT, (VIEW.x + 14, VIEW.y + 10))
        g = f.gains
        text(s, f"Kp {g[0]:.2f}   Ki {g[1]:.2f}   Kd {g[2]:.2f}", F["m"], C_MUTED, (VIEW.x + 14, VIEW.y + 36))
        done = (not self.playing)
        if done:
            sc_txt = f"score {f.score:.1f}"
            if self.best:
                sc_txt += f"   best {self.best.score:.1f}"
            text(s, sc_txt, F["m"], C_TEXT, (VIEW.right - 14, VIEW.y + 12), "topright")
            if f.crashed:
                text(s, f.why, F["l"], C_RED, (VIEW.right - 14, VIEW.y + 34), "topright")
        u = f.u[idx]
        text(s, f"t {t_now:4.2f} s    {self.PHI} {math.degrees(f.phi[idx]):+6.1f}\u00b0    "
                f"x {f.x[idx] - X_REF:+5.2f} m    z {f.z[idx]:4.2f} m    u {u:+5.1f} V",
             F["s"], C_MUTED, (VIEW.x + 14, VIEW.bottom - 24))
        if self.playing:
            text(s, "SPACE = skip", F["xs"], C_MUTED, (VIEW.right - 14, VIEW.bottom - 24), "topright")

        if self.banner and not self.playing:
            self.draw_banner(s, f)

    def draw_banner(self, s, f):
        F = self.F
        r = pygame.Rect(VIEW.x + 28, VIEW.y + 238, VIEW.w - 56, 100)
        box = pygame.Surface(r.size, pygame.SRCALPHA)
        box.fill((10, 44, 30, 235) if f.perfect else (52, 40, 12, 235))
        s.blit(box, r)
        col = C_GREEN if f.perfect else C_AMBER
        pygame.draw.rect(s, col, r, width=2, border_radius=6)
        text(s, "PERFECT GAINS FOUND" if f.perfect else "BEST GAINS FOUND", F["xl"], col,
             (r.centerx, r.y + 8), "midtop")
        g = f.gains
        text(s, f"Kp = {g[0]:.2f}     Ki = {g[1]:.2f}     Kd = {g[2]:.2f}", F["l"], C_TEXT,
             (r.centerx, r.y + 44), "midtop")
        rec = f"{f.recover:.1f} s" if f.recover is not None else "-"
        text(s, f"settles in {f.settle:.1f} s  |  gust recovery {rec}  |  max drift {f.drift * 100:.0f} cm  |  peak {f.peak_u:.1f} V",
             F["s"], C_MUTED, (r.centerx, r.y + 76), "midtop")

    def draw_drone(self, s, cx, cy, sc, phi, tl, tr, t):
        c, sn = math.cos(phi), math.sin(phi)

        def P(px, pz):
            return (cx + (px * c - pz * sn) * sc, cy - (px * sn + pz * c) * sc)

        thr0 = MASS * G / 2
        for side, thr in ((-1, tl), (1, tr)):
            ln = (0.015 + 0.085 * clamp(thr / thr0, 0.0, 2.2)) * (1 + 0.12 * math.sin(t * 70 + side * 2))
            x0 = side * ARM
            pygame.draw.polygon(s, (255, 140, 40), [P(x0 - 0.028, -0.04), P(x0 + 0.028, -0.04), P(x0, -0.04 - ln)])
            pygame.draw.polygon(s, (255, 226, 140), [P(x0 - 0.014, -0.04), P(x0 + 0.014, -0.04), P(x0, -0.04 - ln * 0.6)])
        for side in (-1, 1):                                              # legs
            pygame.draw.line(s, (120, 133, 150), P(side * 0.07, -0.04), P(side * 0.10, -0.10), 3)
            pygame.draw.line(s, (120, 133, 150), P(side * 0.075, -0.10), P(side * 0.125, -0.10), 3)
        pygame.draw.line(s, (150, 162, 178), P(-ARM, 0.02), P(ARM, 0.02), 5)       # arms
        for side in (-1, 1):
            x0 = side * ARM
            pygame.draw.polygon(s, (92, 106, 126), [P(x0 - 0.025, 0.05), P(x0 + 0.025, 0.05),
                                                    P(x0 + 0.025, -0.04), P(x0 - 0.025, -0.04)])
            w = 0.045 + 0.05 * abs(math.cos(t * 38 + side))
            pygame.draw.line(s, C_CYAN, P(x0 - w, 0.078), P(x0 + w, 0.078), 4)
            pygame.draw.line(s, (92, 106, 126), P(x0, 0.05), P(x0, 0.078), 3)
        body = [P(-0.10, 0.045), P(0.10, 0.045), P(0.125, 0.0), P(0.10, -0.045), P(-0.10, -0.045), P(-0.125, 0.0)]
        pygame.draw.polygon(s, (226, 233, 243), body)
        pygame.draw.polygon(s, (120, 133, 150), body, 2)
        pygame.draw.line(s, C_BLUE, P(-0.07, 0.015), P(0.07, 0.015), 3)
        pygame.draw.circle(s, C_RED, P(0.0, -0.015), 4)

    # ---- controls
    def draw_controls(self):
        s, F = self.screen, self.F
        panel(s, CTRL, "ANALOG PID GAINS", F["m"])
        text(s, "drag - mouse wheel for fine steps - release to fly" if not self.auto else "auto-tune is choosing the gains",
             F["xs"], C_MUTED, (CTRL.right - 14, CTRL.y + 10), "topright")
        for sl in self.sliders:
            sl.draw(s, F, enabled=not self.auto)
        mouse = pygame.mouse.get_pos()
        b = self.btn
        b["auto"].draw(s, F["m"], mouse, active=self.auto, accent=C_AMBER,
                       label="STOP" if self.auto else "AUTO-TUNE")
        b["fly"].draw(s, F["m"], mouse, enabled=not self.auto)
        b["best"].draw(s, F["m"], mouse, enabled=not self.auto and (self.saved_best or self.best) is not None)
        b["reset"].draw(s, F["m"], mouse)
        text(s, "Speed", F["s"], C_MUTED, (CTRL.x + 16, CTRL.y + 217))
        for name, v in (("s1", 1), ("s2", 2), ("s4", 4), ("s8", 8)):
            b[name].draw(s, F["m"], mouse, active=(self.speed == v))
        b["noise"].draw(s, F["s"], mouse, active=self.opts["noise"], enabled=not self.auto, accent=C_GREEN,
                        label="Noise: ON" if self.opts["noise"] else "Noise: off")
        b["gust"].draw(s, F["s"], mouse, active=self.opts["gust"], enabled=not self.auto, accent=C_GREEN,
                       label="Gust: ON" if self.opts["gust"] else "Gust: off")
        text(s, self.status, F["xs"], C_MUTED, (CTRL.x + 16, CTRL.bottom - 4), "bottomleft")

    # ---- plots
    def draw_plot_a(self):
        s, F, ax = self.screen, self.F, self.axA
        ax.frame(s, F, [(t, f"{t:g}") for t in range(0, int(T_END) + 1)],
                 [(v, f"{v:+d}" if v else "0") for v in range(-40, 41, 20)], "time (s)", "angle (\u00b0)")
        band_top = ax.map(0, math.degrees(SETTLE_BAND))[1]
        band_bot = ax.map(0, -math.degrees(SETTLE_BAND))[1]
        band = pygame.Surface((ax.inner.w, int(band_bot - band_top)), pygame.SRCALPHA)
        band.fill((61, 220, 151, 28))
        s.blit(band, (ax.inner.x, band_top))
        if self.opts["gust"]:
            gx, _ = ax.map(GUST_T, 0)
            pygame.draw.line(s, (110, 90, 50), (gx, ax.inner.y), (gx, ax.inner.bottom), 1)
            text(s, "gust", F["xs"], (150, 120, 66), (gx + 4, ax.inner.y + 3))
        s.blit(self.layerA, ax.inner)
        s.set_clip(ax.inner)
        f = self.flight
        if f is not None:
            k = min(int(self.play_t / DT) // 8 + 1, len(f.plot_pts))
            pts = [ax.map(t, d) for t, d in f.plot_pts[:k]]
            if len(pts) > 1:
                pygame.draw.lines(s, C_AMBER, False, pts, 3)
            if pts:
                pygame.draw.circle(s, C_AMBER, pts[-1], 5)
        s.set_clip(None)
        text(s, "\u00b12\u00b0 band", F["xs"], (90, 170, 130), (ax.inner.right - 6, ax.inner.y + 3), "topright")

    def draw_plot_b(self):
        s, F, ax = self.screen, self.F, self.axB
        scores = [t.score for t in self.tries]
        lo = 10 ** math.floor(math.log10(max(min(scores) * 0.8, 1e-3))) if scores else 10
        hi = 10 ** math.ceil(math.log10(max(max(scores) * 1.05, lo * 10))) if scores else 1000
        ax.ylim = (lo, hi)
        ax.xlim = (0, max(MAX_TRIES, len(self.tries)) + 1)
        decades = []
        v = lo
        while v <= hi * 1.001:
            decades.append((v, f"{v:g}"))
            v *= 10
        step = 5 if ax.xlim[1] <= 45 else 10
        ax.frame(s, F, [(k, str(k)) for k in range(0, int(ax.xlim[1]) + 1, step)], decades, "try number", "score")
        s.set_clip(ax.inner)
        best, pts_best = float("inf"), []
        for i, tr in enumerate(self.tries):
            best = min(best, tr.score)
            pts_best.append(ax.map(i + 1, best))
        if len(pts_best) > 1:
            pygame.draw.lines(s, C_GREEN, False, pts_best, 2)
        for i, tr in enumerate(self.tries):
            p = ax.map(i + 1, tr.score)
            pygame.draw.circle(s, KIND_COLOR.get(tr.kind, C_BLUE), p, 4)
        if self.best is not None:
            bi = self.tries.index(self.best)
            p = ax.map(bi + 1, self.best.score)
            pygame.draw.circle(s, C_GREEN, p, 8, 2)
        s.set_clip(None)
        lx = ax.inner.right - 6
        for name, col in (("manual", C_BLUE), ("refine", C_CYAN), ("explore", C_VIOLET)):
            r = text(s, name, F["xs"], C_MUTED, (lx, ax.inner.y + 5), "topright")
            pygame.draw.circle(s, col, (r.x - 8, r.centery), 4)
            lx = r.x - 22

    def draw_plot_c(self):
        s, F, ax = self.screen, self.F, self.axC
        ax.xlim = (0, max(MAX_TRIES, len(self.tries)) + 1)
        step = 5 if ax.xlim[1] <= 45 else 10
        ax.frame(s, F, [(k, str(k)) for k in range(0, int(ax.xlim[1]) + 1, step)],
                 [(v, f"{v}%") for v in (0, 25, 50, 75, 100)], "try number", "gain")
        ranges = (KP_RANGE, KI_RANGE, KD_RANGE)
        cols = (C_BLUE, C_AMBER, C_GREEN)
        names = ("Kp", "Ki", "Kd")
        s.set_clip(ax.inner)
        for gi in range(3):
            pts = []
            for i, tr in enumerate(self.tries):
                lo, hi = ranges[gi]
                pts.append(ax.map(i + 1, 100.0 * (tr.gains[gi] - lo) / (hi - lo)))
            if len(pts) > 1:
                pygame.draw.lines(s, cols[gi], False, pts, 2)
            for p in pts:
                pygame.draw.circle(s, cols[gi], p, 3)
        s.set_clip(None)
        g = self.best.gains if self.best else self.gains()
        lx = ax.inner.right - 6
        for gi in (2, 1, 0):
            r = text(s, f"{names[gi]} {g[gi]:.2f}", F["s"], cols[gi], (lx, ax.inner.y + 5), "topright")
            lx = r.x - 14
        text(s, "best:" if self.best else "now:", F["xs"], C_MUTED, (lx, ax.inner.y + 7), "topright")


def main():
    pygame.init()
    flags = pygame.SCALED | pygame.RESIZABLE
    try:
        screen = pygame.display.set_mode((W, H), flags)
    except pygame.error:
        screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Drone hover lab - analog PID (TIPE)")
    app = App(screen)
    clock = pygame.time.Clock()
    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            else:
                app.handle(ev)
        app.update(dt)
        app.draw()
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
