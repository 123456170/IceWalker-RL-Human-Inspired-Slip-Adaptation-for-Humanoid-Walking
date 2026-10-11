import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageDraw
import time
import math
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="IceWalker RL Live Demo",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEED = 7

OBS_NAMES = [
    "vel_err",
    "lateral_pos",
    "disturbance",
    "mu_hat",
    "mu_trend",
    "phase_sin",
    "phase_cos",
    "contact_left",
    "contact_right",
    "last_slip",
    "stance_norm",
    "stride_norm",
    "prev_a0",
    "prev_a1",
    "prev_a2",
    "human_stance_norm",
    "human_stride_norm",
]

OBS_DIM = len(OBS_NAMES)
ACT_DIM = 3
IDX = {name: i for i, name in enumerate(OBS_NAMES)}


def default_env_cfg(**kwargs):
    cfg = {
        "name": "default",
        "dt": 0.02,
        "steps": 220,
        "cmd_v": 0.8,
        "base_width": 0.24,
        "base_stride": 0.60,
        "mu_base": 0.55,
        "mu_noise": 0.04,
        "patch_prob": 0.25,
        "patch_mu_range": (0.04, 0.13),
        "patch_duration_range": (0.35, 1.10),
        "fall_threshold": 0.48,
        "seed": SEED,
    }
    cfg.update(kwargs)
    return cfg


def default_curriculum():
    return [
        default_env_cfg(
            name="dry_floor",
            mu_base=0.72,
            mu_noise=0.03,
            patch_prob=0.04,
            patch_mu_range=(0.35, 0.55),
            seed=101,
        ),
        default_env_cfg(
            name="variable_friction",
            mu_base=0.45,
            mu_noise=0.06,
            patch_prob=0.18,
            patch_mu_range=(0.15, 0.38),
            seed=102,
        ),
        default_env_cfg(
            name="icy_patches",
            mu_base=0.28,
            mu_noise=0.05,
            patch_prob=0.42,
            patch_mu_range=(0.03, 0.11),
            seed=103,
        ),
    ]


def as_bool_series(series, n=None):
    if series is None:
        return pd.Series(False, index=range(n or 0), dtype=bool)

    s = series.fillna(False).astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "yes", "y", "t"])


def generate_human_mocap(seed=SEED):
    """
    Synthetic human motion capture demo.

    For real deployment, replace this with output from a pose estimator:
    time, left_foot_x, left_foot_y, right_foot_x, right_foot_y,
    or directly provide foot speed / stance width / stride length.
    """
    rng = np.random.default_rng(seed)
    fps = 60
    dur = 10.0
    n = int(fps * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)

    condition = np.where((t >= 4.0) & (t < 7.0), "ice", "normal")

    stance = np.where(condition == "ice", 0.34, 0.24) + rng.normal(0.0, 0.008, n)
    stride = np.where(condition == "ice", 0.45, 0.62) + rng.normal(0.0, 0.012, n)

    step_freq = np.where(condition == "ice", 2.0, 1.8)
    forward_speed = np.where(condition == "ice", 0.62, 0.80)

    phase = np.cumsum(2.0 * np.pi * step_freq / fps)
    pelvis_x = np.cumsum(forward_speed / fps)

    left_x = pelvis_x - 0.16 * np.cos(phase)
    right_x = pelvis_x - 0.16 * np.cos(phase + np.pi)

    left_y = -stance / 2.0 + 0.03 * np.sin(phase)
    right_y = stance / 2.0 + 0.03 * np.sin(phase + np.pi)

    base_left = 0.65 + 0.25 * np.sin(phase)
    base_right = 0.65 + 0.25 * np.sin(phase + np.pi)

    left_speed = np.abs(base_left) + np.abs(rng.normal(0.0, 0.04, n))
    right_speed = np.abs(base_right) + np.abs(rng.normal(0.0, 0.04, n))

    slip_times = np.arange(4.05, 6.95, 0.55)
    for k, s in enumerate(slip_times):
        spike = 2.4 * np.exp(-0.5 * ((t - s) / 0.045) ** 2)
        left_speed += spike * (0.75 + 0.25 * rng.random())
        right_speed += np.roll(spike, 2 + k) * (0.75 + 0.25 * rng.random())

    left_vx = np.gradient(left_x, t)
    left_vy = np.gradient(left_y, t)
    right_vx = np.gradient(right_x, t)
    right_vy = np.gradient(right_y, t)

    df = pd.DataFrame(
        {
            "time": t,
            "condition": condition,
            "pelvis_x": pelvis_x,
            "pelvis_y": np.zeros(n),
            "left_foot_x": left_x,
            "left_foot_y": left_y,
            "right_foot_x": right_x,
            "right_foot_y": right_y,
            "left_foot_vx": left_vx,
            "left_foot_vy": left_vy,
            "right_foot_vx": right_vx,
            "right_foot_vy": right_vy,
            "left_foot_speed": left_speed,
            "right_foot_speed": right_speed,
            "stance_width": stance,
            "stride_length": stride,
        }
    )
    return df


def ensure_motion_columns(df):
    df = df.copy()

    if "time" not in df.columns:
        df["time"] = np.arange(len(df)) / 60.0

    for side in ["left", "right"]:
        speed_col = f"{side}_foot_speed"
        x_col = f"{side}_foot_x"
        y_col = f"{side}_foot_y"

        if speed_col not in df.columns:
            if x_col in df.columns and y_col in df.columns:
                vx = np.gradient(df[x_col], df["time"])
                vy = np.gradient(df[y_col], df["time"])
                df[speed_col] = np.sqrt(vx * vx + vy * vy)
            else:
                df[speed_col] = 0.0

    if "stance_width" not in df.columns:
        if "left_foot_y" in df.columns and "right_foot_y" in df.columns:
            df["stance_width"] = np.abs(df["right_foot_y"] - df["left_foot_y"])
        else:
            df["stance_width"] = 0.24

    if "stride_length" not in df.columns:
        if "left_foot_x" in df.columns:
            df["stride_length"] = np.abs(np.gradient(df["left_foot_x"], df["time"])) * 0.45 + 0.35
        else:
            df["stride_length"] = 0.60

    return df


def detect_slip_events(df, thresh=1.90):
    df = ensure_motion_columns(df)

    left = df.get("left_foot_speed", pd.Series(np.zeros(len(df))))
    right = df.get("right_foot_speed", pd.Series(np.zeros(len(df))))
    speed = np.maximum(left.values, right.values)

    df["max_foot_speed"] = speed
    raw_spike = speed > thresh

    events = np.zeros(len(df), dtype=bool)
    if len(df) > 1:
        events[1:] = raw_spike[1:] & (~raw_spike[:-1])

    df["slip_spike"] = raw_spike
    df["slip_event"] = events

    if "condition" not in df.columns:
        rolling_slip = (
            df["slip_spike"]
            .astype(int)
            .rolling(window=20, min_periods=1)
            .max()
            .astype(bool)
        )
        df["condition"] = np.where(rolling_slip, "ice", "normal")

    return df


@st.cache_resource(show_spinner=False)
def get_human_reference():
    df = detect_slip_events(generate_human_mocap())

    normal = df[df["condition"] == "normal"]
    ice = df[df["condition"] == "ice"]

    normal_stance = float(normal["stance_width"].mean()) if len(normal) else 0.24
    ice_stance = float(ice["stance_width"].mean()) if len(ice) else 0.34
    normal_stride = float(normal["stride_length"].mean()) if len(normal) else 0.62
    ice_stride = float(ice["stride_length"].mean()) if len(ice) else 0.45

    mus = np.array([0.03, 0.08, 0.15, 0.25, 0.40, 0.60, 0.80, 0.95])
    x = np.clip((mus - 0.08) / (0.60 - 0.08), 0.0, 1.0)

    stance_table = ice_stance + (normal_stance - ice_stance) * x
    stride_table = ice_stride + (normal_stride - ice_stride) * x

    # Enforce monotonic human-like behavior:
    # lower mu -> wider stance, shorter stride.
    stance_table = np.maximum.accumulate(stance_table[::-1])[::-1]
    stride_table = np.maximum.accumulate(stride_table)

    return {
        "mus": mus,
        "stance_table": stance_table,
        "stride_table": stride_table,
        "normal_stance": normal_stance,
        "ice_stance": ice_stance,
        "normal_stride": normal_stride,
        "ice_stride": ice_stride,
        "slip_events": int(df["slip_event"].sum()),
    }


def interp_human(mu, table, mus):
    return float(np.interp(np.clip(mu, 0.0, 1.0), mus, table))


def human_stance_ref(mu, ref=None):
    ref = ref or get_human_reference()
    return interp_human(mu, ref["stance_table"], ref["mus"])


def human_stride_ref(mu, ref=None):
    ref = ref or get_human_reference()
    return interp_human(mu, ref["stride_table"], ref["mus"])


class IceWalkerEnv:
    def __init__(self, cfg=None):
        self.cfg = default_env_cfg()
        if cfg is not None:
            self.cfg.update(cfg)
        self.ref = get_human_reference()
        self.rng = np.random.default_rng(self.cfg.get("seed", SEED))
        self.reset()

    def reset(self):
        cfg = self.cfg
        seed = cfg.get("seed")
        self.rng = np.random.default_rng(seed if seed is not None else np.random.randint(0, 2**31 - 1))

        self.t = 0.0
        self.x = 0.0
        self.y = 0.0
        self.vx = cfg["cmd_v"] * 0.92
        self.vy = 0.0
        self.phase = 0.0
        self.disturbance = 0.0
        self.fallen = False
        self.slip_count = 0
        self.last_slip = 0.0

        self.mu_hat = 0.45
        self.prev_mu_hat = 0.45

        self.patch_timer = 0.0
        self.patch_mu = None

        self.contact_left = True
        self.contact_right = True

        self.last_stance = cfg["base_width"]
        self.last_stride = cfg["base_stride"]
        self.last_freq = 1.8
        self.last_action = np.zeros(ACT_DIM)

        self.episode_reward = 0.0
        return self._obs()

    def _sample_mu(self):
        cfg = self.cfg
        dt = cfg["dt"]

        if self.patch_timer > 0.0:
            mu = float(self.patch_mu)
            self.patch_timer -= dt
            return mu

        p = float(cfg["patch_prob"]) * dt
        if self.rng.random() < p:
            dur = float(self.rng.uniform(*cfg["patch_duration_range"]))
            self.patch_timer = dur
            self.patch_mu = float(self.rng.uniform(*cfg["patch_mu_range"]))
            return float(self.patch_mu)

        mu = float(np.clip(cfg["mu_base"] + self.rng.normal(0.0, cfg["mu_noise"]), 0.03, 0.95))
        return mu

    def _obs(self):
        cfg = self.cfg
        h_stance = interp_human(self.mu_hat, self.ref["stance_table"], self.ref["mus"])
        h_stride = interp_human(self.mu_hat, self.ref["stride_table"], self.ref["mus"])

        mu_trend = (self.mu_hat - self.prev_mu_hat) / max(1e-6, cfg["dt"])

        obs = np.array(
            [
                (self.vx - cfg["cmd_v"]) / max(0.1, cfg["cmd_v"]),
                self.y / 0.20,
                self.disturbance / 0.45,
                self.mu_hat,
                np.clip(mu_trend / 4.0, -1.0, 1.0),
                math.sin(self.phase),
                math.cos(self.phase),
                float(self.contact_left),
                float(self.contact_right),
                float(self.last_slip),
                (self.last_stance - cfg["base_width"]) / 0.18,
                (self.last_stride - cfg["base_stride"]) / 0.22,
                float(self.last_action[0]),
                float(self.last_action[1]),
                float(self.last_action[2]),
                (h_stance - cfg["base_width"]) / 0.18,
                (h_stride - cfg["base_stride"]) / 0.22,
            ],
            dtype=float,
        )
        return np.clip(obs, -3.0, 3.0)

    def step(self, action):
        cfg = self.cfg
        dt = cfg["dt"]

        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.last_action = action.copy()

        stance = float(np.clip(cfg["base_width"] * (1.0 + 0.45 * action[0]), 0.14, 0.55))
        stride = float(np.clip(cfg["base_stride"] * (1.0 + 0.30 * action[1]), 0.28, 0.82))
        freq = float(np.clip(1.8 * (1.0 + 0.12 * action[2]), 1.2, 2.5))

        self.last_stance = stance
        self.last_stride = stride
        self.last_freq = freq

        self.phase += 2.0 * np.pi * freq * dt
        mu = self._sample_mu()

        s = math.sin(self.phase)
        self.contact_left = bool(s > -0.22)
        self.contact_right = bool(s < 0.22)
        contact_count = max(1, int(self.contact_left) + int(self.contact_right))

        vel_err = self.vx - cfg["cmd_v"]

        # Dimensionless required friction coefficient.
        tau_coef = (
            0.12 * (stride / cfg["base_stride"]) * (freq / 1.8)
            + 0.22 * abs(vel_err) / max(0.1, cfg["cmd_v"])
            + 0.015 * self.rng.normal()
        )
        tau_coef = max(0.0, float(tau_coef))

        slip_now = bool(tau_coef > mu + 0.005)
        excess = max(0.0, tau_coef - mu)

        if slip_now:
            self.slip_count += 1
            self.last_slip = 1.0
            self.disturbance += excess * 0.85 * (cfg["base_width"] / max(0.10, stance))
            self.vx -= excess * 0.25
            self.vy += self.rng.normal(0.0, 0.08 * excess + 0.01)
        else:
            self.last_slip *= 0.75

        # Online friction estimator surrogate.
        if slip_now:
            obs_mu = float(np.clip(tau_coef * 0.72 + self.rng.normal(0.0, 0.015), 0.02, 0.28))
        else:
            obs_mu = float(np.clip(mu + self.rng.normal(0.0, 0.035), 0.02, 0.95))

        self.prev_mu_hat = self.mu_hat
        self.mu_hat = float(np.clip(0.62 * self.mu_hat + 0.38 * obs_mu, 0.02, 0.95))

        # Simplified base dynamics.
        self.vx += (-0.35 * (self.vx - cfg["cmd_v"]) - 0.25 * self.disturbance * self.rng.normal()) * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy *= 0.88

        recovery_rate = 0.18 * (stance / cfg["base_width"])
        self.disturbance = max(0.0, self.disturbance - recovery_rate * dt)

        fall_limit = cfg["fall_threshold"] * (stance / cfg["base_width"])
        if self.disturbance > fall_limit or abs(self.y) > 0.34:
            self.fallen = True

        h_stance = interp_human(self.mu_hat, self.ref["stance_table"], self.ref["mus"])
        h_stride = interp_human(self.mu_hat, self.ref["stride_table"], self.ref["mus"])

        r = 0.0
        if not self.fallen:
            r += 0.09 * max(0.0, self.vx) * dt
            r -= 0.18 * abs(vel_err) / max(0.1, cfg["cmd_v"]) * dt
            r -= 0.45 * self.disturbance ** 2 * dt
            r -= 0.025 * float(np.sum(np.abs(action))) * dt
            r += 0.28 * math.exp(-((stance - h_stance) / 0.055) ** 2)
            r += 0.20 * math.exp(-((stride - h_stride) / 0.075) ** 2)
            r -= 0.12 * float(slip_now)
        else:
            r -= 3.0

        self.t += dt
        self.episode_reward += r

        done = bool(self.fallen or self.t >= cfg["steps"] * dt)

        info = {
            "mu_true": mu,
            "mu_hat": self.mu_hat,
            "stance": stance,
            "stride": stride,
            "freq": freq,
            "slip": slip_now,
            "fallen": self.fallen,
            "disturbance": self.disturbance,
            "human_stance": h_stance,
            "human_stride": h_stride,
            "contact_count": contact_count,
            "reward": r,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "phase": self.phase,
            "tau_coef": tau_coef,
        }

        return self._obs(), r, done, info


class LinearGaussianPPO:
    """
    Lightweight PPO-style clipped surrogate updater.

    This is intentionally dependency-light so the demo runs immediately.
    For large-scale research, swap this for Stable-Baselines3 / TRL / JAX PPO.
    """

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, seed=SEED):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rng = np.random.default_rng(seed)
        self.feat_dim = obs_dim + 1

        self.W = self.rng.normal(0.0, 0.01, size=(self.feat_dim, act_dim))
        self.log_std = np.full(act_dim, -0.65, dtype=float)

        self.train_history = pd.DataFrame()
        self._init_expert_prior()

    def _init_expert_prior(self):
        # Wider stance when estimated friction is low and human reference is wide.
        self.W[IDX["mu_hat"], 0] += -0.20
        self.W[IDX["human_stance_norm"], 0] += 1.25
        self.W[IDX["disturbance"], 0] += 0.20
        self.W[IDX["last_slip"], 0] += 0.15
        self.W[-1, 0] += 0.05

        # Shorter stride when estimated friction is low and human reference is short.
        self.W[IDX["mu_hat"], 1] += -0.22
        self.W[IDX["human_stride_norm"], 1] += 1.15
        self.W[-1, 1] += -0.02

        # Small step-frequency adaptation.
        self.W[IDX["vel_err"], 2] += 0.10

    def _feat(self, obs):
        obs = np.asarray(obs, dtype=float)
        if obs.ndim == 1:
            return np.concatenate([obs, [1.0]])
        return np.column_stack([obs, np.ones(obs.shape[0], dtype=float)])

    def _log_prob(self, raw, mean, log_std):
        raw = np.asarray(raw, dtype=float)
        mean = np.asarray(mean, dtype=float)
        log_std = np.asarray(log_std, dtype=float)
        var = np.exp(2.0 * log_std)
        return -0.5 * np.sum(
            (raw - mean) ** 2 / var + 2.0 * log_std + math.log(2.0 * math.pi),
            axis=1,
        )

    def act(self, obs):
        f = self._feat(obs).reshape(1, -1)
        mean = (f @ self.W)[0]
        std = np.exp(self.log_std)
        raw = mean + std * self.rng.standard_normal(self.act_dim)
        action = np.clip(raw, -1.0, 1.0)
        logp = float(self._log_prob(raw.reshape(1, -1), mean.reshape(1, -1), self.log_std.reshape(1, -1))[0])
        return action, logp, raw

    def mean_action(self, obs):
        f = self._feat(obs).reshape(1, -1)
        return np.clip((f @ self.W)[0], -1.0, 1.0)

    def update(
        self,
        feats,
        raws,
        old_logps,
        advs,
        lr=0.04,
        eps=0.18,
        epochs=4,
        batch_size=32,
        ent_coef=0.004,
    ):
        feats = np.asarray(feats, dtype=float)
        raws = np.asarray(raws, dtype=float)
        old_logps = np.asarray(old_logps, dtype=float)
        advs = np.asarray(advs, dtype=float)

        n = len(advs)
        if n == 0:
            return {"loss": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}

        advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-6)

        total_loss = 0.0
        total_kl = 0.0
        total_clip = 0.0
        count = 0

        idx_all = np.arange(n)

        for _ in range(epochs):
            self.rng.shuffle(idx_all)

            for start in range(0, n, batch_size):
                mb = idx_all[start : start + batch_size]
                f = feats[mb]
                a = raws[mb]
                old = old_logps[mb]
                adv = advs[mb]

                mean = f @ self.W
                logp = self._log_prob(a, mean, self.log_std)
                ratio = np.exp(np.clip(logp - old, -10.0, 10.0))

                surr1 = ratio * adv
                surr2 = np.clip(ratio, 1.0 - eps, 1.0 + eps) * adv
                loss = -float(np.mean(np.minimum(surr1, surr2)))

                active = ((adv > 0) & (ratio < 1.0 + eps)) | ((adv < 0) & (ratio > 1.0 - eps))

                var = np.exp(2.0 * self.log_std)
                diff = a - mean

                coeff = np.where(active, -ratio * adv, 0.0) / float(len(mb))
                grad_mean = coeff[:, None] * diff / var[None, :]
                grad_W = f.T @ grad_mean

                dlogp_dlogstd = -1.0 + diff ** 2 / var[None, :]
                grad_logstd = np.sum(coeff[:, None] * dlogp_dlogstd, axis=0) - ent_coef

                gw_norm = np.linalg.norm(grad_W)
                if gw_norm > 5.0:
                    grad_W *= 5.0 / gw_norm

                gl_norm = np.linalg.norm(grad_logstd)
                if gl_norm > 2.0:
                    grad_logstd *= 2.0 / gl_norm

                self.W -= lr * grad_W
                self.log_std -= lr * grad_logstd

                kl = float(np.mean(old - logp))
                clipfrac = float(np.mean(np.abs(ratio - 1.0) > eps))

                total_loss += loss
                total_kl += kl
                total_clip += clipfrac
                count += 1

        denom = max(1, count)
        return {
            "loss": total_loss / denom,
            "approx_kl": total_kl / denom,
            "clipfrac": total_clip / denom,
        }


def train_ppo(
    policy,
    curriculum=None,
    n_eps=2,
    max_steps=90,
    progress_bar=None,
    status_text=None,
):
    if curriculum is None:
        curriculum = default_curriculum()

    rows = []
    total = len(curriculum) * n_eps
    done = 0

    for cfg in curriculum:
        env = IceWalkerEnv(cfg)

        for ep in range(n_eps):
            obs = env.reset()

            feats = []
            raws = []
            logps = []
            rew = []
            dones = []
            infos = []

            for _ in range(int(max_steps)):
                action, logp, raw = policy.act(obs)
                feat = policy._feat(obs)

                next_obs, r, done_flag, info = env.step(action)

                feats.append(feat)
                raws.append(raw)
                logps.append(logp)
                rew.append(r)
                dones.append(float(done_flag))
                infos.append(info)

                obs = next_obs
                if done_flag:
                    break

            rew = np.asarray(rew, dtype=float)
            dones = np.asarray(dones, dtype=float)

            returns = np.zeros_like(rew)
            R = 0.0
            for t in range(len(rew) - 1, -1, -1):
                R = rew[t] + 0.99 * R * (1.0 - dones[t])
                returns[t] = R

            adv = returns - np.mean(returns)
            metrics = policy.update(
                np.asarray(feats),
                np.asarray(raws),
                np.asarray(logps),
                adv,
            )

            ep_return = float(np.sum(rew))
            slips = int(sum(bool(x["slip"]) for x in infos))
            fell = bool(any(bool(x["fallen"]) for x in infos))
            if infos:
                match = float(np.mean([abs(x["stance"] - x["human_stance"]) < 0.045 for x in infos]))
            else:
                match = 0.0

            rows.append(
                {
                    "stage": cfg.get("name", "stage"),
                    "episode": ep,
                    "return": ep_return,
                    "slips": slips,
                    "fell": fell,
                    "adaptation_match": match,
                    "loss": metrics["loss"],
                    "approx_kl": metrics["approx_kl"],
                    "clipfrac": metrics["clipfrac"],
                }
            )

            done += 1
            if progress_bar is not None:
                progress_bar.progress(min(1.0, done / max(1, total)))
            if status_text is not None:
                status_text.markdown(
                    f"**{cfg.get('name', 'stage')}** ep {ep + 1}: "
                    f"return={ep_return:.2f}, slips={slips}, match={match:.0%}"
                )

    policy.train_history = pd.DataFrame(rows)
    return policy, policy.train_history


@st.cache_resource(show_spinner="Training IceWalker PPO policy...")
def get_canonical_policy(train=True):
    policy = LinearGaussianPPO(OBS_DIM, ACT_DIM, seed=SEED)
    if train:
        policy, _ = train_ppo(policy, default_curriculum(), n_eps=2, max_steps=85)
    else:
        policy.train_history = pd.DataFrame()
    return policy


def run_episode(policy, cfg, max_steps=None):
    env = IceWalkerEnv(cfg)
    obs = env.reset()

    steps = int(max_steps or cfg.get("steps", 220))
    rows = []

    for i in range(steps):
        action = policy.mean_action(obs)
        obs, r, done, info = env.step(action)

        row = {
            "step": i,
            "time": env.t,
            "action0": float(action[0]),
            "action1": float(action[1]),
            "action2": float(action[2]),
        }
        row.update(info)
        rows.append(row)

        if done:
            break

    return pd.DataFrame(rows)


def fig_to_pil(fig, size=(640, 360), dpi=80):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=None)
    buf.seek(0)
    im = Image.open(buf).convert("RGB").resize(size)
    buf.close()
    return im


def make_episode_frame(df, fi):
    row = df.iloc[fi]
    trail = df.iloc[: fi + 1]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 3.2),
        dpi=80,
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    ax = axes[0]
    mu_norm = float(np.clip(row["mu_true"] / 0.80, 0.0, 1.0))
    ax.set_facecolor((1.0 - mu_norm, 0.25 + 0.55 * mu_norm, 0.85))

    ax.plot(trail["x"], trail["y"], lw=1.0, color="gray", alpha=0.55)

    lx = row["x"] - row["stride"] / 2.0
    rx = row["x"] + row["stride"] / 2.0
    ly = row["y"] - row["stance"] / 2.0
    ry = row["y"] + row["stance"] / 2.0

    ax.scatter([lx, rx], [ly, ry], s=120, c=["#1f77b4", "#d62728"], edgecolors="black", zorder=5)
    ax.scatter(
        [row["x"]],
        [row["y"]],
        s=180,
        marker="s",
        c="green" if not bool(row["fallen"]) else "red",
        edgecolors="black",
        zorder=6,
    )

    if bool(row["slip"]):
        ax.scatter([lx, rx], [ly, ry], s=330, facecolors="none", edgecolors="orange", linewidths=2.2, zorder=7)

    ax.set_xlim(float(row["x"]) - 1.15, float(row["x"]) + 1.15)
    ax.set_ylim(-0.48, 0.48)
    ax.set_aspect("equal")
    ax.set_title("Top-down: stance widens, stride shortens")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.20)

    ax2 = axes[1]
    win = df.iloc[max(0, fi - 70) : fi + 1]

    ax2.plot(win["time"], win["mu_true"], label="true mu", color="black", lw=1.4)
    ax2.plot(win["time"], win["mu_hat"], label="estimated mu", color="tab:blue", lw=1.4)
    ax2.plot(win["time"], win["stance"], label="stance (m)", color="tab:green", lw=1.4)
    ax2.plot(win["time"], win["human_stance"], "--", label="human ref stance", color="tab:orange", lw=1.2)

    ax2.axvline(row["time"], color="red", alpha=0.35)
    if bool(row["slip"]):
        ax2.scatter([row["time"]], [row["mu_hat"]], color="red", zorder=5)

    ax2.set_title("Friction estimate and adaptation")
    ax2.set_xlabel("time (s)")
    ax2.legend(fontsize=6)
    ax2.grid(alpha=0.20)

    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner="Generating live demo media...")
def get_demo_media():
    policy = get_canonical_policy(True)

    cfg = default_env_cfg(
        name="demo_ice",
        mu_base=0.38,
        patch_prob=0.55,
        patch_mu_range=(0.03, 0.10),
        seed=202,
        steps=160,
    )

    df = run_episode(policy, cfg)

    n_frames = 32
    frame_ids = np.linspace(0, max(0, len(df) - 1), n_frames, dtype=int)

    pil_frames = []
    for fi in frame_ids:
        fig = make_episode_frame(df, int(fi))
        pil_frames.append(fig_to_pil(fig))
        plt.close(fig)

    if not pil_frames:
        blank = Image.new("RGB", (640, 360), (20, 20, 20))
        buf = BytesIO()
        blank.save(buf, format="PNG")
        return buf.getvalue(), "image/png", "icewalker_live_demo.png"

    out = BytesIO()

    try:
        pil_frames[0].save(
            out,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=85,
            loop=0,
            optimize=False,
        )
        return out.getvalue(), "image/gif", "icewalker_live_demo.gif"
    except Exception:
        buf = BytesIO()
        pil_frames[len(pil_frames) // 2].save(buf, format="PNG")
        return buf.getvalue(), "image/png", "icewalker_live_demo.png"


def make_synthetic_ice_image(w=640, h=360):
    rng = np.random.default_rng(33)
    img = Image.new("RGB", (w, h), (92, 94, 98))
    d = ImageDraw.Draw(img)

    # Pavement tiles.
    for x in range(0, w, 64):
        for y in range(0, h, 48):
            shade = int(80 + rng.integers(0, 25))
            d.rectangle([x, y, x + 62, y + 46], fill=(shade, shade + 2, shade + 4))

    # Icy patch.
    points = [(170, 105), (480, 95), (520, 245), (145, 260)]
    d.polygon(points, fill=(214, 229, 239))

    # Specular streaks.
    for _ in range(28):
        x0 = int(rng.integers(160, 500))
        y0 = int(rng.integers(105, 245))
        x1 = x0 + int(rng.integers(-55, 55))
        y1 = y0 + int(rng.integers(-18, 18))
        d.line([(x0, y0), (x1, y1)], fill=(245, 250, 255), width=int(rng.integers(1, 3)))

    # Some darker wet edges.
    d.ellipse([90, 245, 260, 330], fill=(70, 78, 86))
    d.ellipse([390, 245, 575, 335], fill=(74, 82, 90))

    return img


def analyze_ice_image(pil_img):
    small = pil_img.convert("RGB").resize((160, 120))
    arr = np.asarray(small, dtype=float) / 255.0

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    brightness = arr.mean(axis=2)
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    sat = (maxc - minc) / (maxc + 1e-6)
    blue_bias = b - 0.5 * (r + g)

    shifted = [
        brightness,
        np.roll(brightness, -2, axis=0),
        np.roll(brightness, 2, axis=0),
        np.roll(brightness, -2, axis=1),
        np.roll(brightness, 2, axis=1),
    ]
    local_std = np.std(np.stack(shifted, axis=0), axis=0)

    ice_mask = (brightness > 0.60) & (sat < 0.22) & (blue_bias > 0.005) & (local_std < 0.13)
    wet_mask = (brightness > 0.42) & (sat < 0.28) & (local_std < 0.16)

    score = float(max(ice_mask.mean(), 0.65 * wet_mask.mean()))
    mu = float(np.clip(0.80 - 0.68 * score, 0.04, 0.85))

    ref = get_human_reference()
    stance = interp_human(mu, ref["stance_table"], ref["mus"])

    return score, mu, stance, ice_mask


def annotate_ice_image(pil_img, score, mu, stance, mask):
    img = pil_img.convert("RGB").copy()
    w, h = img.size
    d = ImageDraw.Draw(img)

    mask_resized = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
    ) > 128

    ys, xs = np.where(mask_resized)
    if len(xs) > 10:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        d.rectangle([x0, y0, x1, y1], outline=(255, 220, 0), width=4)

    cx = w // 2
    cy = int(h * 0.82)
    half = int(np.clip(stance / 0.55, 0.0, 1.0) * w * 0.28)

    d.line([(cx - half, cy), (cx + half, cy)], fill=(0, 255, 120), width=8)
    d.ellipse([cx - half - 10, cy - 10, cx - half + 10, cy + 10], fill=(0, 255, 120))
    d.ellipse([cx + half - 10, cy - 10, cx + half + 10, cy + 10], fill=(0, 255, 120))

    d.rectangle([8, 8, 285, 92], fill=(0, 0, 0))
    d.text((14, 14), f"Ice/wet score: {score:.2f}", fill=(255, 255, 0))
    d.text((14, 36), f"Estimated friction mu: {mu:.2f}", fill=(180, 230, 255))
    d.text((14, 58), f"Predicted stance: {stance:.3f} m", fill=(120, 255, 180))

    return img


def generate_hardware_logs(seed=17):
    rng = np.random.default_rng(seed)
    ref = get_human_reference()

    n = 800
    dt = 0.04
    t = np.arange(n) * dt

    mu_true = np.clip(0.58 + rng.normal(0.0, 0.04, n), 0.08, 0.92)

    for _ in range(6):
        start = int(rng.integers(60, n - 100))
        dur = int(rng.integers(25, 70))
        mu_true[start : start + dur] = rng.uniform(0.04, 0.11)

    mu_hat = np.zeros(n)
    mu_hat[0] = 0.55
    alpha = 0.55

    for i in range(1, n):
        obs = mu_true[i] + rng.normal(0.0, 0.015)
        mu_hat[i] = alpha * mu_hat[i - 1] + (1.0 - alpha) * obs

    stance = np.array([interp_human(m, ref["stance_table"], ref["mus"]) for m in mu_hat])
    stance += rng.normal(0.0, 0.006, n)

    stride = np.array([interp_human(m, ref["stride_table"], ref["mus"]) for m in mu_hat])
    stride += rng.normal(0.0, 0.010, n)

    slip = (mu_hat < 0.13) & (rng.random(n) < 0.22)
    fall = np.zeros(n, dtype=bool)

    slip_idx = np.where(slip)[0]
    for idx in slip_idx:
        if rng.random() < 0.06:
            end = min(n, idx + 25)
            fall[idx:end] = True

    recovery = slip & (~fall)

    df = pd.DataFrame(
        {
            "time": t,
            "mu_true": mu_true,
            "mu_hat": mu_hat,
            "stance_width": stance,
            "stride_length": stride,
            "slip": slip,
            "fall": fall,
            "recovery": recovery,
        }
    )
    return df


def ensure_hardware_columns(df):
    df = df.copy()
    ref = get_human_reference()
    n = len(df)

    if "time" not in df.columns:
        df["time"] = np.arange(n) * 0.04

    if "mu_hat" not in df.columns:
        df["mu_hat"] = 0.5

    if "mu_true" not in df.columns:
        df["mu_true"] = df["mu_hat"]

    if "stance_width" not in df.columns:
        df["stance_width"] = np.array(
            [interp_human(float(m), ref["stance_table"], ref["mus"]) for m in df["mu_hat"]]
        )

    if "stride_length" not in df.columns:
        df["stride_length"] = np.array(
            [interp_human(float(m), ref["stride_table"], ref["mus"]) for m in df["mu_hat"]]
        )

    df["slip"] = as_bool_series(df.get("slip"), n)
    df["fall"] = as_bool_series(df.get("fall"), n)

    if "recovery" not in df.columns:
        df["recovery"] = df["slip"] & (~df["fall"])

    return df


def compute_kpis(df):
    df = ensure_hardware_columns(df)
    ref = get_human_reference()

    slip = df["slip"].astype(bool).reset_index(drop=True)
    fall = df["fall"].astype(bool).reset_index(drop=True)

    prev_slip = slip.shift(1, fill_value=False)
    slip_starts = np.where(slip & (~prev_slip))[0]
    slips = int(len(slip_starts))

    fall_after = 0
    for i in slip_starts:
        window = fall.iloc[i : min(len(fall), i + 30)]
        if bool(window.any()):
            fall_after += 1

    slip_to_fall = float(fall_after / max(1, slips)) if slips > 0 else 0.0
    recovery_success = 1.0 - slip_to_fall if slips > 0 else 1.0

    pred_stance = np.array(
        [interp_human(float(m), ref["stance_table"], ref["mus"]) for m in df["mu_hat"]]
    )
    adaptation_match = float(np.mean(np.abs(df["stance_width"].values - pred_stance) < 0.045))

    if "mu_true" in df.columns and len(df) > 30:
        err = np.abs(df["mu_hat"].values[25:] - df["mu_true"].values[25:])
        est_err = float(np.mean(err))
    else:
        est_err = float("nan")

    return {
        "slips": slips,
        "slip_to_fall": slip_to_fall,
        "recovery_success": recovery_success,
        "adaptation_match": adaptation_match,
        "friction_estimate_error": est_err,
    }


class HardwareBridge:
    """
    Safety-first hardware stub.

    Real deployment should replace this with your robot SDK / ROS 2 bridge.
    Keep this separate from the policy core in production.
    """

    def __init__(self, robot_ip="192.168.0.50", max_speed=0.35):
        self.robot_ip = robot_ip
        self.max_speed = max_speed
        self.connected = False
        self.last_action = None

    def connect(self):
        self.connected = False
        return False, "Simulation-only stub. No physical robot SDK is bundled in this demo app."

    def send_action(self, action):
        if not self.connected:
            return False
        self.last_action = np.asarray(action, dtype=float)
        return True


def run_self_tests():
    results = []

    def add(name, passed, detail=""):
        results.append({"Test": name, "Status": "PASS" if passed else "FAIL", "Detail": detail})

    try:
        ref = get_human_reference()
        add(
            "Human stance widens on ice",
            ref["ice_stance"] > ref["normal_stance"],
            f"{ref['normal_stance']:.3f} m -> {ref['ice_stance']:.3f} m",
        )
        add(
            "Human stride shortens on ice",
            ref["ice_stride"] < ref["normal_stride"],
            f"{ref['normal_stride']:.3f} m -> {ref['ice_stride']:.3f} m",
        )

        df = detect_slip_events(generate_human_mocap())
        add(
            "Slip events detected from foot velocity spikes",
            int(df["slip_event"].sum()) > 0,
            f"{int(df['slip_event'].sum())} synthetic slip events",
        )

        policy = LinearGaussianPPO(OBS_DIM, ACT_DIM, seed=1)

        obs_low = np.zeros(OBS_DIM)
        obs_high = np.zeros(OBS_DIM)

        obs_low[IDX["mu_hat"]] = 0.08
        obs_high[IDX["mu_hat"]] = 0.72

        obs_low[IDX["human_stance_norm"]] = (human_stance_ref(0.08) - 0.24) / 0.18
        obs_high[IDX["human_stance_norm"]] = (human_stance_ref(0.72) - 0.24) / 0.18

        obs_low[IDX["human_stride_norm"]] = (human_stride_ref(0.08) - 0.60) / 0.22
        obs_high[IDX["human_stride_norm"]] = (human_stride_ref(0.72) - 0.60) / 0.22

        a_low = policy.mean_action(obs_low)
        a_high = policy.mean_action(obs_high)

        add(
            "Policy widens stance pre-emptively on low friction estimate",
            a_low[0] > a_high[0] + 0.05,
            f"low-mu action0={a_low[0]:.2f}, high-mu action0={a_high[0]:.2f}",
        )
        add(
            "Policy shortens stride pre-emptively on low friction estimate",
            a_low[1] < a_high[1] - 0.05,
            f"low-mu action1={a_low[1]:.2f}, high-mu action1={a_high[1]:.2f}",
        )

        env = IceWalkerEnv(
            default_env_cfg(
                name="estimator_test",
                patch_prob=0.70,
                patch_mu_range=(0.03, 0.09),
                seed=5,
                steps=180,
            )
        )
        obs = env.reset()
        errs = []
        slips = 0

        for _ in range(180):
            action = policy.mean_action(obs)
            obs, r, done, info = env.step(action)
            errs.append(abs(float(info["mu_hat"]) - float(info["mu_true"])))
            slips += int(bool(info["slip"]))
            if done:
                break

        med_err = float(np.median(errs)) if errs else 99.0
        add(
            "Friction estimator median error below 0.10 on demo sequence",
            med_err < 0.10,
            f"median |mu_hat - mu_true| = {med_err:.3f}, slips={slips}",
        )

        rng = np.random.default_rng(0)
        obs_batch = rng.normal(size=(24, OBS_DIM))
        feats = policy._feat(obs_batch)
        raws = rng.normal(size=(24, ACT_DIM))
        old_logps = policy._log_prob(raws, feats @ policy.W, policy.log_std)
        advs = rng.normal(size=24)

        metrics = policy.update(feats, raws, old_logps, advs, epochs=1)
        add(
            "PPO update produces finite loss",
            bool(np.isfinite(metrics["loss"])),
            f"loss={metrics['loss']:.4f}, kl={metrics['approx_kl']:.4f}",
        )

        img = make_synthetic_ice_image()
        score, mu, stance, mask = analyze_ice_image(img)
        add(
            "Image heuristic detects synthetic icy patch",
            score > 0.15,
            f"score={score:.2f}, mu={mu:.2f}, stance={stance:.3f} m",
        )

        logs = generate_hardware_logs()
        kpis = compute_kpis(logs)
        add(
            "Synthetic hardware slip-to-fall conversion below 10%",
            kpis["slip_to_fall"] < 0.10,
            f"{kpis['slip_to_fall']:.1%}",
        )
        add(
            "Synthetic hardware stance adaptation match above 75%",
            kpis["adaptation_match"] > 0.75,
            f"{kpis['adaptation_match']:.1%}",
        )

    except Exception as e:
        add("Unexpected exception during self-tests", False, str(e))

    return pd.DataFrame(results)


def plot_human_motion(df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(df["time"], df["stance_width"], label="stance width", color="tab:green")
    ice = df["condition"] == "ice"
    ax.scatter(df.loc[ice, "time"], df.loc[ice, "stance_width"], s=8, color="tab:red", label="ice segment")
    ax.set_ylabel("stance (m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(df["time"], df["stride_length"], label="stride length", color="tab:blue")
    ax.set_ylabel("stride (m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.plot(df["time"], df["max_foot_speed"], label="max foot speed", color="black")
    ax.axhline(1.90, color="red", ls="--", label="slip threshold")
    slip_t = df.loc[df["slip_event"], "time"]
    slip_v = df.loc[df["slip_event"], "max_foot_speed"]
    ax.scatter(slip_t, slip_v, color="red", zorder=5, label="detected slip")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("foot speed (m/s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig


def plot_sim_episode(df):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(df["time"], df["mu_true"], label="true friction", color="black")
    ax.plot(df["time"], df["mu_hat"], label="estimated friction", color="tab:blue")
    slip_t = df.loc[df["slip"], "time"]
    slip_y = df.loc[df["slip"], "mu_hat"]
    ax.scatter(slip_t, slip_y, color="red", zorder=5, label="slip")
    ax.set_ylabel("mu")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(df["time"], df["stance"], label="policy stance", color="tab:green")
    ax.plot(df["time"], df["human_stance"], "--", label="human reference", color="tab:orange")
    ax.set_ylabel("stance (m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.plot(df["time"], df["stride"], label="policy stride", color="tab:purple")
    ax.plot(df["time"], df["human_stride"], "--", label="human reference", color="tab:brown")
    ax.set_ylabel("stride (m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[3]
    ax.plot(df["time"], df["disturbance"], label="disturbance", color="red")
    ax.plot(df["time"], df["reward"], label="step reward", color="tab:blue", alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig


def plot_training_history(hist):
    if hist is None or len(hist) == 0:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

    ax = axes[0]
    ax.plot(hist.index, hist["return"], marker="o", label="episode return")
    ax.set_ylabel("return")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(hist.index, hist["adaptation_match"], marker="o", label="stance adaptation match")
    ax.set_xlabel("training update")
    ax.set_ylabel("match")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig


def plot_hardware_logs(df):
    df = ensure_hardware_columns(df)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(df["time"], df["mu_true"], label="true mu", color="black")
    ax.plot(df["time"], df["mu_hat"], label="estimated mu", color="tab:blue")
    ax.set_ylabel("mu")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(df["time"], df["stance_width"], label="stance width", color="tab:green")
    slip_t = df.loc[df["slip"], "time"]
    slip_y = df.loc[df["slip"], "stance_width"]
    ax.scatter(slip_t, slip_y, color="red", label="slip", zorder=5)
    fall_t = df.loc[df["fall"], "time"]
    fall_y = df.loc[df["fall"], "stance_width"]
    ax.scatter(fall_t, fall_y, marker="x", color="purple", label="fall window", zorder=6)
    ax.set_ylabel("stance (m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.plot(df["time"], df["stride_length"], label="stride length", color="tab:purple")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("stride (m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig


def main():
    st.title("IceWalker RL: Human-Inspired Slip Adaptation for Humanoid Walking")
    st.caption(
        "Live synthetic demo: human slip capture → friction-conditioned reference → PPO in randomized-friction simulation → "
        "ice-photo inference → hardware KPI logging. No API key required."
    )

    st.sidebar.header("Controls")

    auto_train = st.sidebar.checkbox("Auto-train canonical PPO policy", value=True, help="Runs a quick NumPy PPO curriculum.")
    cmd_v = st.sidebar.slider("Commanded walking speed (m/s)", 0.30, 1.40, 0.80, 0.05)
    patch_rate = st.sidebar.slider("Low-friction patch rate (/s)", 0.00, 1.00, 0.35, 0.05)
    ice_mu = st.sidebar.slider("Ice patch friction mu", 0.01, 0.30, 0.08, 0.01)
    episode_seed = st.sidebar.number_input("Interactive episode seed", value=42, step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Safety note:** hardware tests must use a harness, spotter, low speed, and emergency stop."
    )

    if "policy" not in st.session_state:
        st.session_state.policy = get_canonical_policy(auto_train)

    if "train_history" not in st.session_state:
        st.session_state.train_history = getattr(st.session_state.policy, "train_history", pd.DataFrame())

    if "test_results" not in st.session_state:
        st.session_state.test_results = run_self_tests()

    tabs = st.tabs(
        [
            "1. Human Capture",
            "2. Sim + PPO",
            "3. Live Demo Video",
            "4. Ice Image Test",
            "5. Hardware Logs + KPIs",
            "6. Tests + Deployment",
        ]
    )

    # ---------------- Tab 1: Human Capture ----------------
    with tabs[0]:
        st.header("1. Capture human walking and slip response")
        st.markdown(
            """
            This tab demonstrates the motion-analysis stage.

            - Synthetic person walks normally, then steps onto an icy patch.
            - Foot velocity spikes are detected as slip events.
            - Stance width and stride length are extracted.
            - The resulting friction-conditioned reference is used by the RL reward.

            For real data, upload a CSV with foot landmarks or foot speeds.
            """
        )

        uploaded_motion = st.file_uploader("Upload human motion CSV (optional)", type=["csv"])

        if uploaded_motion is not None:
            try:
                motion_df = pd.read_csv(uploaded_motion)
                motion_df = detect_slip_events(motion_df)
                st.success("Loaded uploaded motion CSV.")
            except Exception as e:
                st.error(f"Could not parse uploaded CSV: {e}")
                motion_df = detect_slip_events(generate_human_mocap())
        else:
            motion_df = detect_slip_events(generate_human_mocap())
            st.info("Using built-in synthetic human motion demo.")

        ref = get_human_reference()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Normal stance", f"{ref['normal_stance']:.3f} m")
        c2.metric("Ice stance", f"{ref['ice_stance']:.3f} m", delta=f"+{ref['ice_stance'] - ref['normal_stance']:.3f} m")
        c3.metric("Normal stride", f"{ref['normal_stride']:.3f} m")
        c4.metric("Ice stride", f"{ref['ice_stride']:.3f} m", delta=f"{ref['ice_stride'] - ref['normal_stride']:.3f} m")

        st.pyplot(plot_human_motion(motion_df))
        plt.close("all")

        st.subheader("Detected slip events")
        st.dataframe(
            motion_df.loc[motion_df["slip_event"], ["time", "max_foot_speed", "stance_width", "stride_length", "condition"]],
            use_container_width=True,
        )

    # ---------------- Tab 2: Sim + PPO ----------------
    with tabs[1]:
        st.header("2. Randomized-friction simulation and PPO adaptation")
        st.markdown(
            """
            The simulated humanoid receives:
            - commanded velocity,
            - online friction estimate,
            - gait phase,
            - contact state,
            - slip/disturbance history,
            - human stance/stride reference.

            It outputs:
            - stance-width adjustment,
            - stride-length adjustment,
            - small step-frequency adjustment.
            """
        )

        colA, colB = st.columns([1, 2])

        with colA:
            st.subheader("PPO policy")
            if st.button("Retrain PPO on current curriculum"):
                policy = LinearGaussianPPO(OBS_DIM, ACT_DIM, seed=int(time.time() * 1000) % 100000)
                progress = st.progress(0.0)
                status = st.empty()

                curriculum = [
                    default_env_cfg(
                        name="current_slider_ice",
                        mu_base=0.45,
                        patch_prob=patch_rate,
                        patch_mu_range=(max(0.02, ice_mu - 0.04), min(0.25, ice_mu + 0.04)),
                        seed=None,
                    )
                ] + default_curriculum()

                with st.spinner("Training PPO..."):
                    policy, hist = train_ppo(
                        policy,
                        curriculum,
                        n_eps=2,
                        max_steps=90,
                        progress_bar=progress,
                        status_text=status,
                    )

                st.session_state.policy = policy
                st.session_state.train_history = hist

                try:
                    get_demo_media.clear()
                except Exception:
                    pass

                st.success("Retrained policy.")

            hist = st.session_state.get("train_history", pd.DataFrame())
            if len(hist) > 0:
                fig = plot_training_history(hist)
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info("No training history yet. Use the button above or enable auto-train.")

        with colB:
            st.subheader("Validation episode")

            cfg = default_env_cfg(
                name="interactive",
                cmd_v=cmd_v,
                patch_prob=patch_rate,
                patch_mu_range=(max(0.02, ice_mu - 0.04), min(0.25, ice_mu + 0.04)),
                mu_base=0.52,
                seed=int(episode_seed),
                steps=220,
            )

            df = run_episode(st.session_state.policy, cfg)

            st.pyplot(plot_sim_episode(df))
            plt.close("all")

            slip_bool = as_bool_series(df.get("slip"), len(df))
            fall_bool = as_bool_series(df.get("fallen"), len(df))

            slips = int(slip_bool.sum())
            falls = int(bool(fall_bool.any()))
            mean_match = float(np.mean(np.abs(df["stance"] - df["human_stance"]) < 0.045))

            slip_indices = np.where(slip_bool)[0]
            pre_window = 15
            pre_rows = []
            for idx in slip_indices:
                start = max(0, idx - pre_window)
                pre_rows.extend(range(start, idx))

            if pre_rows:
                pre_stance = float(df.iloc[pre_rows]["stance"].mean())
            else:
                pre_stance = float(df["stance"].mean())

            baseline_stance = float(df.loc[~slip_bool, "stance"].mean()) if (~slip_bool).any() else float(df["stance"].mean())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Slips", slips)
            k2.metric("Falls", falls)
            k3.metric("Stance match", f"{mean_match:.0%}")
            k4.metric("Pre-slip widening", f"{pre_stance - baseline_stance:+.3f} m")

            st.markdown(
                """
                **Interpretation:** a positive pre-slip widening value means the policy increased stance width before the slip event,
                using the friction estimator signal rather than waiting for the fall.
                """
            )

    # ---------------- Tab 3: Live Demo Video ----------------
    with tabs[2]:
        st.header("3. Live demo video")
        st.markdown(
            """
            The app auto-generates a looping demo video from a simulated icy-floor episode.

            The video shows:
            - top-down humanoid foot placement,
            - true and estimated friction,
            - policy stance width,
            - human reference stance width,
            - slip markers.
            """
        )

        if st.button("Regenerate live demo media"):
            try:
                get_demo_media.clear()
            except Exception:
                pass
            st.rerun()

        media_bytes, media_mime, media_name = get_demo_media()

        st.image(media_bytes, use_column_width=True)
        st.download_button(
            f"Download live demo media ({media_name.split('.')[-1].upper()})",
            data=media_bytes,
            file_name=media_name,
            mime=media_mime,
        )

        if st.button("Play 4-second live animation in browser"):
            policy = st.session_state.policy
            cfg = default_env_cfg(
                name="live_animation",
                cmd_v=cmd_v,
                patch_prob=patch_rate,
                patch_mu_range=(max(0.02, ice_mu - 0.04), min(0.25, ice_mu + 0.04)),
                mu_base=0.42,
                seed=999,
                steps=120,
            )

            env = IceWalkerEnv(cfg)
            obs = env.reset()
            history = []
            placeholder = st.empty()

            for _ in range(55):
                action = policy.mean_action(obs)
                obs, r, done, info = env.step(action)
                info["time"] = env.t
                history.append(info)

                df_anim = pd.DataFrame(history)
                fig = make_episode_frame(df_anim, len(df_anim) - 1)
                placeholder.pyplot(fig)
                plt.close(fig)

                time.sleep(0.07)
                if done:
                    break

            st.success("Live animation finished.")

    # ---------------- Tab 4: Ice Image Test ----------------
    with tabs[3]:
        st.header("4. Real-image ice/sidewalk test")
        st.markdown(
            """
            Upload a photo of an icy sidewalk or wet tile area.

            This uses a no-API visual heuristic:
            - high brightness,
            - low saturation,
            - bluish tint,
            - low local texture variance.

            It outputs an estimated friction coefficient and predicted stance width.
            """
        )

        uploaded_img = st.file_uploader("Upload icy sidewalk photo", type=["png", "jpg", "jpeg"])

        if uploaded_img is not None:
            try:
                pil_img = Image.open(uploaded_img)
                st.success("Loaded uploaded image.")
            except Exception as e:
                st.error(f"Could not open image: {e}")
                pil_img = make_synthetic_ice_image()
        else:
            pil_img = make_synthetic_ice_image()
            st.info("Using built-in synthetic icy sidewalk image.")

        score, mu, stance, mask = analyze_ice_image(pil_img)
        annotated = annotate_ice_image(pil_img, score, mu, stance, mask)

        c1, c2 = st.columns(2)
        c1.image(pil_img, use_column_width=True, caption="Input image")
        c2.image(annotated, use_column_width=True, caption="Predicted stance overlay")

        m1, m2, m3 = st.columns(3)
        m1.metric("Ice/wet score", f"{score:.2f}")
        m2.metric("Estimated friction mu", f"{mu:.2f}")
        m3.metric("Predicted stance width", f"{stance:.3f} m")

        st.markdown(
            """
            **Note:** for production, replace the heuristic with a trained vision friction estimator or combine it with
            proprioceptive friction estimation during walking.
            """
        )

    # ---------------- Tab 5: Hardware Logs + KPIs ----------------
    with tabs[4]:
        st.header("5. Hardware deployment logs and KPIs")
        st.markdown(
            """
            This tab simulates hardware logs from a wet-tile or low-friction mat test.

            In a real deployment, log:
            - estimated friction,
            - commanded/actual stance width,
            - slip events,
            - fall events,
            - recovery success,
            - harness/spotter status.
            """
        )

        uploaded_log = st.file_uploader("Upload hardware log CSV (optional)", type=["csv"])

        if uploaded_log is not None:
            try:
                logs = pd.read_csv(uploaded_log)
                logs = ensure_hardware_columns(logs)
                st.success("Loaded hardware log CSV.")
            except Exception as e:
                st.error(f"Could not parse log CSV: {e}")
                logs = generate_hardware_logs()
        else:
            logs = generate_hardware_logs()
            st.info("Using built-in synthetic hardware log demo.")

        kpis = compute_kpis(logs)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Slip-to-fall conversion", f"{kpis['slip_to_fall']:.1%}", delta="target < 10%", delta_color="inverse")
        c2.metric("Stance adaptation match", f"{kpis['adaptation_match']:.1%}", delta="target > 75%")
        c3.metric("Friction estimate error", f"{kpis['friction_estimate_error']:.3f}", delta="target < 0.10", delta_color="inverse")
        c4.metric("Recovery success after slip", f"{kpis['recovery_success']:.1%}", delta="target > 85%")

        st.pyplot(plot_hardware_logs(logs))
        plt.close("all")

        st.subheader("Hardware safety checklist")
        st.markdown(
            """
            - [ ] Harness attached and load-tested.
            - [ ] Spotter present within arm's reach.
            - [ ] Emergency stop tested.
            - [ ] Start at low speed, below 0.35 m/s.
            - [ ] Use a removable wet-tile mat or controlled low-friction patch.
            - [ ] Log all slip events and recovery outcomes.
            - [ ] Do not remove harness until 50 consecutive successful recoveries.
            """
        )

        bridge = HardwareBridge(max_speed=0.35)
        ok, msg = bridge.connect()
        st.write(f"Hardware bridge status: **{'connected' if ok else 'not connected'}**")
        st.caption(msg)

    # ---------------- Tab 6: Tests + Deployment ----------------
    with tabs[5]:
        st.header("6. Built-in tests and next deployment step")

        if st.button("Re-run built-in tests"):
            with st.spinner("Running tests..."):
                st.session_state.test_results = run_self_tests()

        test_df = st.session_state.get("test_results", pd.DataFrame())
        st.dataframe(test_df, use_container_width=True)

        if len(test_df) > 0 and (test_df["Status"] == "FAIL").any():
            st.warning("Some tests failed. Inspect the Detail column.")
        else:
            st.success("All built-in checks passed on the synthetic demo path.")

        st.subheader("End-to-end deployment plan")
        st.markdown(
            """
            1. Replace synthetic human mocap with real pose output.
            2. Retrain the human friction-conditioned stance/stride reference.
            3. Fine-tune PPO with your real robot's latency and actuator limits.
            4. Run domain randomization:
               - friction,
               - payload,
               - motor delay,
               - sensor noise,
               - patch duration,
               - surface slope.
            5. Deploy first in simulation with hardware-in-the-loop logging.
            6. Move to physical robot:
               - harness,
               - spotter,
               - low speed,
               - small low-friction mat,
               - incremental speed increases.
            7. Log slip-to-fall conversion, recovery success, stance match, and friction estimation error.
            """
        )

        st.subheader("Confirmation gate")
        st.markdown(
            """
            This package stops before real hardware SDK integration.

            To proceed to the next step, confirm that you want a physical robot bridge for your specific humanoid stack,
            for example:
            - Unitree,
            - Fourier,
            - Booster,
            - Open-source biped,
            - ROS 2 / DDS,
            - MuJoCo / Isaac Lab export.
            """
        )


if __name__ == "__main__":
    main()