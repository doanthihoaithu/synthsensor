import logging
import math
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import multivariate_normal
from datetime import datetime, timedelta


# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def lagged_ema(x, alpha):
    """
    Python equivalent of the R lagged_ema().
    Exponential moving average with lag.

    If alpha = 1, return x unchanged.
    """
    x = np.asarray(x, dtype=float)

    if alpha == 1:
        return x.copy()

    y = np.zeros_like(x)
    y[0] = x[0]

    for t in range(1, len(x)):
        y[t] = (1 - alpha) * y[t - 1] + alpha * x[t]

    return y

def safe_buffer(n, duration, desired=100):
    """
    Python version of safe_buffer().
    """
    max_buff = max(0, math.floor((n - duration) / 2) - 1)
    return min(desired, max_buff)

def safe_buffer_from(start, end, duration, desired=100):
    """
    Python version of safe_buffer().
    """
    max_buff = max(0, math.floor((end + start - duration) / 2) - 1)
    return min(desired + start, max_buff)

def pick_spike_starts(n, k, max_len, buffer=100):
    """
    Python version of pick_spike_starts().
    Returns a list of 1-based start indices.
    """

    if n <= 0 or max_len <= 0:
        return []

    # Conservative min duration = 1
    b = safe_buffer(n, 1, buffer)
    left = max(1, b + 1)
    right = n - max_len - b

    # Fallback: no buffer
    if right < left:
        left = 1
        right = n - max_len
        if right < left:
            return []

    candidates = list(range(left, right + 1))
    if len(candidates) == 0:
        return []

    # Pick up to k unique starts
    return random.sample(candidates, k=min(k, len(candidates)))

def pick_spike_starts_from(start, n, k, max_len, buffer=100):
    """
    Python version of pick_spike_starts().
    Returns a list of 1-based start indices.
    """

    if n <= 0 or max_len <= 0 or start + max_len >= n:
        return []

    # Conservative min duration = 1
    b = safe_buffer(n-start, 1, buffer)
    left = max(start, b + start)
    right = n - max_len - b

    # Fallback: no buffer
    if right < left:
        left = 1
        right = n - max_len
        if right < left:
            return []

    candidates = list(range(left, right + 1))
    if len(candidates) == 0:
        return []

    # Pick up to k unique starts
    return random.sample(candidates, k=min(k, len(candidates)))


def pick_segment(n, duration, buffer=100):
    """
    Python version of pick_segment().
    Returns (start, end) using 1-based indexing, or None if not possible.
    """
    if n <= 0 or duration <= 0 or n < duration:
        return None

    b = safe_buffer(n, duration, buffer)
    left = max(1, b + 1)
    right = n - duration - b + 1

    # If invalid, fall back to no buffer
    if right < left:
        left = 1
        right = n - duration + 1
        if right < left:
            return None

    # Pick uniformly from [left, right]
    start = random.randint(left, right)
    end = start + duration - 1

    return (start, end)

def pick_segment_from(start, end, duration, buffer=100):
    """
    Python version of pick_segment().
    Returns (start, end) using 1-based indexing, or None if not possible.
    """
    if end <= 0 or duration <= 0 or end < duration or start + duration >=end:
        return None

    b = safe_buffer(start - end, duration, buffer)
    left = max(start, b + start)
    right = end - duration - b + 1

    # If invalid, fall back to no buffer
    if right < left:
        left = 1
        right = end - duration + 1
        if right < left:
            return None

    # Pick uniformly from [left, right]
    start = random.randint(left, right)
    end = start + duration - 1

    return (start, end)
# ============================================================
#  Mock helper functions (placeholders)
# ============================================================

# def lagged_ema(arr, alpha):
#     """Mock EMA with lagging. Replace with your actual logic."""
#     out = np.zeros_like(arr, dtype=float)
#     out[0] = arr[0]
#     for i in range(1, len(arr)):
#         out[i] = alpha * arr[i-1] + (1 - alpha) * out[i-1]
#     return out


# def new_flag_vec(n):
#     """Mock flag vector: initialize with empty string lists."""
#     return [""] * n

ANOM_LEVELS = ["Normal", "Drift", "Spike", "Both", "SpikeCorr","DriftCorr"]


def new_flag_vec(n):
    """
    Create a fresh vector of length n with all values = "Normal",
    represented as a pandas Categorical with fixed levels.
    """
    return pd.Categorical(["Normal"] * n, categories=ANOM_LEVELS)


def update_flags(flag_vec, idx, new_tag):
    """
    Update flag_vec at the given indices following overlap rules.

    Parameters
    ----------
    flag_vec : pandas.Categorical
        Vector of anomaly labels.
    idx : array-like or slice
        Indices to update.
    new_tag : str
        One of ANOM_LEVELS.

    Returns
    -------
    pandas.Categorical
    """

    if new_tag not in ANOM_LEVELS:
        raise ValueError(f"new_tag must be one of {ANOM_LEVELS}")

    # Convert to numpy array of strings for manipulation
    cur = flag_vec.astype(str).copy()

    if isinstance(idx, (list, np.ndarray, pd.Index)):
        affected = idx
    else:
        # assume slice
        affected = np.arange(len(cur))[idx]

    # Apply rules
    for i in affected:
        old = cur[i]

        if new_tag in ("Drift", "DriftCorr"):
            if old in ("Spike", "SpikeCorr"):
                cur[i] = "Both"
            elif old == "Normal":
                cur[i] = new_tag

        elif new_tag in ("Spike", "SpikeCorr"):
            if old == "Drift":
                cur[i] = "Both"
            elif old == "Normal":
                cur[i] = new_tag
            # Spike, SpikeCorr, Both remain unchanged

        elif new_tag == "Both":
            cur[i] = "Both"

    # Return re-categorized
    return pd.Categorical(cur, categories=ANOM_LEVELS)


# def pick_spike_starts(n, n_spikes, max_len, buffer):
#     """Mock: randomly choose starting points with minimum spacing."""
#     if n_spikes == 0:
#         return []
#     possible = np.arange(1, n - max_len - buffer)
#     if len(possible) == 0:
#         return []
#     return list(np.random.choice(possible, size=n_spikes, replace=False))


# def update_flags(flags, idx_range, label):
#     """Append anomaly labels to flags."""
#     for i in idx_range:
#         flags[i-1] = (flags[i-1] + ";" + label).strip(";")
#     return flags


# def pick_segment(n, duration, buffer):
#     """Mock segment picker: choose any valid segment."""
#     if duration + buffer >= n:
#         return None
#     start = np.random.randint(1, n - duration - buffer)
#     return (start, start + duration - 1)


# ============================================================
#  Background generator (your provided implementation)
# ============================================================

def generate_background_function(n, background_type,
                                poisson_k=None,
                                poisson_lambda=None,
                                randomwalk_sd=None,
                                background_rho_rw=None,
                                sine_amplitude=None,
                                sine_period=None,
                                background_phi=None,
                                background_rho=None):

    n = max(int(n), 100)

    if background_type == "Poisson Moving Average":
        moving = np.random.poisson(lam=poisson_lambda, size=n + poisson_k)
        # moving average (simple)
        ma = pd.Series(moving).rolling(window=poisson_k).mean().dropna().values
        ma = ma[:n]
        return dict(sensor1=ma, sensor2=ma)

    elif background_type == "Random Walk":
        cov = np.array([[1, background_rho_rw],
                        [background_rho_rw, 1]])
        steps = multivariate_normal(mean=[0, 0], cov=cov).rvs(size=n)
        steps = steps * randomwalk_sd
        s1 = np.cumsum(steps[:, 0])
        s2 = np.cumsum(steps[:, 1])
        return dict(sensor1=s1, sensor2=s2)

    elif background_type == "Sine Wave":
        x = np.arange(1, n + 1)
        sw = sine_amplitude * np.sin(2 * np.pi * x / sine_period)
        return dict(sensor1=sw, sensor2=sw)

    elif background_type == "AR(1) Process":
        s1 = np.zeros(n)
        s2 = np.zeros(n)

        if background_rho == 1:
            init_val = np.random.normal(0, np.sqrt(1 / (1 - background_phi**2)))
            s1[0], s2[0] = init_val, init_val
            eps = np.random.normal(size=n)
            for t in range(1, n):
                s1[t] = background_phi * s1[t-1] + eps[t]
                s2[t] = background_phi * s2[t-1] + eps[t]
        else:
            cov = np.array([[1, background_rho],
                            [background_rho, 1]])
            init_cov = cov / (1 - background_phi**2)
            init_vals = multivariate_normal(mean=[0, 0], cov=init_cov).rvs()
            s1[0], s2[0] = init_vals
            innovations = multivariate_normal(mean=[0, 0], cov=cov).rvs(size=n)
            for t in range(1, n):
                s1[t] = background_phi * s1[t-1] + innovations[t, 0]
                s2[t] = background_phi * s2[t-1] + innovations[t, 1]

        return dict(sensor1=s1, sensor2=s2)

    else:
        raise ValueError("Unknown background_type")


# ============================================================
#  Main dataset generator (python version of your R function)
# ============================================================

def generate_data_function(
        n,
        add_background=True,
        background_type="Poisson Moving Average",
        poisson_k=5,
        poisson_lambda=3,
        background_rho_rw=0,
        randomwalk_sd=0.1,
        sine_amplitude=1,
        sine_period=20,
        background_phi=0.8,
        background_rho=0,
        delayed_sensor="None",
        alpha_ema=0.3,
        sd1=0.1, sd2=0.1,
        crosscor_noise=0,
        mean1=0, mean2=0,
        n_spikes_corr=0,
        n_drift_corr=0,
        n_spikes_s1=0, n_spikes_s2=0,
        max_spike_length=5,
        spike_size=1,
        n_drifts_s1=0, n_drifts_s2=0,
        drift_duration=(20, 40),
        drift_slope=(-0.05, 0.05)
):
    n = max(int(n), 100)

    # Base dataframe
    df = pd.DataFrame({
        "Time": np.arange(1, n + 1),
        "Sensor1": np.zeros(n),
        "Sensor2": np.zeros(n),
    })
    df["Date"] = [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(n)]

    df["AnomalyFlag1"] = new_flag_vec(n)
    df["AnomalyFlag2"] = new_flag_vec(n)

    # Background
    if add_background:
        bg = generate_background_function(
            n=n,
            background_type=background_type,
            poisson_k=poisson_k,
            poisson_lambda=poisson_lambda,
            randomwalk_sd=randomwalk_sd,
            background_rho_rw=background_rho_rw,
            sine_amplitude=sine_amplitude,
            sine_period=sine_period,
            background_phi=background_phi,
            background_rho=background_rho
        )
        s1_bg = np.array(bg["sensor1"])
        s2_bg = np.array(bg["sensor2"])

        if delayed_sensor == "Sensor1":
            df["Sensor1"] = lagged_ema(s1_bg, alpha_ema)
            df["Sensor2"] = s2_bg
        elif delayed_sensor == "Sensor2":
            df["Sensor1"] = s1_bg
            df["Sensor2"] = lagged_ema(s2_bg, alpha_ema)
        else:
            df["Sensor1"] = s1_bg
            df["Sensor2"] = s2_bg

        df["Measurand1"] = s1_bg + mean1
        df["Measurand2"] = s2_bg + mean2
    else:
        df["Measurand1"] = mean1
        df["Measurand2"] = mean2

    # Gaussian noise with correlation
    cov = np.array([
        [sd1**2, crosscor_noise * sd1 * sd2],
        [crosscor_noise * sd1 * sd2, sd2**2]
    ])

    noise = multivariate_normal(mean=[0, 0], cov=cov).rvs(size=n)
    # mean shift
    noise[:, 0] += (mean1 - noise[:, 0].mean())
    noise[:, 1] += (mean2 - noise[:, 1].mean())

    df["Sensor1"] += noise[:, 0]
    df["Sensor2"] += noise[:, 1]

    # Spike magnitude bounds
    min_amp1, min_amp2 = 2*sd1, 2*sd2
    max_amp1 = max(mean1 * spike_size, abs(mean1) * 1)
    max_amp2 = max(mean2 * spike_size, abs(mean2) * 1)

    def draw_amp(minv, maxv):
        return np.random.uniform(minv, maxv)

    # Correlated spikes
    if n_spikes_corr > 0:
        starts = pick_spike_starts(n, n_spikes_corr, max_spike_length, buffer=100)
        for st in starts:
            ln = np.random.randint(1, max_spike_length + 1)
            ed = min(st + ln - 1, n)
            u = np.random.rand()
            amp1 = min_amp1 + u * (max_amp1 - min_amp1)
            amp2 = min_amp2 + u * (max_amp2 - min_amp2)
            sgn = np.random.choice([-1, 1])
            df.loc[st-1:ed-1, "Sensor1"] += sgn * amp1
            df.loc[st-1:ed-1, "Sensor2"] += sgn * amp2
            df["AnomalyFlag1"] = update_flags(df["AnomalyFlag1"], range(st, ed+1), "SpikeCorr")
            df["AnomalyFlag2"] = update_flags(df["AnomalyFlag2"], range(st, ed+1), "SpikeCorr")

    # Uncorrelated spikes sensor 1
    if n_spikes_s1 > 0:
        starts = pick_spike_starts(n, n_spikes_s1, max_spike_length, buffer=100)
        for st in starts:
            ln = np.random.randint(1, max_spike_length + 1)
            ed = min(st + ln - 1, n)
            amp = draw_amp(min_amp1, max_amp1)
            sgn = np.random.choice([-1, 1])
            df.loc[st-1:ed-1, "Sensor1"] += sgn * amp
            df["AnomalyFlag1"] = update_flags(df["AnomalyFlag1"], range(st, ed+1), "Spike")

    # Uncorrelated spikes sensor 2
    if n_spikes_s2 > 0:
        starts = pick_spike_starts(n, n_spikes_s2, max_spike_length, buffer=100)
        for st in starts:
            ln = np.random.randint(1, max_spike_length + 1)
            ed = min(st + ln - 1, n)
            amp = draw_amp(min_amp2, max_amp2)
            sgn = np.random.choice([-1, 1])
            df.loc[st-1:ed-1, "Sensor2"] += sgn * amp
            df["AnomalyFlag2"] = update_flags(df["AnomalyFlag2"], range(st, ed+1), "Spike")

    # Correlated spikes
    if n_drift_corr > 0:
        # duration = np.random.randint(drift_duration[0], drift_duration[1] + 1)
        # starts = pick_spike_starts(n, n_drift_corr, duration, buffer=100)
        for st in range(n_drift_corr):
            duration = np.random.randint(drift_duration[0], drift_duration[1] + 1)
            # duration2 = np.random.randint(drift_duration[0], drift_duration[1] + 1)
            slope = np.random.uniform(drift_slope[0], drift_slope[1])
            seg = pick_segment(n, duration, buffer=100)
            if seg is None:
                continue
            st, ed = seg
            drift = np.linspace(0, slope, duration)
            df.loc[st - 1:ed - 1, "Sensor1"] += drift
            df.loc[st - 1:ed - 1, "Sensor2"] += drift
            df["AnomalyFlag1"] = update_flags(df["AnomalyFlag1"], range(st, ed + 1), "DriftCorr")
            df["AnomalyFlag2"] = update_flags(df["AnomalyFlag2"], range(st, ed + 1), "DriftCorr")

    # Drifts S1
    for _ in range(n_drifts_s1):
        duration = np.random.randint(drift_duration[0], drift_duration[1] + 1)
        slope = np.random.uniform(drift_slope[0], drift_slope[1])
        seg = pick_segment(n, duration, buffer=100)
        if seg is None:
            continue
        st, ed = seg
        drift = np.linspace(0, slope, duration)
        df.loc[st-1:ed-1, "Sensor1"] += drift
        df["AnomalyFlag1"] = update_flags(df["AnomalyFlag1"], range(st, ed+1), "Drift")

    # Drifts S2
    for _ in range(n_drifts_s2):
        duration = np.random.randint(drift_duration[0], drift_duration[1] + 1)
        slope = np.random.uniform(drift_slope[0], drift_slope[1])
        seg = pick_segment(n, duration, buffer=100)
        if seg is None:
            continue
        st, ed = seg
        drift = np.linspace(0, slope, duration)
        df.loc[st-1:ed-1, "Sensor2"] += drift
        df["AnomalyFlag2"] = update_flags(df["AnomalyFlag2"], range(st, ed+1), "Drift")

    df["Diff"] = df["Sensor1"] - df["Sensor2"]

    return df








# Source - https://stackoverflow.com/q
# Posted by Sergio, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-25, License - CC BY-SA 3.0

if __name__ == '__main__':
    # background_type = "AR(1) Process"
    background_type = "Sine Wave"
    df = generate_data_function(
        n=1000,
        add_background=True,
        background_type=background_type,
        sd1=0.1,
        sd2=0.2,
        crosscor_noise=0,
        mean1=10,
        mean2=5,
        n_spikes_corr=10,
        n_spikes_s1=5,
        n_spikes_s2=5,
        max_spike_length=10,
        n_drift_corr=3,
        n_drifts_s1=5,
        n_drifts_s2=5,
        drift_duration=(10, 20),
        drift_slope=(-0.05, 0.05),
        delayed_sensor="None",
        alpha_ema=0.3)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 5))
    axes[0].plot(df["Sensor1"], label="Sensor1")
    axes[0].legend(loc="upper right")
    axes[0].set_xlim(0, df.index.max())
    sensor1_anomalies = df[df["AnomalyFlag1"] != 'Normal']
    for index, anomaly in sensor1_anomalies.iterrows():
        print(index, anomaly['AnomalyFlag1'])
        color = 'red'
        if anomaly['AnomalyFlag1'] == 'Spike':
            color = 'red'
        elif anomaly['AnomalyFlag1'] == 'Drift':
            color = 'orange'
        elif anomaly['AnomalyFlag1'] == 'SpikeCorr':
            color = 'green'
        elif anomaly['AnomalyFlag1'] == 'DriftCorr':
            color = 'blue'
        elif anomaly['AnomalyFlag1'] == 'Both':
            color = 'violet'
        axes[0].axvline(index, color=color, alpha=0.5)

    sensor2_anomalies = df[df["AnomalyFlag2"] != 'Normal']
    axes[1].plot(df["Sensor2"], label="Sensor2")
    axes[1].legend(loc='upper right')
    axes[1].set_xlim(0, df.index.max())
    for index, anomaly in sensor2_anomalies.iterrows():
        print(index, anomaly['AnomalyFlag2'])
        color = 'red'
        if anomaly['AnomalyFlag2'] == 'Spike':
            color = 'red'
        elif anomaly['AnomalyFlag2'] == 'Drift':
            color = 'orange'
        elif anomaly['AnomalyFlag2'] == 'SpikeCorr':
            color = 'green'
        elif anomaly['AnomalyFlag2'] == 'DriftCorr':
            color = 'blue'
        elif anomaly['AnomalyFlag2'] == 'Both':
            color = 'violet'
        axes[1].axvline(index, color=color, alpha=0.5)


    red_patch = mpatches.Patch(color='red', label='Spike')
    orange_patch = mpatches.Patch(color='orange', label='Drift')
    green_patch = mpatches.Patch(color='green', label='SpikeCorr')
    blue_patch = mpatches.Patch(color='blue', label='DriftCorr')
    violet_patch = mpatches.Patch(color='violet', label='Both')
    #
    patches = [red_patch, orange_patch, green_patch, blue_patch, violet_patch]
    # patches = [red_patch]

    fig.suptitle(f'Two sensors of {background_type}',fontsize=16, y=1.08)
    # for patch in patches:
    #     axes[1].add_patch(patch)
    # axes[1].legend(loc="best")

    fig.legend(handles=patches, loc="lower center", ncol=len(patches), bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig('synthetic_sensors.png', bbox_inches='tight')

    # print(df.head())
