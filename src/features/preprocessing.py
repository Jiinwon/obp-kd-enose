"""Signal preprocessing utilities.

The real project contains a fairly involved preprocessing pipeline operating on
time–series coming from the e‑nose hardware.  For the purposes of the unit
tests in this kata we only require a small, dependency free subset of the
functionality.  Nevertheless the implementation below mirrors the behaviour of
the original code base and can be used as a light‑weight drop in
replacement.

The steps implemented are:

* removal of a warm‑up period
* optional resampling to a target frequency (linear interpolation)
* Hampel based outlier rejection or σ‑clipping
* optional RH/T conditional normalisation
* baseline alignment to the first sample (Δ signal)
* creation of sliding windows with 50 % overlap and augmentation with an
  exponential moving average (EMA) and first differences as additional
  channels

All functions operate on iterables of numbers and return plain Python lists so
that no external dependencies such as :mod:`numpy` are required.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence
import statistics


# ---------------------------------------------------------------------------
# Helpers

def _to_list(x: Iterable) -> List[float]:
    return list(x)


# ---------------------------------------------------------------------------
# Basic transformations

def warmup_cut(x: Iterable, n: int = 5) -> List[float]:
    """Remove an initial warm‑up section from ``x``."""

    return _to_list(x)[n:]


def resample_signal(x: Iterable, orig_rate: float, target_rate: float = 2.0) -> List[float]:
    """Resample ``x`` from ``orig_rate`` to ``target_rate`` using linear interpolation.

    The implementation is intentionally very small and only relies on basic
    Python constructs.  It is sufficient for the unit tests and examples.
    """

    data = _to_list(x)
    if orig_rate == target_rate or not data:
        return data

    duration = len(data) / float(orig_rate)
    new_length = max(int(round(duration * target_rate)), 1)
    result: List[float] = []
    for i in range(new_length):
        t = i / target_rate
        pos = t * orig_rate
        idx0 = int(pos)
        idx1 = min(idx0 + 1, len(data) - 1)
        frac = pos - idx0
        val = data[idx0] * (1 - frac) + data[idx1] * frac
        result.append(val)
    return result


def hampel_filter(x: Iterable, window_size: int = 5, n_sigmas: float = 3.0) -> List[float]:
    """Very small Hampel filter for outlier rejection."""

    data = _to_list(x)
    n = len(data)
    result = data.copy()
    k = max(1, int(window_size))
    for i in range(n):
        start = max(0, i - k)
        end = min(n, i + k + 1)
        window = data[start:end]
        med = statistics.median(window)
        mad = statistics.median([abs(v - med) for v in window])
        if mad == 0:
            continue
        threshold = n_sigmas * 1.4826 * mad
        if abs(data[i] - med) > threshold:
            result[i] = med
    return result


def sigma_clip(x: Iterable, n_sigmas: float = 3.0) -> List[float]:
    """Clip values outside ``n_sigmas`` standard deviations."""

    data = _to_list(x)
    if not data:
        return data
    mean = statistics.fmean(data)
    stdev = statistics.pstdev(data) or 1.0
    lower, upper = mean - n_sigmas * stdev, mean + n_sigmas * stdev
    return [min(max(v, lower), upper) for v in data]


def baseline_delta(x: Iterable) -> List[float]:
    """Align the baseline so that the first sample becomes zero."""

    data = _to_list(x)
    if not data:
        return data
    base = data[0]
    return [v - base for v in data]


# ---------------------------------------------------------------------------
# Conditional normalisation

def _bin_index(val: float, bins: Sequence[float]) -> int:
    for i in range(len(bins) - 1):
        if bins[i] <= val < bins[i + 1]:
            return i
    return len(bins) - 2


def conditional_normalize(
    x: Iterable,
    rh: Iterable,
    temp: Iterable,
    rh_bins: Sequence[float],
    t_bins: Sequence[float],
) -> List[float]:
    """Normalise ``x`` conditioned on RH/T bins."""

    x_list = _to_list(x)
    rh_list = _to_list(rh)
    t_list = _to_list(temp)
    if not (len(x_list) == len(rh_list) == len(t_list)):
        raise ValueError("Input lengths must match")

    stats: dict[tuple[int, int], List[float]] = {}
    for val, r, t in zip(x_list, rh_list, t_list):
        key = (_bin_index(r, rh_bins), _bin_index(t, t_bins))
        stats.setdefault(key, []).append(val)

    means_stds: dict[tuple[int, int], tuple[float, float]] = {}
    for key, vals in stats.items():
        mean = statistics.fmean(vals)
        stdev = statistics.pstdev(vals) or 1.0
        means_stds[key] = (mean, stdev)

    out: List[float] = []
    for val, r, t in zip(x_list, rh_list, t_list):
        key = (_bin_index(r, rh_bins), _bin_index(t, t_bins))
        mean, stdev = means_stds[key]
        out.append((val - mean) / stdev)
    return out


# ---------------------------------------------------------------------------
# Windowing and feature augmentation

def sliding_window(x: Iterable, window_size: int, step: int) -> List[List[float]]:
    """Create sliding windows with a given ``window_size`` and ``step``."""

    data = _to_list(x)
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive")
    windows: List[List[float]] = []
    for start in range(0, max(len(data) - window_size + 1, 1), step):
        end = start + window_size
        if end <= len(data):
            windows.append(data[start:end])
    return windows


def ema(x: Iterable, alpha: float = 0.1) -> List[float]:
    """Exponential moving average of ``x``."""

    data = _to_list(x)
    if not data:
        return data
    out = [data[0]]
    for val in data[1:]:
        out.append(alpha * val + (1 - alpha) * out[-1])
    return out


def first_difference(x: Iterable) -> List[float]:
    """First order difference of ``x`` with the same length."""

    data = _to_list(x)
    if not data:
        return data
    out = [0.0]
    for i in range(1, len(data)):
        out.append(data[i] - data[i - 1])
    return out


def add_ema_and_diff_channels(windows: Sequence[Iterable], alpha: float = 0.1) -> List[List[List[float]]]:
    """Augment each window with EMA and first difference channels.

    The returned structure is ``list`` of windows where each window is a list of
    three channels ``[raw, ema, diff]``.
    """

    processed: List[List[List[float]]] = []
    for win in windows:
        w = _to_list(win)
        processed.append([w, ema(w, alpha), first_difference(w)])
    return processed


# ---------------------------------------------------------------------------
# High level convenience wrapper

def preprocess_signal(
    x: Iterable,
    *,
    rh: Iterable | None = None,
    temp: Iterable | None = None,
    cfg: dict | None = None,
) -> List[List[List[float]]]:
    """Run the full preprocessing pipeline on ``x``.

    The ``cfg`` dictionary can contain the following optional keys:

    ``warmup`` (int)
        Number of initial samples to drop (default ``5``).
    ``orig_rate`` (float)
        Sampling rate of ``x``.  Default is ``len(x)`` which effectively
        disables resampling.
    ``target_rate`` (float)
        Desired sampling rate after resampling (default ``2`` Hz).
    ``window_sec`` (float)
        Duration of a window in seconds (default ``5``).
    ``step_sec`` (float)
        Step size in seconds; defaults to 50 % overlap.
    ``hampel`` (bool)
        Apply the Hampel filter (default ``True``).
    ``sigma_clip`` (float | None)
        If provided, apply σ‑clipping with the given threshold.
    ``rh_bins``/``t_bins`` (sequence)
        Bin edges for conditional normalisation.  ``rh``/``temp`` arrays must be
        supplied as well when using this option.
    ``ema_alpha`` (float)
        Smoothing factor for the EMA (default ``0.1``).
    """

    cfg = cfg or {}
    warm_n = int(cfg.get("warmup", 5))
    orig_rate = float(cfg.get("orig_rate", len(_to_list(x))))
    target_rate = float(cfg.get("target_rate", 2.0))
    window_sec = float(cfg.get("window_sec", 5.0))
    step_sec = float(cfg.get("step_sec", window_sec / 2.0))
    alpha = float(cfg.get("ema_alpha", 0.1))

    data = warmup_cut(x, warm_n)
    data = resample_signal(data, orig_rate, target_rate)
    if cfg.get("hampel", True):
        data = hampel_filter(data)
    if cfg.get("sigma_clip") is not None:
        data = sigma_clip(data, float(cfg["sigma_clip"]))

    if rh is not None and temp is not None and cfg.get("rh_bins") and cfg.get("t_bins"):
        data = conditional_normalize(data, rh, temp, cfg["rh_bins"], cfg["t_bins"])

    data = baseline_delta(data)

    window_size = max(int(round(window_sec * target_rate)), 1)
    step = max(int(round(step_sec * target_rate)), 1)
    windows = sliding_window(data, window_size, step)

    return add_ema_and_diff_channels(windows, alpha=alpha)


__all__ = [
    "warmup_cut",
    "resample_signal",
    "hampel_filter",
    "sigma_clip",
    "conditional_normalize",
    "baseline_delta",
    "sliding_window",
    "ema",
    "first_difference",
    "add_ema_and_diff_channels",
    "preprocess_signal",
]

