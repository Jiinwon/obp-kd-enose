from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import yaml

try:
    from torch.utils.data import Dataset  # type: ignore
except Exception:
    class Dataset:  # type: ignore
        pass


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average(단일 채널)."""
    y = np.empty_like(x, dtype=np.float32)
    if len(x) == 0:
        return y
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = alpha * x[t] + (1.0 - alpha) * y[t - 1]
    return y


def detrend_ema(X: np.ndarray, alpha: float) -> np.ndarray:
    """(C,T) 입력에 대해 채널별 EMA 추세를 빼서 detrend."""
    C, T = X.shape
    Y = np.empty_like(X, dtype=np.float32)
    for c in range(C):
        trend = _ema(X[c], float(alpha))
        Y[c] = X[c] - trend
    return Y


def detrend_poly(X: np.ndarray, deg: int = 1) -> np.ndarray:
    """채널별 poly(최소자승)로 추세 제거."""
    C, T = X.shape
    t = np.arange(T, dtype=np.float32)
    Y = np.empty_like(X, dtype=np.float32)
    for c in range(C):
        coef = np.polyfit(t, X[c], deg=max(0, int(deg)))
        trend = np.polyval(coef, t)
        Y[c] = X[c] - trend
    return Y


def rr0_transform(
    X: np.ndarray,
    *,
    mode: str = "ratio",
    baseline_method: str = "first_k",
    k: int = 50,
) -> np.ndarray:
    """
    R/R0 또는 ΔR 변환. baseline은 세션 내부만 활용(누수 없음).
    X: (C,T)
    """
    C, T = X.shape
    k = max(1, min(int(k), T))
    if baseline_method == "first_k":
        r0 = X[:, :k].mean(axis=1, keepdims=True)
    elif baseline_method == "median_k":
        r0 = np.median(X[:, :k], axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown baseline_method={baseline_method}")
    if mode == "ratio":
        return X / np.clip(r0, 1e-6, None)
    elif mode == "delta":
        return X - r0
    else:
        raise ValueError(f"unknown mode={mode}")


def zscore_standardize(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    mean = mean.astype(np.float32, copy=False)[:, None]
    std = std.astype(np.float32, copy=False)[:, None]
    return (X - mean) / (np.abs(std) + float(eps))


def window_tensor(
    X: np.ndarray,
    *,
    size: int,
    stride: int,
    drop_last: bool = True,
) -> np.ndarray:
    """
    (C,T) -> (N,C,W) 슬라이딩 윈도. 파일(세션) 내부에서만 윈도잉하여 split 경계 누수 방지.
    """
    C, T = X.shape
    W = int(size)
    S = int(stride)
    if T < W:
        return np.empty((0, C, W), dtype=X.dtype)
    idx = list(range(0, T - W + 1, S))
    if not idx:
        return np.empty((0, C, W), dtype=X.dtype)
    if drop_last:
        if idx[-1] != T - W:
            idx = idx[:-1]
            if not idx:
                return np.empty((0, C, W), dtype=X.dtype)
    else:
        if idx[-1] != T - W:
            idx.append(T - W)
    windows = np.stack([X[:, i : i + W] for i in idx], axis=0)
    return windows


@dataclass
class StandardizeState:
    mean: np.ndarray
    std: np.ndarray
    by_device: Dict[Union[str, int], Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


@dataclass
class PreprocessConfig:
    window: Dict[str, Any]
    rr0: Dict[str, Any]
    detrend: Dict[str, Any]
    standardize: Dict[str, Any]
    temp_humid: Dict[str, Any]


class PreprocessPipeline:
    """
    e-nose 전처리 파이프라인
    순서: (선택) temp/humid 선형보정 -> (선택) detrend -> (선택) RR0 -> (선택) standardize -> 윈도잉
    - fit(): train split의 통계(표준화 mean/std 및 선택적 temp/humid 계수)만 추정
    - transform(): 저장된 통계로 변환 (val/test는 학습 통계를 사용; 누수 방지)
    """

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.std_state: Optional[StandardizeState] = None
        # temp/humid 보정 계수: device -> (A,B,C) per-channel 행렬/벡터
        self.th_coeffs: Dict[Union[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @staticmethod
    def from_yaml(path_common: Union[str, Path], path_ds: Union[str, Path]) -> "PreprocessPipeline":
        common = load_yaml(path_common)
        ds = load_yaml(path_ds)
        if "inherit" in ds:
            if str(ds["inherit"]) != str(path_common):
                # 여전히 공통 설정과 병합
                pass
        merged = deep_update(common, ds.get("overrides", {}))
        cfg = PreprocessConfig(
            window=merged["window"],
            rr0=merged["rr0"],
            detrend=merged["detrend"],
            standardize=merged["standardize"],
            temp_humid=merged["temp_humid"],
        )
        return PreprocessPipeline(cfg)

    def _apply_temp_humid(self, X: np.ndarray, aux: Optional[Dict[str, np.ndarray]], device_id: Union[str, int, None]) -> np.ndarray:
        th = self.cfg.temp_humid
        if not th.get("enabled", False):
            return X
        if aux is None:
            return X
        T_arr = aux.get("temp", None)
        H_arr = aux.get("humid", None)
        if T_arr is None and H_arr is None:
            return X
        C, T = X.shape
        t = T_arr if T_arr is not None else np.zeros(T, dtype=np.float32)
        h = H_arr if H_arr is not None else np.zeros(T, dtype=np.float32)

        # 계수 조회(없으면 0으로 처리)
        key = device_id if device_id is not None else "__global__"
        zeros = np.zeros(C, dtype=np.float32)
        A, B, Cc = self.th_coeffs.get(key, (zeros, zeros, zeros))
        # 선형 보정: X' = X - (A*T + B*H + C)
        corr = (A[:, None] * t[None, :]) + (B[:, None] * h[None, :]) + Cc[:, None]
        return X - corr.astype(np.float32)

    def _apply_detrend(self, X: np.ndarray) -> np.ndarray:
        dt = self.cfg.detrend
        if not dt.get("enabled", False):
            return X
        method = dt.get("method", "ema")
        if method == "ema":
            return detrend_ema(X, float(dt.get("ema_alpha", 0.02)))
        elif method == "poly":
            return detrend_poly(X, int(dt.get("poly_deg", 1)))
        else:
            raise ValueError(f"unknown detrend.method={method}")

    def _apply_rr0(self, X: np.ndarray) -> np.ndarray:
        cfg = self.cfg.rr0
        if not cfg.get("enabled", False):
            return X
        return rr0_transform(
            X,
            mode=str(cfg.get("mode", "ratio")),
            baseline_method=str(cfg.get("baseline", {}).get("method", "first_k")),
            k=int(cfg.get("baseline", {}).get("k", 50)),
        )

    def _apply_standardize(self, X: np.ndarray, device_id: Union[str, int, None]) -> np.ndarray:
        st = self.cfg.standardize
        if not st.get("enabled", False):
            return X
        if self.std_state is None:
            raise RuntimeError("standardize state is not fitted. call fit() first.")
        eps = float(st.get("eps", 1e-6))
        if st.get("per_device", False) and device_id in self.std_state.by_device:
            mean, std = self.std_state.by_device[device_id]
        else:
            mean, std = self.std_state.mean, self.std_state.std
        return zscore_standardize(X, mean, std, eps=eps)

    def _collect_stats_from_iter(
        self,
        iterable: Iterable[Tuple[np.ndarray, Dict[str, Any], Any]],
        device_ids: Iterable[Union[str, int, None]],
        aux_provider: Optional[callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[Union[str, int], Tuple[np.ndarray, np.ndarray]]]:
        """
        iterable: (X, y, meta) 시퀀스. X:(C,T)
        device_ids: 각 샘플의 device_id
        aux_provider(meta)->dict: temp/humid 등의 보조 시계열을 제공(없으면 None 허용)
        """
        sums = None
        sums2 = None
        counts = 0
        per_dev: Dict[Union[str, int], Tuple[np.ndarray, np.ndarray, int]] = {}

        st = self.cfg.standardize
        th = self.cfg.temp_humid
        dt = self.cfg.detrend
        rr = self.cfg.rr0

        for (X, _, meta), dev in zip(iterable, device_ids):
            Xp = X.astype(np.float32, copy=False)
            aux = aux_provider(meta) if aux_provider else None
            if th.get("enabled", False):
                Xp = self._apply_temp_humid(Xp, aux, dev)
            if dt.get("enabled", False):
                Xp = self._apply_detrend(Xp)
            if rr.get("enabled", False):
                Xp = self._apply_rr0(Xp)

            # 전체 통계
            csum = Xp.sum(axis=1)  # (C,)
            csum2 = (Xp ** 2).sum(axis=1)
            n = Xp.shape[1]
            if sums is None:
                sums = csum
                sums2 = csum2
            else:
                sums += csum
                sums2 += csum2
            counts += n

            # per-device 통계
            if st.get("per_device", False) and dev is not None:
                if dev not in per_dev:
                    per_dev[dev] = (csum.copy(), csum2.copy(), n)
                else:
                    s0, s20, n0 = per_dev[dev]
                    per_dev[dev] = (s0 + csum, s20 + csum2, n0 + n)

        assert sums is not None and sums2 is not None and counts > 0, "no samples to fit"
        mean = sums / counts
        var = np.maximum(sums2 / counts - mean ** 2, 0.0)
        std = np.sqrt(var + float(st.get("eps", 1e-6)))

        by_dev: Dict[Union[str, int], Tuple[np.ndarray, np.ndarray]] = {}
        for dev, (s0, s20, n0) in per_dev.items():
            m = s0 / n0
            v = np.maximum(s20 / n0 - m ** 2, 0.0)
            by_dev[dev] = (m, np.sqrt(v + float(st.get("eps", 1e-6))))
        return mean.astype(np.float32), std.astype(np.float32), by_dev

    def fit_temp_humid(
        self,
        iterable: Iterable[Tuple[np.ndarray, Dict[str, Any], Any]],
        device_ids: Iterable[Union[str, int, None]],
        aux_provider: Optional[callable],
    ) -> None:
        """단순 선형 보정 계수 학습: 채널별 X ~ a*T + b*H + c (최소자승)."""
        th = self.cfg.temp_humid
        if not th.get("enabled", False):
            return
        self.th_coeffs.clear()
        # 전역/디바이스별 회귀. 여기서는 디바이스별로 학습.
        # 보조 시계열이 없으면 skip.
        cache: Dict[Union[str, int], List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        for (X, _, meta), dev in zip(iterable, device_ids):
            aux = aux_provider(meta) if aux_provider else None
            if aux is None:
                continue
            T_arr = aux.get("temp", None)
            H_arr = aux.get("humid", None)
            if T_arr is None and H_arr is None:
                continue
            C, T = X.shape
            t = T_arr if T_arr is not None else np.zeros(T, dtype=np.float32)
            h = H_arr if H_arr is not None else np.zeros(T, dtype=np.float32)
            one = np.ones(T, dtype=np.float32)
            # 디자인 행렬: [T, H, 1]
            D = np.stack([t, h, one], axis=1)  # (T,3)
            DtD = D.T @ D  # (3,3)
            DtD_inv = np.linalg.pinv(DtD)     # 안정성 위해 pinv
            Dt = D.T
            A = np.empty(C, dtype=np.float32)
            B = np.empty(C, dtype=np.float32)
            Cc = np.empty(C, dtype=np.float32)
            for c in range(C):
                y = X[c].astype(np.float32)
                coef = DtD_inv @ (Dt @ y)  # (3,)
                A[c], B[c], Cc[c] = coef.astype(np.float32)
            key = dev if dev is not None else "__global__"
            cache.setdefault(key, []).append((A, B, Cc))
        # 평균 계수 저장
        for key, lst in cache.items():
            A = np.mean([x[0] for x in lst], axis=0)
            B = np.mean([x[1] for x in lst], axis=0)
            Cc = np.mean([x[2] for x in lst], axis=0)
            self.th_coeffs[key] = (A.astype(np.float32), B.astype(np.float32), Cc.astype(np.float32))

    def fit(
        self,
        train_iterable: Iterable[Tuple[np.ndarray, Dict[str, Any], Any]],
        *,
        train_device_ids: Iterable[Union[str, int, None]],
        aux_provider: Optional[callable] = None,
    ) -> None:
        """train split만으로 모든 통계를 학습."""
        train_list = list(train_iterable)
        dev_list = list(train_device_ids)
        assert len(train_list) == len(dev_list), "train samples and device_ids length mismatch"

        # 1) temp/humid 계수
        self.fit_temp_humid(train_list, dev_list, aux_provider)
        # 2) 표준화 통계
        mean, std, by_dev = self._collect_stats_from_iter(train_list, dev_list, aux_provider)
        self.std_state = StandardizeState(mean=mean, std=std, by_device=by_dev)

    def transform_one(
        self,
        X: np.ndarray,
        *,
        device_id: Union[str, int, None],
        aux: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """단일 세션(파일)의 전처리."""
        Xp = X.astype(np.float32, copy=False)
        Xp = self._apply_temp_humid(Xp, aux, device_id)
        Xp = self._apply_detrend(Xp)
        Xp = self._apply_rr0(Xp)
        Xp = self._apply_standardize(Xp, device_id)
        return Xp

    def transform_and_window(
        self,
        X: np.ndarray,
        *,
        device_id: Union[str, int, None],
        aux: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        Xp = self.transform_one(X, device_id=device_id, aux=aux)
        Wcfg = self.cfg.window
        return window_tensor(Xp, size=int(Wcfg["size"]), stride=int(Wcfg["stride"]), drop_last=bool(Wcfg.get("drop_last", True)))
