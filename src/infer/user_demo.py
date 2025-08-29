"""Single entry point for non-HPC users."""
from __future__ import annotations

from pathlib import Path
import json
import time


def _load_model(model_path: Path):
    """Load an ONNX or TorchScript model depending on availability."""

    if model_path.suffix == ".onnx":
        try:  # pragma: no cover - onnxruntime is optional
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("onnxruntime not available") from exc

        session = ort.InferenceSession(str(model_path))

        def _run(x):
            return session.run(None, {session.get_inputs()[0].name: x})[0]

        return _run
    else:  # assume TorchScript
        try:  # pragma: no cover - torch may be missing
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torch not available") from exc

        module = torch.jit.load(str(model_path))
        module.eval()

        def _run(x):
            with torch.no_grad():
                return module(torch.tensor(x)).numpy()

        return _run


def run_demo(config_path: str) -> None:
    """Load a model and run a simple inference demo."""

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(config_path)
    cfg = json.loads(cfg_path.read_text()) if cfg_path.suffix == ".json" else {}
    model_file = Path(cfg.get("model", "release/student.onnx"))
    runner = _load_model(model_file)

    # Dummy input for demonstration purposes
    import random

    x = [[random.random() for _ in range(10)]]  # 1 x 10
    start = time.time()
    out = runner(x)
    latency = (time.time() - start) * 1000
    print(f"Output: {out}; latency {latency:.2f} ms")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/user_infer.yaml")
    run_demo(parser.parse_args().config)
