"""Export utilities for ONNX."""
from __future__ import annotations

import torch

from ..models.student import StudentModel


def export_student(model_path: str, onnx_path: str = "student.onnx") -> None:
    """Export a trained student model to ONNX."""
    model = StudentModel()
    model.eval()
    dummy = torch.zeros(1, 1, 10)
    torch.onnx.export(model, dummy, onnx_path)
    print(f"Exported model to {onnx_path}")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to trained model")
    parser.add_argument("--out", default="student.onnx", help="Output ONNX path")
    args = parser.parse_args()
    export_student(args.model, args.out)
