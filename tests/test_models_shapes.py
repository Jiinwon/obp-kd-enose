import pytest

torch = pytest.importorskip("torch")

from src.models.teacher import TeacherModel
from src.models.student import StudentModel


def test_teacher_shape():
    model = TeacherModel(hidden_dim=8, prior_dim=4, num_classes=2)
    x = torch.zeros(1, 1, 10)
    P = torch.zeros(2, 4)
    out = model(x, P)
    assert out.shape == (1, 2)


def test_student_shape():
    model = StudentModel(num_classes=2, hidden_dim=8)
    x = torch.zeros(1, 1, 10)
    out = model(x)
    assert out.shape == (1, 2)
