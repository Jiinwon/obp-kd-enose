import torch

from src.models.teacher import TeacherModel
from src.models.student import StudentModel


def test_teacher_shape():
    model = TeacherModel(num_classes=2, prior_dim=4)
    x = torch.zeros(1, 1, 10)
    p = torch.zeros(1, 4)
    out = model(x, p)
    assert out.shape == (1, 2)


def test_student_shape():
    model = StudentModel(num_classes=2)
    x = torch.zeros(1, 1, 10)
    out = model(x)
    assert out.shape == (1, 2)
