.PHONY: setup train-teacher train-student eval user-demo

setup:
	python3 -m venv .venv || true
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

train-teacher:
	python -m src.train.train_teacher --config configs/train_teacher.yaml

train-student:
	python -m src.train.train_student --config configs/train_student.yaml

eval:
	python -m src.eval.evaluate --config configs/eval.yaml

user-demo:
	bash scripts/user_run_local.sh --input sample.csv
