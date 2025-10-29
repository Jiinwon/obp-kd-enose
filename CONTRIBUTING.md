# Contributing

## 환경
- Python >= 3.10 (권장: Conda)
- Dev deps 설치:
```bash
pip install -e ".[dev]"
pre-commit install
```

## 개발 흐름
1) 이슈 생성 → 브랜치 생성(`feature/<slug>` 또는 `chore/ci`)
2) 커밋 규칙: Conventional Commits (feat, fix, chore, docs, test 등)
3) 로컬 검사:
```bash
make precommit
make test
```
4) PR 제출: 템플릿 작성, CI 통과 확인

## 테스트
- 테스트는 `tests/`에 배치.
- 빠른 스모크 테스트 예시: `tests/test_smoke.py`.
