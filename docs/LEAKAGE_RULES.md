# 누수 방지 규칙(전처리 단계)

1) **학습 통계 분리**  
- 표준화(mean/std), 온습도 보정 계수 등은 **train split**에서만 추정한다.  
- `PreprocessPipeline.fit()` 이후에는 통계를 고정(freeze)하고 `transform()`에서 재추정 금지.

2) **세션 경계 준수**  
- 윈도잉은 **파일(세션) 내부에서만** 수행한다. 서로 다른 파일/세션 사이를 넘나드는 윈도 금지.

3) **시간 순서 보존**  
- forward-chaining 실험에서는 배치/날짜 기준으로 **train → val → test**를 유지하며, train 통계로만 변환한다.

4) **온·습도/메타 신호 사용**  
- temp/humid 보정은 보조 시계열이 제공된 경우에만 사용하고, train에서만 계수를 학습한다.  
- 보조 시계열이 없을 때 임의 추정 금지(`fit_only_when_available: true` 권장).

5) **RR0 베이스라인**  
- R/R0 baseline은 **세션 내부 early 구간**에서만 계산한다(`first_k`/`median_k`).  
- 전역 데이터 누출을 유발하는 전체 구간 기반 baseline 금지.
