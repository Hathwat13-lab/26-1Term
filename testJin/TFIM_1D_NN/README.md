# 1D NN TFIM FSS 신경망 실험

1차원 최근접 이웃 TFIM에 대해 finite-size scaling(FSS)을 신경망으로 학습하는 JAX 실험 폴더입니다.

## 기본 파이프라인

1. `L = 6, 8, 10, 12`는 학습용 finite-size exact label로 생성합니다.
2. `L = 14, 16`은 학습에 쓰지 않고 holdout 검증용으로 남깁니다.
3. 신경망은 `F(T, h, 1/L)`를 예측합니다.
4. JAX Hessian으로 `Cv = -T d2F/dT2`, `chi = -d2F/dh2`를 계산합니다.
5. `1/L -> 0` 외삽 결과를 열역학 극한 해석해와 비교합니다.

## 실행

프로젝트 루트의 `venv`를 사용합니다.

```powershell
..\..\venv\Scripts\python.exe src\1_generate_data.py
..\..\venv\Scripts\python.exe src\3_train.py
..\..\venv\Scripts\python.exe src\4_evaluate.py
```

출력은 아래 폴더에 저장됩니다.

- `data/`: 학습/검증/해석해 csv
- `models/`: 학습된 weight와 training log
- `figures/`: 학습 곡선, free energy/chi 비교 그림

## 문제 원인과 수정 분기 비교

상전이 학습 실패 원인과 troubleshooting 과정을 설명하기 위해 두 분기를 보존했습니다.

```powershell
..\..\venv\Scripts\python.exe src\variants\baseline_derivative_blind\3_train.py
..\..\venv\Scripts\python.exe src\variants\sobolev_critical\3_train.py
..\..\venv\Scripts\python.exe src\5_compare_variants.py
```

빠르게 현황을 보려면 아래 문서를 보면 됩니다.

- `TROUBLESHOOTING_STORY.md`
- `results_variant_comparison.md`
