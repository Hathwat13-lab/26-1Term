# 문제 해결 기록: 왜 MSE만으로는 부족했나

## 1. 처음 관찰한 문제

첫 번째 신경망은 자유에너지 `F`의 MSE만 보면 꽤 좋아 보였습니다. 하지만 `chi = -d2F/dh2` 그래프를 보면 상전이 근처 구조를 충분히 잘 잡지 못했습니다.

이건 `FSS_idea_memo.md`에서 예상했던 실패 지점과 맞습니다.

- `F(T, h, 1/L)` 자체는 비교적 매끄럽고 맞추기 쉽습니다.
- 상전이 신호는 `F`의 높이가 아니라 곡률, 즉 `chi = -d2F/dh2`에 들어 있습니다.
- 따라서 신경망은 `F` MSE를 낮게 만들면서도, 실제로는 `chi` 피크를 흐리게 만들거나 위치를 살짝 틀릴 수 있습니다.

## 2. 기준선 분기

파일:

- `src/variants/baseline_derivative_blind/2_jax_model.py`
- `src/variants/baseline_derivative_blind/3_train.py`

특징:

- ReLU는 쓰지 않고 GELU MLP를 사용했습니다. 그래서 Hessian 자체는 계산 가능합니다.
- 하지만 학습은 여전히 자유에너지 `F` 오차에 거의 지배됩니다.
- `chi` 패널티는 매우 약합니다. `chi_weight = 2e-3`
- 임계 영역을 따로 더 많이 샘플링하지 않습니다.

이 분기는 “`F`는 잘 맞는데 곡률은 충분히 학습되지 않는” 원래 실패 모드를 보여주기 위한 기준선입니다.

## 3. 문제 해결 분기

파일:

- `src/variants/sobolev_critical/2_jax_model.py`
- `src/variants/sobolev_critical/3_train.py`
- `src/variants/sobolev_critical/4_evaluate.py`

수정 내용:

- `h = 1` 근처를 잘 표현하도록 smooth critical feature를 추가했습니다.
- activation은 미분 가능한 Swish 계열을 사용했습니다.
- `chi` Sobolev loss를 `2e-3`에서 `8e-2`로 키웠습니다.
- 낮은 온도, `h ~= 1` 근처 샘플을 derivative batch에 더 많이 넣었습니다.
- 약한 `Cv` 곡률 loss도 추가했습니다.

이 분기는 “상전이 fan 근처의 곡률을 직접 학습시키면 `chi` 구조가 나아지는가?”를 테스트합니다.

## 4. 비교 결과가 말해준 것

문제 해결 분기는 일부 상전이 특징을 개선했습니다. 특히 가장 낮은 온도에서 `chi` 피크 위치가 `h = 1` 근처로 더 잘 붙었습니다.

하지만 새 문제가 생겼습니다. 낮은 온도 열역학 극한에서 `chi` 피크 높이가 과하게 커지는 overshoot가 나타났고, 전체 thermodynamic-limit `chi` 지표는 오히려 나빠질 수 있었습니다.

이건 실험 실패라기보다 다음 방향을 알려주는 신호입니다. 다음 모델은 세 가지를 동시에 맞춰야 합니다.

- `F` 자체의 정확도 유지
- `chi` 피크 위치를 올바른 `h`에 배치
- 피크를 과하게 세우지 않고 높이까지 맞추기

## 5. MSE 외에 필요한 평가 기준

비교 스크립트는 이제 아래 기준을 함께 봅니다.

- `MAE`: MSE보다 직관적으로 읽기 쉬운 평균 절대오차
- `NRMSE`: target 변동폭으로 정규화한 RMSE
- `corr`: 예측 곡선의 모양이 맞는지 보는 상관계수
- `R2`: target 분산을 얼마나 설명하는지
- `SMAPE`: scale-aware 상대오차
- critical-region metric: `T <= 0.35`, `|h - 1| <= 0.25` 영역만 따로 평가
- `h_peak_abs_error`: 상전이 피크 위치 오차
- `chi_peak_rel_error`: 피크 높이 상대오차
- low-temperature peak metric: `T <= 0.2`의 피크만 따로 평가

실행:

```powershell
..\..\venv\Scripts\python.exe src\5_compare_variants.py
```

주요 출력:

- `results_variant_comparison.md`
- `figures/variant_comparison/fig_variant_chi_curves.png`
