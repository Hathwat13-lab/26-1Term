# 1D TFIM Surrogate Project: 소스 코드 상세 설명 및 기술 심층 분석 (src)

이 문서는 `src` 폴더 내의 각 Python 스크립트가 어떤 역할을 하며, 어떤 논리로 작동하는지 상세히 설명합니다. 물리적 배경, 신경망 아키텍처, 그리고 핵심 코드 리뷰를 포함하여 입문자부터 전문가까지 이해할 수 있도록 구성되었습니다.

---

## 1. 개요: 데이터의 흐름 (Data Flow)

모든 코드의 흐름은 아래와 같은 물리적 인과 관계를 따릅니다.

$$ (T, h, 1/L) \xrightarrow{MLP} F \xrightarrow{\text{JAX Diff}} (C_v, \chi) $$

*   **입력 (Inputs):** 온도($T$), 횡자기장($h$), 시스템 크기의 역수($1/L$).
*   **핵심 출력 (Main Output):** 헬름홀츠 자유 에너지 밀도 ($F$).
*   **유도된 출력 (Derived Observables):** $F$를 미분하여 얻는 비열($C_v$)과 자기감수율($\chi$).

---

## 2. 엔진과 데이터 생성 (`tfim_exact.py` & `1_generate_data.py`)

*   **`tfim_exact.py` (물리 엔진):** 1D TFIM의 해석해를 계산하는 수학 함수들의 집합입니다. 어떤 입력이든 물리 법칙에 따라 즉시 정답을 내놓는 **'계산기'**입니다.
*   **`1_generate_data.py` (데이터 샘플러):** 물리 엔진을 반복 호출하여 훈련에 사용할 CSV 데이터셋을 만듭니다. 매번 무거운 계산을 하지 않기 위해 **'문제집'**을 미리 만들어두는 과정입니다.

### 핵심 코드: $k$-그리드 생성
```python
# 유한 시스템: L개의 이산적 파동 (Anti-Periodic Boundary Condition)
k_finite = (jnp.arange(L) + 0.5) * jnp.pi / L

# 무한 시스템 (열역학적 극한): 4096개의 촘촘한 샘플링을 통한 적분 근사
k_infinite = jnp.linspace(0.0, jnp.pi, 4096)
```

---

## 3. 신경망 아키텍처 (`2_jax_model.py`)

이 모델은 **다층 퍼셉트론(MLP)** 구조의 심층 신경망입니다.

*   **레이어 구조:** 총 **5개의 레이어** (입력층 1, 은닉층 3, 출력층 1)
    *   `widths = (9, 80, 80, 80, 1)`
    *   입력(9개: $T, h, 1/L$ + 6개 물리 특징) -> 은닉층(80개 x 3) -> 출력(1개: $F$)
*   **활성화 함수:** **Swish** ($x \cdot \text{sigmoid}(x)$)
    *   **왜 Swish인가?** ReLU와 달리 전 구간 매끄러운 미분이 가능하여, $F$를 두 번 미분해 $C_v, \chi$를 얻을 때 수치적으로 안정적입니다.
*   **물리 기반 특징 (Physics-informed features):**
```python
def critical_features(x):
    # 상전이(h=1) 근처의 발산 거동을 돕기 위해 dh, log(r2) 등을 입력에 추가
    thermal_rounding = 0.85 * T + 1.35 * inv_L + 0.06
    r2 = dh * dh + thermal_rounding * thermal_rounding
    return jnp.stack([dh, dh * dh, thermal_rounding, 0.45 * jnp.log(r2), ...], axis=-1)
```

---

## 4. 학습 전략 및 손실 함수 (`3_train.py`)

신경망은 아래 정의된 **전체 손실 함수(Total Loss)**를 최소화하도록 학습됩니다.

$$L_{total} = L_F + \lambda_{\chi} L_{\chi} + \lambda_{C_v} L_{C_v} + \lambda_{guard} L_{guard}$$

*   **소볼레프 학습 (Sobolev Training):** 값($F$)뿐만 아니라 미분값($\chi, C_v$)까지 정답에 맞추도록 강제합니다.
```python
def _loss(params, xb, yb, xd, yd, ...):
    f_pred = batched_free_energy(params, xb)
    f_loss = jnp.mean((f_pred - yb[:, 0]) ** 2) # 자유 에너지 값 MSE

    obs_pred = batched_observables(params, xd)
    chi_loss = jnp.mean(w * ((obs_pred[:, 2] - yd[:, 2]) / chi_scale) ** 2) # 자기감수율 미분 MSE
```
*   **일반화의 원리:** 모델은 $1/L$ 변수를 통해 **시스템 크기에 따른 물리량의 변화 추세**를 배웁니다. 따라서 학습하지 않은 $L=16$이나 $L \to \infty$ 영역에서도 정확한 예측이 가능해집니다.

---

## 5. 외삽 및 성능 평가 (`4_evaluate.py`)

여기서 **외삽(Extrapolation)**이라는 결정적인 단계가 수행됩니다.

*   **외삽이란?** 학습된 가중치를 고정하고, 입력값에 $1/L = 0$을 넣어 **무한한 시스템($L = \infty$)의 거동을 예측**하는 것입니다.
*   **신경망 업데이트 여부:** **아니오.** 외삽은 훈련 과정이 아니라, 신경망이 스케일링 추세를 제대로 이해했는지 확인하는 **'최종 시험'**입니다.
*   **시각화 결과 상세:**
    *   **학습 곡선 (`figA_learning_curve.png`):** 에포크(Epoch)가 지남에 따라 오차가 줄어드는 과정을 보여줍니다. 학습 데이터(L=6~12)뿐만 아니라 **검증 데이터(L=14, 16)**의 오차가 함께 낮아지는 것을 통해 모델이 물리적 추세를 성공적으로 학습했음을 보여주는 가장 중요한 그래프입니다.
    *   **자유 에너지 외삽 (`figB_free_energy_extrapolation.png`):** $1/L=0$을 대입하여 예측한 무한 시스템의 $F$ 곡선이 실제 해석해와 얼마나 일치하는지 보여줍니다.
    *   **자기감수율 피크 (`figC_susceptibility_derivative_check.png`):** 상전이 지점($h=1$)의 뾰족한 피크를 신경망이 얼마나 정확하게 포착했는지 확인합니다. (값뿐만 아니라 미분값까지 학습한 Sobolev 학습의 성과입니다.)
    *   **오차 분포 히스토그램 (`figD_holdout_error_hist.png`):** 학습에 쓰이지 않은 $L=14, 16$ 데이터에 대해 예측 오차가 0을 중심으로 얼마나 조밀하게 분포하는지 보여줍니다.
    *   **크리티컬 팬 히트맵 (`figE_chi_critical_fan_heatmap.png`):** $T$와 $h$ 평면에서 자기감수율($\chi$)의 세기를 히트맵으로 그려 상전이 영역의 V자 형태(Critical Fan)가 해석해와 시각적으로 일치함을 증명합니다.

---

### 요약: 이 코드는 어떻게 작동하는가?
1.  `tfim_exact.py` (엔진)가 정답을 계산하는 법을 정의하면,
2.  `1_generate_data.py` (샘플러)가 학습용 문제지(CSV)를 만들고,
3.  `2_jax_model.py` (MLP)가 Swish 활성화 함수와 물리적 힌트를 가진 두뇌를 설계한 뒤,
4.  `3_train.py` (훈련)가 **미분값까지 포함된 복합 손실 함수**로 빡세게 훈련시켜서,
5.  `4_evaluate.py` (외삽)가 **학습하지 않은 무한한 세상을 예측**하며 실력을 검증합니다.
