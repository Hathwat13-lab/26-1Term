# 1D TFIM Surrogate Project: 소스 코드 상세 설명 및 기술 심층 분석 (src)

이 문서는 `src` 폴더 내의 각 Python 스크립트가 어떤 역할을 하며, 어떤 논리로 작동하는지 상세히 설명합니다. 단순한 기능 요약을 넘어, 사용된 알고리즘의 핵심 코드 리뷰와 물리적 배경을 깊이 있게 다룹니다.

---

## 1. `tfim_exact.py` (물리 엔진 및 해석해)

이 파일은 프로젝트의 "정답지" 역할을 하며, 1D TFIM의 정확한 에너지 값을 계산합니다.

### 핵심 개념: 시스템 크기 $L$과 $k$-그리드
*   **$L$ (System Size):** 스핀의 개수입니다. $L$이 작으면 유한 시스템, $L \to \infty$이면 열역학적 극한입니다.
*   **그리드 생성 로직:**
    ```python
    # 유한 시스템: Anti-Periodic Boundary Condition을 고려한 L개의 이산적 파동
    k_finite = (jnp.arange(L) + 0.5) * jnp.pi / L
    
    # 무한 시스템: 촘촘한(4096개) 샘플링을 통한 적분 근사
    k_infinite = jnp.linspace(0.0, jnp.pi, 4096)
    ```

---

## 2. `2_jax_model.py` (심층 신경망 아키텍처)

이 모델은 **다층 퍼셉트론(MLP)** 구조의 **심층 신경망(FNN)**입니다.

*   **왜 FNN인가?** 입력 $(T, h, 1/L)$ 사이의 복잡한 비선형 함수 관계를 근사하는 데 가장 유연하기 때문입니다.
*   **핵심 코드: 물리 기반 특징 (Physics-informed features)**
    ```python
    def critical_features(x):
        # 상전이(h=1) 근처의 발산 거동을 돕기 위해 미리 계산된 특징들을 입력에 추가
        thermal_rounding = 0.85 * T + 1.35 * inv_L + 0.06
        r2 = dh * dh + thermal_rounding * thermal_rounding
        return jnp.stack([dh, dh * dh, thermal_rounding, 0.45 * jnp.log(r2), ...], axis=-1)
    ```
    이러한 특징 공학을 통해 신경망이 상전이의 뾰족한 피크를 훨씬 더 잘 학습할 수 있습니다.

---

## 3. `3_train.py` (소볼레프 학습 및 추세 학습)

신경망이 정답을 맞히도록 훈련시키는 핵심 스크립트입니다.

*   **소볼레프 학습 (Sobolev Training):** 값($F$)뿐만 아니라 미분값($C_v, \chi$)까지 정답에 맞추도록 강제합니다.
    ```python
    def _loss(params, xb, yb, xd, yd, ...):
        f_pred = batched_free_energy(params, xb)
        f_loss = jnp.mean((f_pred - yb[:, 0]) ** 2) # 값 MSE

        obs_pred = batched_observables(params, xd)
        chi_loss = jnp.mean(w * ((obs_pred[:, 2] - yd[:, 2]) / chi_scale) ** 2) # 미분 MSE
        return f_loss + chi_weight * chi_loss + ...
    ```
*   **왜 학습하지 않은(Holdout) 데이터의 에러가 줄어드는가?** 
    모델은 단순히 값을 암기하는 것이 아니라, $1/L$ 입력 변수를 통해 **시스템 크기에 따른 물리량의 변화 추세**를 배웁니다. $L=12$까지의 추세를 정확히 파악하면, 자연스럽게 $L=16$이나 $L \to \infty$ 영역으로의 물리적 일반화(Generalization)가 가능해집니다.

---

## 4. `4_evaluate.py` (외삽 및 결과 분석)

학습이 끝난 모델이 무한한 시스템($L \to \infty$)을 얼마나 잘 맞추는지 시각화합니다.

*   **자유 에너지 외삽 (`figB`):** $1/L=0$을 대입하여 무한 시스템의 $F$ 곡선을 재현합니다.
*   **자기감수율 체크 (`figC`):** 상전이 지점($h=1$)에서 솟구치는 피크를 포착합니다. 이는 미분 학습의 결과입니다.
*   **크리티컬 팬 히트맵 (`figE`):** $T-h$ 평면에서 상전이 영역이 퍼져나가는 형태가 해석해와 일치함을 보여줍니다.

---

## 5. `1_generate_data.py` (데이터 구성)

*   **입력:** $T, h, 1/L$
*   **라벨:** $F$ (자유 에너지), $C_v$ (비열), $\chi$ (자기감수율)
*   **전략:** 상전이 근처와 저온 영역을 더 촘촘하게 샘플링하여 학습 난이도가 높은 구간을 집중 공략합니다.

---

### 요약: 이 코드는 어떻게 작동하는가?
1.  `tfim_exact.py`가 물리 법칙으로 **정답**을 알려주면,
2.  `1_generate_data.py`가 학습용 **문제지**를 만들고,
3.  `2_jax_model.py`가 물리적 힌트를 가진 **두뇌**를 설계한 뒤,
4.  `3_train.py`가 **미분값까지 빡세게 훈련**시켜서,
5.  `4_evaluate.py`가 **무한한 세상을 예측**하며 성적을 확인합니다.
