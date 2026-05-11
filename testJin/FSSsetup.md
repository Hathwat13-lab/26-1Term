# 🌌 1D TFIM Finite-Size Scaling (FSS) 설계도

이 설계도는 총 4개의 파이썬 파일(스크립트)이 순차적으로 실행되어, **양자 이징 모델(TFIM)의 유한 크기 효과를 학습하고 무한 시스템으로 외삽(Extrapolation)**하는 파이프라인을 구축합니다.

---

### 📂 프로젝트 구조 (Directory Tree)

```text
1d_tfim_project/
│
├── data/                             # [데이터 저장소] 정답지 파일 저장
│   ├── ed_train_data.csv             # L=4,6,8,10의 작은 격자 데이터 (학습용)
│   └── analytic_truth_data.csv       # L=∞의 해석적 정답지 (검증용)
│
├── models/                           # [모델 저장소]
│   └── saved_weights.eqx             # 학습 완료된 JAX/Equinox 가중치
│
└── src/                              # [코드 저장소]
    ├── 1_generate_data.py            # 데이터 생성 (ED & Analytical)
    ├── 2_jax_model.py                # 모델 아키텍처 및 AD(자동 미분) 정의
    ├── 3_train.py                    # 학습 루프
    └── 4_evaluate.py                 # 무한대 외삽 및 결과 시각화
```

---

### 📝 파이프라인 단계별 핵심 역할

#### 1️⃣ `src/1_generate_data.py` - 데이터 생성 단계
대리 모델이 학습할 '과거 기출문제(작은 L)'와 검증을 위한 '수능 정답지(무한대 L)'를 생성합니다.

*   **Grid 정의**: 
    *   **자기장 ($h$)**: 0.1 ~ 2.0 (임계점 $h=1.0$ 주변 집중)
    *   **온도 ($T$)**: 0.05 ~ 1.0 (저온 한계 고려)
    *   **시스템 크기 ($L$)**: 4, 6, 8, 10
*   **ED(Exact Diagonalization)**: 각 $(h, T, L)$ 조합에 대해 해밀토니안 행렬($2^L \times 2^L$)을 대각화하여 자유에너지($F$)를 계산.
*   **해석해(Analytical Solution)**: 요르단-위그너 변환을 통한 적분 공식으로 $L=\infty$인 경우의 $F$를 계산.

#### 2️⃣ `src/2_jax_model.py` - 물리 엔진 설계
물리 법칙을 미분으로 추출할 수 있는 신경망 뼈대를 정의합니다.

*   **입출력**: `[T, h, 1/L]` $\to$ `[F_pred]`
*   **아키텍처**: Deep Neural Network (MLP)
    *   **Activation**: `Softplus` 또는 `GELU` (고차 미분 가능성 확보)
*   **AD(Automatic Differentiation)**:
    *   `get_Cv`: $\frac{\partial^2 F}{\partial T^2}$ (비열)
    *   `get_chi`: $\frac{\partial^2 F}{\partial h^2}$ (자기화율)

#### 3️⃣ `src/3_train.py` - 모델 학습
작은 시스템($L$)의 데이터를 통해 물리적 상관관계를 학습합니다.

*   `ed_train_data.csv`를 기반으로 MSE Loss 최적화.
*   학습된 가중치를 `models/saved_weights.eqx`에 저장.

#### 4️⃣ `src/4_evaluate.py` - 시험 및 시각화
학습된 모델을 이용해 **열역학적 극한($L \to \infty$)**을 예측합니다.

*   **무한대 외삽**: 모델의 입력값 중 `1/L` 자리에 **`0`**을 대입하여 $L=\infty$ 상태의 $F$를 예측.
*   **물리량 추출**: 예측된 $F$로부터 AD를 통해 $C_V$와 $\chi$ 도출.
*   **시각화 및 검증**:
    *   $h$-$T$ 평면의 히트맵 (상전이 V-shape 확인).
    *   해석해와 외삽 결과의 비교 그래프를 통한 정확도 증명.