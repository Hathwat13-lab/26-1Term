# 1차원 횡자기장 이징 모델의 정확한 열역학적 해법

> **모델:** 1D Transverse-Field Ising Model (TFIM)  
> **방법:** Jordan–Wigner 변환 → 자유 페르미온 대각화 → 자동 미분 기반 열역학량 도출

---

## 1. 모델 해밀토니안

1차원 횡자기장 이징 모델의 해밀토니안은 다음과 같다:

$$
\hat{H} = -J \sum_{i=1}^{L} \sigma_i^z \sigma_{i+1}^z - h \sum_{i=1}^{L} \sigma_i^x
$$

여기서 $J$는 스핀 간 교환 상호작용의 세기, $h$는 $x$-방향 횡자기장의 세기, $\sigma_i^{x,z}$는 파울리 행렬이다. 이 모델은 $h/J = 1$에서 양자 위상 전이(quantum phase transition)를 보인다. **본 구현에서는 $J = 1$로 고정**한다.

---

## 2. Jordan–Wigner 변환과 에너지 분산 관계

### 2.1 변환 개요

Jordan–Wigner 변환을 통해 스핀 연산자를 스핀리스 페르미온 생성·소멸 연산자 $c_i^\dagger, c_i$로 사상(mapping)한다:

$$
\sigma_i^+ = \left(\prod_{j<i} (1 - 2c_j^\dagger c_j)\right) c_i^\dagger, \qquad
\sigma_i^z = 1 - 2c_i^\dagger c_i
$$

이 변환 후 해밀토니안은 자유 페르미온(free fermion) 형태로 쓰이며, 푸리에 변환을 통해 파수(momentum) $k$ 공간에서 완전히 대각화된다.

### 2.2 에너지 분산 관계

$$
\boxed{\varepsilon_k = 2\sqrt{J^2 + h^2 - 2Jh\cos k} = 2\sqrt{1 + h^2 - 2h\cos k}}
$$

이것이 준입자(quasiparticle)의 에너지 분산 관계다.

- $h = J = 1$ (임계점)일 때: $\varepsilon_k = 2\sqrt{2}\,|\sin(k/2)|$ → $k=0$에서 에너지 갭(gap)이 닫힘  
- $h \ll 1$ (강결합 한계): $\varepsilon_k \approx 2J$ (평탄한 분산)  
- $h \gg 1$ (약결합 한계): $\varepsilon_k \approx 2h$ (자기장 지배)

**코드 구현 (`dispersion`):**

```python
def dispersion(k, h):
    arg = 1.0 + h*h - 2.0*h*jnp.cos(k) + EPS   # EPS: 임계점 k=0에서 √0 의 기울기(NaN) 방지
    return 2.0 * jnp.sqrt(arg)
```

> **수치 안정성:** 임계점 $h=1$, $k=0$ 근방에서 $\text{arg} \to 0^+$이 되어 $\nabla\sqrt{\text{arg}}$이 발산한다. `EPS = 1e-9`를 더해 자동 미분 시 NaN 폭발을 방지한다.

---

## 3. 열역학적 극한에서의 자유 에너지

### 3.1 이론적 유도

$L \to \infty$ 극한에서 파수 $k$는 연속적으로 변하며, 분배 함수는 다음과 같이 인수분해된다:

$$
\ln Z = \sum_k \ln\left(2\cosh\frac{\varepsilon_k}{2T}\right)
$$

자유 에너지 밀도(단위 스핀당)는 적분으로 표현된다:

$$
\boxed{f_\infty(T, h) = \frac{F}{L} = -\frac{T}{\pi} \int_0^{\pi} \ln\!\left(2\cosh\frac{\varepsilon_k}{2T}\right) dk}
$$

### 3.2 수치 근사

위 적분을 $[0, \pi]$ 구간의 리만 합(Riemann sum)으로 근사한다 (`num_k = 4096` 격자점):

$$
f_\infty \approx -T \cdot \frac{1}{N_k} \sum_{n=0}^{N_k - 1} \ln\!\left(2\cosh\frac{\varepsilon_{k_n}}{2T}\right), \quad k_n = \frac{n\pi}{N_k - 1}
$$

**코드 구현 (`thermodynamic_free_energy`):**

```python
k = jnp.linspace(0.0, jnp.pi, num_k)
eps_k = dispersion(k, h)
x = eps_k / (2.0 * T)

# ln(2·cosh(x)) = logaddexp(x, -x)  ← 오버플로우 없이 정확하게 계산
return -T * jnp.mean(jnp.logaddexp(x, -x))
```

> **LogSumExp 트릭:** $\ln(2\cosh x) = \ln(e^x + e^{-x}) = \texttt{logaddexp}(x, -x)$  
> 이 항등식을 이용하면 큰 $x$ (저온 한계)에서도 지수 오버플로우 없이 안정적으로 계산된다.

---

## 4. 유한 크기 계의 자유 에너지

### 4.1 Jordan–Wigner 패리티 투영

유한 크기 $L$ 계에서 Jordan–Wigner 변환 후 올바른 스핀 힐베르트 공간을 복원하려면 **페르미온 패리티(parity)** 를 투영해야 한다. 이는 경계 조건(BC)을 짝수/홀수 패리티 섹터로 분리하는 것에 해당한다.

| 섹터 | 경계 조건 | 파수 양자화 | 물리 의미 |
|:---:|:---:|:---:|:---:|
| APBC | 반주기적 (Neveu–Schwarz) | $k_n = \dfrac{(2n+1)\pi}{L}$ | 짝수 패리티 |
| PBC  | 주기적 (Ramond) | $k_n = \dfrac{2n\pi}{L}$ | 홀수 패리티 |

### 4.2 4개 부분배함수 분해

각 경계 조건 섹터에서 분배 함수는 모든 $k$ 모드의 인수분해로 쓰인다:

$$
Z_{\text{APBC}}^{+} = \prod_{k \in \text{APBC}} 2\cosh\frac{\varepsilon_k^A}{2T}, \quad
Z_{\text{APBC}}^{-} = \prod_{k \in \text{APBC}} 2\sinh\frac{\varepsilon_k^A}{2T}
$$

$$
Z_{\text{PBC}}^{+} = \prod_{k \in \text{PBC}} 2\cosh\frac{\varepsilon_k^P}{2T}, \quad
Z_{\text{PBC}}^{-} = \prod_{k \in \text{PBC}} 2\sinh\frac{\varepsilon_k^P}{2T}
$$

패리티 투영 후 물리적인 분배 함수는 위상에 따라 달라진다:

$$
\boxed{Z = \frac{1}{2}\left(Z_1 + Z_2 + Z_3 - \operatorname{sgn}(1-h)\cdot Z_4\right)}
$$

단, $Z_1 \equiv Z_{\text{APBC}}^+$, $Z_2 \equiv Z_{\text{APBC}}^-$, $Z_3 \equiv Z_{\text{PBC}}^+$, $Z_4 \equiv Z_{\text{PBC}}^-$이다.

#### 왜 부호가 위상에 따라 달라지는가?

PBC 섹터에는 $k=0$ 영모드(zero mode)가 존재하며, 그 에너지는:

$$
\varepsilon_{k=0} = 2|1 - h|
$$

이 모드의 기저 상태 점유 여부가 양자 위상 전이를 경계로 반전된다:

| 위상 | 조건 | $\operatorname{sgn}(1-h)$ | $k=0$ 모드 | $Z_4$ 기여 |
|:---:|:---:|:---:|:---:|:---:|
| 강결합 (ordered) | $h < 1$ | $+1$ | 기저 상태에서 **비점유** | $-Z_4$ |
| 약결합 (disordered) | $h > 1$ | $-1$ | 기저 상태에서 **점유** | $+Z_4$ |

$h < 1$일 때 PBC 영모드가 비어 있으면 기저 상태는 짝수 패리티 섹터(APBC)에 머물고, $Z_4$의 기여는 음수여야 올바른 힐베르트 공간 제약이 구현된다. $h > 1$을 넘으면 영모드가 채워지면서 기저 상태의 패리티가 홀수로 전환되고, $Z_4$의 부호가 뒤집혀야 분배 함수가 물리적으로 올바른 상태를 반영한다. 이 부호 교대는 TFIM의 **위상학적 구조(topological structure)** 를 직접 인코딩한다.

### 4.3 로그 스케일에서의 계산

$Z_i$는 $L$에 지수적으로 커지므로, **로그 분배 함수** $\ln Z_i$를 직접 계산한다:

$$
\ln Z_1 = \sum_{k \in A} \ln\!\left(2\cosh\frac{\varepsilon_k^A}{2T}\right) = \sum_{k \in A} \texttt{logaddexp}(x_k^A, -x_k^A)
$$

$$
\ln Z_2 = \sum_{k \in A} \ln\!\left(2\sinh\frac{\varepsilon_k^A}{2T}\right) = \sum_{k \in A} \left[x_k^A + \ln\!\left(1 - e^{-2x_k^A}\right)\right]
$$

(단, $x_k \equiv \varepsilon_k / 2T$이다. $Z_3, Z_4$는 PBC 파수로 동일하게 계산.)

전체 로그 분배 함수는 최대값 $\ln Z_1$을 기준으로 뽑아내어(LogSumExp 트릭) 지수 오버플로우를 방지한다:

$$
\ln Z = \ln Z_1 + \ln\!\left[\,1 + e^{\ln Z_2 - \ln Z_1} + e^{\ln Z_3 - \ln Z_1} - \operatorname{sgn}(1-h)\cdot e^{\ln Z_4 - \ln Z_1}\,\right] - \ln 2
$$

유한 크기 자유 에너지 밀도:

$$
\boxed{f_L(T, h) = -\frac{T}{L}\ln Z}
$$

**코드 구현 (`finite_free_energy`):**

```python
ln_Z1 = jnp.sum(jnp.logaddexp(x_A, -x_A))              # cosh, APBC
ln_Z3 = jnp.sum(jnp.logaddexp(x_P, -x_P))              # cosh, PBC
ln_Z2 = jnp.sum(x_A + jnp.log1p(-jnp.exp(-2.0*x_A) + EPS_LOG))  # sinh, APBC
ln_Z4 = jnp.sum(x_P + jnp.log1p(-jnp.exp(-2.0*x_P) + EPS_LOG))  # sinh, PBC

# LogSumExp 트릭 (기준: ln_Z1)
# h > 1 (비정렬 위상)에서 sign(1-h) = -1 → Z4의 부호가 뒤집혀 +Z4
sum_exp = (
    jnp.exp(ln_Z1 - max_ln_Z) +
    jnp.exp(ln_Z2 - max_ln_Z) +
    jnp.exp(ln_Z3 - max_ln_Z) -
    jnp.sign(1.0 - h) * jnp.exp(ln_Z4 - max_ln_Z)
)
ln_Z = max_ln_Z + jnp.log(sum_exp) - jnp.log(2.0)

return -T * ln_Z / L
```

> **수치 안정성:** `EPS_LOG = 1e-30`은 $T \to 0$ 극한에서 `log1p(-exp(-2x))`의 인수가 0으로 수렴할 때 $\ln(0) = -\infty$ 발산을 방지한다.  
> **$h = 1$ 임계점:** $\operatorname{sgn}(0) = 0$이 되어 $Z_4$ 항이 통째로 소거된다. 임계점에서 PBC 영모드 기여가 사라지는 것은 물리적으로도 타당하다 — 정확히 임계점에서 두 위상의 기여가 상쇄되기 때문이다.

---

## 5. 열역학적 물리량의 자동 미분 도출

### 5.1 열역학 항등식

자유 에너지 $f(T, h)$로부터 다른 열역학 물리량들은 편미분으로 정의된다:

| 물리량 | 정의 | 의미 |
|:---:|:---:|:---:|
| 엔트로피 | $S = -\left(\dfrac{\partial f}{\partial T}\right)_h$ | 계의 무질서도 |
| 자화 | $M = -\left(\dfrac{\partial f}{\partial h}\right)_T$ | 스핀의 평균 $x$-방향 정렬 |
| 비열 | $C_v = -T\left(\dfrac{\partial^2 f}{\partial T^2}\right)_h$ | 온도 변화에 따른 에너지 흡수 |
| 자화율 | $\chi = -\left(\dfrac{\partial^2 f}{\partial h^2}\right)_T$ | 자기장 변화에 대한 자화 반응 |

이 관계들은 열역학적으로 **정확하며**, 근사 없이 닫힌 형태(closed-form)로 자유 에너지로부터 유도된다.

### 5.2 JAX 자동 미분 구현

JAX의 `jax.grad`를 사용하면 위 편미분들을 수치 차분(finite difference) 없이 **해석적 정확도**로 계산한다:

```python
df_dT  = jax.grad(scalar, argnums=0)          # ∂f/∂T
df_dh  = jax.grad(scalar, argnums=1)          # ∂f/∂h
d2f_dT2 = jax.grad(df_dT, argnums=0)          # ∂²f/∂T²  (grad of grad)
d2f_dh2 = jax.grad(df_dh, argnums=1)          # ∂²f/∂h²

entropy       = lambda T, h: -df_dT(T, h)
magnetization = lambda T, h: -df_dh(T, h)
cv            = lambda T, h: -T * d2f_dT2(T, h)
chi           = lambda T, h: -d2f_dh2(T, h)
```

> **자동 미분의 장점:** 수치 차분법 $(\partial f/\partial T \approx \Delta f / \Delta T)$은 스텝 크기에 따른 절단 오차(truncation error)와 소거 오차(cancellation error)를 피할 수 없다. JAX의 역전파(reverse-mode AD)는 연산 그래프를 통해 **기계 정밀도(machine precision)** 수준의 1, 2차 미분을 보장한다.

---

## 6. 대량 계산을 위한 벡터화

### 6.1 `vmap`을 이용한 배치 처리

훈련 데이터 생성 시 수천~수만 개의 $(T, h)$ 쌍에 대해 물리량을 계산해야 한다. JAX의 `vmap`(벡터화된 map)을 적용하면 Python 루프 없이 하드웨어 수준에서 병렬 처리된다:

$$
(T_1, h_1), (T_2, h_2), \ldots, (T_N, h_N) \;\xrightarrow{\texttt{vmap}}\; [f_1, f_2, \ldots, f_N]
$$

```python
# 유한 크기 계: L 고정 후 (T, h) 배열을 한번에 처리
v_f   = jax.jit(jax.vmap(f,   in_axes=(0, 0)))
v_m   = jax.jit(jax.vmap(m,   in_axes=(0, 0)))
v_cv  = jax.jit(jax.vmap(cv,  in_axes=(0, 0)))
v_s   = jax.jit(jax.vmap(s,   in_axes=(0, 0)))
v_chi = jax.jit(jax.vmap(chi, in_axes=(0, 0)))

# 열역학적 극한: L → ∞ 동일 구조
v_thermo_f   = jax.jit(jax.vmap(thermo_f,   in_axes=(0, 0)))
# ...
```

### 6.2 컴파일 전략

| 최적화 기법 | 역할 |
|:---:|:---:|
| `@jax.jit` | XLA 컴파일러로 첫 호출 시 연산 그래프를 최적화된 기계어로 변환 |
| `static_argnames=("L",)` | $L$을 정적 인수(상수)로 처리하여 $L$이 달라질 때만 재컴파일 |
| `vmap` | 루프를 내재화하여 GPU/TPU의 SIMD 병렬성 최대 활용 |

---

## 7. 수치 안정성 요약

| 극한 상황 | 발생 문제 | 방어 전략 |
|:---:|:---:|:---:|
| $h=1,\; k=0$ (임계점) | $\sqrt{\text{arg}} \to 0$, 기울기 NaN | `EPS = 1e-9` 추가 |
| $T \to 0$ (sinh 계산) | $\ln(1 - e^{-2x}) \to \ln 0 = -\infty$ | `EPS_LOG = 1e-30` 추가 |
| 큰 $L$ (분배 함수 계산) | $Z_i \sim e^{L}$, 지수 오버플로우 | LogSumExp 트릭 ($Z_1$ 기준) |
| 큰 $x$ (자유 에너지 계산) | $e^x$ 오버플로우 | `logaddexp(x, -x)` 사용 |

---

## 부록: 기호 정리

| 기호 | 의미 |
|:---:|:---:|
| $J$ | 스핀 교환 상호작용 (본 구현에서 $J=1$) |
| $h$ | 횡자기장 세기 |
| $T$ | 온도 ($k_B = 1$ 단위계) |
| $L$ | 스핀 체인의 길이 |
| $\varepsilon_k$ | 파수 $k$에서의 준입자 에너지 |
| $f$ | 자유 에너지 밀도 ($F/L$) |
| $S, M, C_v, \chi$ | 엔트로피, 자화, 비열, 자화율 |
| APBC / PBC | 반주기적 / 주기적 경계 조건 |
