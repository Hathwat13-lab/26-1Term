from functools import partial

import jax
import jax.numpy as jnp
# 반드시 다른 JAX 연산이 시작되기 전에 선언해야 합니다.
jax.config.update("jax_enable_x64", True)

# EPS: 에너지 분산 관계(Dispersion)에서 루트 안이 완벽한 0이 되어 
# 그래디언트(NaN)가 폭발하는 것을 막는 미세한 값
EPS = 1e-9

# EPS_LOG: T가 0으로 가거나 Zero mode에서 sinh 항의 log1p 내부가 
# -inf로 발산하는 것을 막아주는 극소값 (수치적 한계 방어)
EPS_LOG = 1e-30


def dispersion(k, h):
    """
    준입자(Quasiparticle)의 에너지 분산 관계 (Energy Dispersion).
    epsilon_k = 2 * sqrt(J^2 + h^2 - 2Jh*cos(k)) 
    (여기서는 J=1로 둔 상태입니다.)
    """
    arg = 1.0 + h * h - 2.0 * h * jnp.cos(k) + EPS
    return 2.0 * jnp.sqrt(arg)


@partial(jax.jit, static_argnames=("num_k",))
def thermodynamic_free_energy(T, h, num_k=4096):
    """
    [무한 사슬 / 열역학적 극한 (L -> inf)]
    파수 k가 연속적으로 변한다고 가정하고 적분(0~pi)을 통해 자유 에너지를 구합니다.
    여기서는 num_k 크기의 리만 합(Riemann sum)으로 적분을 정밀하게 근사합니다.
    """
    k = jnp.linspace(0.0, jnp.pi, num_k)
    eps_k = dispersion(k, h)
    x = eps_k / (2.0 * T)
    
    # ln(2*cosh(x)) = logaddexp(x, -x)
    # 자유 에너지 밀도 F = -k_B * T * (1/pi) \int \ln(2\cosh(\epsilon_k / 2T)) dk
    return -T * jnp.mean(jnp.logaddexp(x, -x))


@partial(jax.jit, static_argnames=("L",))
def finite_free_energy(T, h, L):
    """
    [유한 크기 (Finite-L) 헬름홀츠 자유 에너지 밀도]
    Jordan-Wigner 변환에 따른 패리티(Parity) 경계 조건 분리 및 투영 적용.
    수치적 오버플로우와 언더플로우를 방지하기 위해 LogSumExp 트릭을 사용합니다.
    """
    n = jnp.arange(L)
    
    # 1. 파수(Momentum) 양자화
    # APBC (Even Parity, Neveu-Schwarz): 반주기적 경계 조건
    k_A = (2.0 * n + 1.0) * jnp.pi / L  
    # PBC (Odd Parity, Ramond): 주기적 경계 조건
    k_P = 2.0 * n * jnp.pi / L          

    # 2. 각 섹터별 분산 관계
    eps_A = dispersion(k_A, h)
    eps_P = dispersion(k_P, h)

    x_A = eps_A / (2.0 * T)
    x_P = eps_P / (2.0 * T)

    # 3. 4개의 하위 파티션(Z1, Z2, Z3, Z4) 분배 함수 (Log 스케일)
    # Z1, Z3는 cosh 항의 곱 -> sum(ln(2cosh(x))) -> logaddexp 이용
    ln_Z1 = jnp.sum(jnp.logaddexp(x_A, -x_A))
    ln_Z3 = jnp.sum(jnp.logaddexp(x_P, -x_P))
    
    # Z2, Z4는 sinh 항의 곱 -> sum(ln(2sinh(x))) -> sum(x + ln(1 - e^{-2x}))
    # T->0, x->0 일 때 발생하는 NaN을 막기 위해 EPS_LOG 추가
    ln_Z2 = jnp.sum(x_A + jnp.log1p(-jnp.exp(-2.0 * x_A) + EPS_LOG))
    ln_Z4 = jnp.sum(x_P + jnp.log1p(-jnp.exp(-2.0 * x_P) + EPS_LOG))

    # 4. 패리티 투영: Z = 0.5 * (Z1 + Z2 + Z3 - Z4)
    # 가장 큰 값인 ln_Z1을 기준으로 묶어내어 지수 폭발(Overflow) 방지
    max_ln_Z = ln_Z1
    
    # h가 1보다 크면 Z4의 부호를 뒤집어주는 sign(1 - h) 곱하기 추가
    sum_exp = (
        jnp.exp(ln_Z1 - max_ln_Z) +
        jnp.exp(ln_Z2 - max_ln_Z) +
        jnp.exp(ln_Z3 - max_ln_Z) -
        jnp.sign(1.0 - h) * jnp.exp(ln_Z4 - max_ln_Z)
    )
    
    # 전체 분배 함수의 로그: ln(Z)
    ln_Z = max_ln_Z + jnp.log(sum_exp) - jnp.log(2.0)

    # 자유 에너지 '밀도(density)' 이므로 크기 L로 나누어 줌
    return -T * ln_Z / L


def _finite_scalar(T, h, L):
    """make_observable_fns로 넘기기 위한 래퍼(Wrapper) 함수"""
    return finite_free_energy(T, h, L)


def _thermo_scalar(T, h):
    """make_observable_fns로 넘기기 위한 래퍼(Wrapper) 함수"""
    return thermodynamic_free_energy(T, h)


def make_observable_fns(f_scalar, static_L=None):
    """
    [물리량 파생 함수 (자동 미분 활용)]
    JAX의 자동 미분(jax.grad)을 사용하여 자유 에너지로부터 다른 열역학적 물리량을 유도합니다.
    """
    # L이 주어지는 유한 크기와 무한 크기를 유연하게 처리
    if static_L is None:
        scalar = lambda T, h: f_scalar(T, h)
    else:
        scalar = lambda T, h: f_scalar(T, h, static_L)

    # 1차 미분
    df_dT = jax.grad(scalar, argnums=0)
    df_dh = jax.grad(scalar, argnums=1)
    
    # 2차 미분
    d2f_dT2 = jax.grad(df_dT, argnums=0)
    d2f_dh2 = jax.grad(df_dh, argnums=1)

    # 엔트로피 S = - (∂F / ∂T)
    def entropy(T, h):
        return -df_dT(T, h)

    # 자화 M = - (∂F / ∂h)
    def magnetization(T, h):
        return -df_dh(T, h)

    # 비열 Cv = -T * (∂^2 F / ∂T^2)
    def cv(T, h):
        return -T * d2f_dT2(T, h)

    # 자화율(감수율) chi = - (∂^2 F / ∂h^2)
    def chi(T, h):
        return -d2f_dh2(T, h)

    # 계산 효율을 위해 모든 함수를 jit 컴파일하여 반환
    return jax.jit(scalar), jax.jit(magnetization), jax.jit(cv), jax.jit(entropy), jax.jit(chi)


def vectorized_finite_observables(L):
    """
    [유한 크기(Finite-L) 배치 처리 함수]
    L이 고정된 상태에서 다수의 (T, h) 쌍을 한 번에 계산할 수 있도록 vmap 적용.
    생성기(1_generate_data.py)에서 데이터를 대량으로 뽑아낼 때 호출됩니다.
    """
    f, m, cv, s, chi = make_observable_fns(_finite_scalar, static_L=L)
    return (
        jax.jit(jax.vmap(f, in_axes=(0, 0))),
        jax.jit(jax.vmap(m, in_axes=(0, 0))),
        jax.jit(jax.vmap(cv, in_axes=(0, 0))),
        jax.jit(jax.vmap(s, in_axes=(0, 0))),
        jax.jit(jax.vmap(chi, in_axes=(0, 0))),
    )

# [열역학적 극한(Thermodynamic limit) 배치 처리 함수]
# L이 무한대인 상태에 대해 다수의 (T, h) 쌍을 한 번에 계산
thermo_f, thermo_m, thermo_cv, thermo_s, thermo_chi = make_observable_fns(_thermo_scalar)

v_thermo_f = jax.jit(jax.vmap(thermo_f, in_axes=(0, 0)))
v_thermo_m = jax.jit(jax.vmap(thermo_m, in_axes=(0, 0)))
v_thermo_cv = jax.jit(jax.vmap(thermo_cv, in_axes=(0, 0)))
v_thermo_s = jax.jit(jax.vmap(thermo_s, in_axes=(0, 0)))
v_thermo_chi = jax.jit(jax.vmap(thermo_chi, in_axes=(0, 0)))