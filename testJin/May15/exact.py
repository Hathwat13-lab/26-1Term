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
    [완벽 수정본] 패리티 진공 교차(Parity crossing) 자동 처리 및 
    극저온/고온 수치적 아티팩트 완벽 제거 버전.
    """
    n = jnp.arange(L)
    
    # 1. 파수(Momentum) 양자화
    k_A = (2.0 * n + 1.0) * jnp.pi / L  # APBC
    k_P = 2.0 * n * jnp.pi / L          # PBC

    # 2. 각 섹터별 분산 관계 (k=0 모드를 제외하고는 무조건 양수)
    eps_A = dispersion(k_A, h)
    
    # PBC에서 문제의 k=0 모드(n=0) 제외
    eps_P_rest = dispersion(k_P[1:], h)

    x_A = eps_A / (2.0 * T)
    x_P_rest = eps_P_rest / (2.0 * T)
    
    # [핵심 1] Zero mode (k=0)의 에너지를 절댓값 없이 부호 그대로 추적!
    # h > 1 일 때 음수가 되면서 패리티 반전이 JAX 미분망 안에서 부드럽게 일어납니다.
    x_0 = (1.0 - h) / T 

    # 3. 분배 함수 파티션 계산 (log1p 기반, EPS_LOG 삭제)
    # x가 무조건 양수이므로 log1p 내부가 음수가 될 일이 없어 EPS_LOG가 불필요합니다.
    ln_Z1 = jnp.sum(x_A + jnp.log1p(jnp.exp(-2.0 * x_A)))
    ln_Z2 = jnp.sum(x_A + jnp.log1p(-jnp.exp(-2.0 * x_A)))

    ln_Z3_rest = jnp.sum(x_P_rest + jnp.log1p(jnp.exp(-2.0 * x_P_rest)))
    ln_Z4_rest = jnp.sum(x_P_rest + jnp.log1p(-jnp.exp(-2.0 * x_P_rest)))

    # 4. 패리티 투영 결합
    # k=0 모드의 기여분인 2*cosh(x_0)와 2*sinh(x_0)를 밖에서 명시적으로 곱해줍니다.
    # 이렇게 하면 log(sinh)의 발산(Pole)과 sign() 함수의 미분 불가 문제를 완벽히 우회합니다.
    max_ln_Z = jnp.maximum(ln_Z1, ln_Z3_rest + jnp.abs(x_0))
    
    term1 = jnp.exp(ln_Z1 - max_ln_Z)
    term2 = jnp.exp(ln_Z2 - max_ln_Z)
    
    # 2*cosh(x_0) = exp(x_0) + exp(-x_0)
    term3 = (jnp.exp(x_0) + jnp.exp(-x_0)) * jnp.exp(ln_Z3_rest - max_ln_Z)
    
    # 2*sinh(x_0) = exp(x_0) - exp(-x_0)
    term4 = (jnp.exp(x_0) - jnp.exp(-x_0)) * jnp.exp(ln_Z4_rest - max_ln_Z)

    # 최종 결합
    sum_exp = term1 + term2 + term3 - term4

    ln_Z = max_ln_Z + jnp.log(sum_exp) - jnp.log(2.0)

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