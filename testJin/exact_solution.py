import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# ==========================================
# 1. JAX 기반 엄밀한 헬름홀츠 자유에너지 계산
# ==========================================

@jax.jit
def exact_free_energy(h, T, num_k=2000):
    beta = 1.0 / T
    k = jnp.linspace(0, jnp.pi, num_k)
    eps_k = 2.0 * jnp.sqrt(1.0 + h**2 - 2.0 * h * jnp.cos(k))
    
    x = beta * eps_k / 2.0
    integrand = jnp.logaddexp(x, -x)
    
    return -T * jnp.mean(integrand)

# ==========================================
# 2. 임계점 근처의 다른 물리량 해석적 해(Exact Solution)
# ==========================================

def energy_gap(h):
    """ 
    에너지 갭 (Energy Gap)
    바닥상태와 첫 들뜬상태 간의 에너지 차이.
    가장 작은 에너지 간격은 분산관계식 eps_k 에서 k=0 일 때 발생합니다.
    """
    return 2.0 * jnp.abs(1.0 - h)

def order_parameter(h):
    """ 
    Z축 자화 (Longitudinal Magnetization, Order Parameter)
    Pfeuty (1970)에 의해 유도된 해석적 해:
    h < 1 에서는 (1 - h^2)^(1/8), h >= 1 에서는 0
    """
    return jnp.where(h < 1.0, jnp.power(1.0 - h**2, 1.0/8.0), 0.0)

if __name__ == "__main__":
    # ==========================================
    # 3. 플롯팅 (Plotting)
    # ==========================================
    d2F_dh2 = jax.jit(jax.grad(jax.grad(exact_free_energy, argnums=0), argnums=0))

    vmap_F = jax.vmap(exact_free_energy, in_axes=(0, None))
    vmap_chi = jax.vmap(d2F_dh2, in_axes=(0, None))
    vmap_gap = jax.vmap(energy_gap)
    vmap_mz = jax.vmap(order_parameter)

    T_fixed = 0.01  # 극저온
    h_sweep = jnp.linspace(0.01, 2.0, 500)
    
    # 4가지 물리량 계산
    F_sweep = vmap_F(h_sweep, T_fixed)
    chi_sweep = -vmap_chi(h_sweep, T_fixed)
    gap_sweep = vmap_gap(h_sweep)
    mz_sweep = vmap_mz(h_sweep)

    # 2x2 그리드로 플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Quantum Phase Transition in 1D TFIM', fontsize=16)
    
    # 1. 자유에너지 (Top-Left)
    axes[0, 0].plot(h_sweep, F_sweep, color='indigo', lw=2)
    axes[0, 0].axvline(1.0, color='red', linestyle='--', alpha=0.6)
    axes[0, 0].set_title(f'1. Free Energy $F$ (T={T_fixed})')
    axes[0, 0].set_ylabel('$F$')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 자화율 (Top-Right)
    axes[0, 1].plot(h_sweep, chi_sweep, color='darkorange', lw=2)
    axes[0, 1].axvline(1.0, color='red', linestyle='--', alpha=0.6)
    axes[0, 1].set_title(f'2. Transverse Susceptibility ($-\\partial^2 F / \\partial h^2$)')
    axes[0, 1].set_ylabel('Susceptibility')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 에너지 갭 (Bottom-Left)
    axes[1, 0].plot(h_sweep, gap_sweep, color='forestgreen', lw=2)
    axes[1, 0].axvline(1.0, color='red', linestyle='--', alpha=0.6)
    axes[1, 0].set_title('3. Energy Gap ($\\Delta$)')
    axes[1, 0].set_xlabel('Transverse Field Strength ($h$)')
    axes[1, 0].set_ylabel('$\\Delta$')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 질서 매개변수 (Bottom-Right)
    axes[1, 1].plot(h_sweep, mz_sweep, color='crimson', lw=2)
    axes[1, 1].axvline(1.0, color='red', linestyle='--', alpha=0.6)
    axes[1, 1].set_title('4. Order Parameter ($\\langle \\sigma^z \\rangle$)')
    axes[1, 1].set_xlabel('Transverse Field Strength ($h$)')
    axes[1, 1].set_ylabel('Magnetization $m_z$')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 제목을 위한 공간 확보
    plt.show()
