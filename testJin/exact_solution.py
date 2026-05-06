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
    # 3. 온도별 4가지 물리량 — 4개 창 동시 출력
    # ==========================================
    import matplotlib.cm as cm
    import numpy as np
    from pathlib import Path
    HERE = Path(__file__).parent   # 항상 testJin/ 기준으로 저장

    # ── JAX 자동 미분으로 자화율(2차 미분) 정의 ──────────────────────
    # d²F/dh² : vmap은 scalar→scalar 함수에만 적용 가능하므로 grad를 먼저 구성
    dF_dh   = jax.grad(exact_free_energy, argnums=0)          # ∂F/∂h
    d2F_dh2 = jax.grad(dF_dh, argnums=0)                      # ∂²F/∂h²
    d2F_dh2_jit = jax.jit(d2F_dh2)

    vmap_F   = jax.vmap(exact_free_energy, in_axes=(0, None))
    vmap_chi = jax.vmap(d2F_dh2_jit,       in_axes=(0, None)) # χ = -∂²F/∂h²

    h_sweep = jnp.linspace(0.05, 2.5, 500)

    # 비교할 온도 목록 (극저온 ~ 고온)
    T_list = [0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]

    cmap   = plt.get_cmap('plasma', len(T_list))
    colors = [cmap(i) for i in range(len(T_list))]

    BG    = '#0f0f1a'
    PANEL = '#13132a'
    WHITE = '#e8e8ff'
    GRID  = '#2a2a55'

    def style_ax(ax, title, xlabel, ylabel):
        """공통 다크 테마 스타일 적용"""
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=WHITE, fontsize=13, pad=10)
        ax.set_xlabel(xlabel, color=WHITE, fontsize=11)
        ax.set_ylabel(ylabel, color=WHITE, fontsize=11)
        ax.tick_params(colors='#9999cc', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333366')
        ax.grid(True, alpha=0.18, color=GRID, linestyle=':')
        ax.axvline(1.0, color='white', linestyle='--', alpha=0.35,
                   lw=1.2, label='$h_c=1$')

    def add_legend(ax):
        ax.legend(loc='best', framealpha=0.25, labelcolor=WHITE,
                  facecolor='#1a1a3a', edgecolor='#444466', fontsize=9)

    # ── Figure 1: 헬름홀츠 자유에너지 ───────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 5.5))
    fig1.patch.set_facecolor(BG)
    fig1.canvas.manager.set_window_title('Fig 1 — Free Energy')
    style_ax(ax1,
             '1D TFIM — Helmholtz Free Energy vs Field',
             'External Transverse Field  $h / J$',
             'Free Energy  $F$')
    for i, T in enumerate(T_list):
        F_vals = vmap_F(h_sweep, float(T))
        ax1.plot(h_sweep, F_vals, color=colors[i], lw=1.9, label=f'T = {T}')
    add_legend(ax1)
    fig1.tight_layout()
    fig1.savefig(HERE / 'fig1_free_energy.png', dpi=200, bbox_inches='tight',
                 facecolor=fig1.get_facecolor())

    # ── Figure 2: 자화율 (−∂²F/∂h²) ────────────────────────────────
    print("Computing susceptibility via JAX autograd (this may take a moment)...")
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    fig2.patch.set_facecolor(BG)
    fig2.canvas.manager.set_window_title('Fig 2 — Transverse Susceptibility')
    style_ax(ax2,
             '1D TFIM — Transverse Susceptibility vs Field\n'
             r'$\chi = -\partial^2 F / \partial h^2$',
             'External Transverse Field  $h / J$',
             r'Susceptibility  $\chi$')
    for i, T in enumerate(T_list):
        chi_vals = -vmap_chi(h_sweep, float(T))          # χ = -d²F/dh²
        ax2.plot(h_sweep, chi_vals, color=colors[i], lw=1.9, label=f'T = {T}')
    ax2.set_ylim(bottom=0)
    add_legend(ax2)
    fig2.tight_layout()
    fig2.savefig(HERE / 'fig2_susceptibility.png', dpi=200, bbox_inches='tight',
                 facecolor=fig2.get_facecolor())

    # ── Figure 3: 에너지 갭 ─────────────────────────────────────────
    # 에너지 갭은 T=0 해석적 해 (열 요동 보정 없음)
    vmap_gap = jax.vmap(energy_gap)
    gap_vals = vmap_gap(h_sweep)

    fig3, ax3 = plt.subplots(figsize=(9, 5.5))
    fig3.patch.set_facecolor(BG)
    fig3.canvas.manager.set_window_title('Fig 3 — Energy Gap')
    style_ax(ax3,
             '1D TFIM — Energy Gap  (T=0 Analytic)',
             'External Transverse Field  $h / J$',
             r'Energy Gap  $\Delta$')
    ax3.plot(h_sweep, gap_vals, color='#44ffaa', lw=2.2, label=r'$\Delta(h)$ — T=0')
    ax3.annotate('Gap closes\nat QPT', xy=(1.0, 0.0), xytext=(1.3, 0.6),
                 color='#44ffaa', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='#44ffaa', lw=1.2))
    add_legend(ax3)
    fig3.tight_layout()
    fig3.savefig(HERE / 'fig3_energy_gap.png', dpi=200, bbox_inches='tight',
                 facecolor=fig3.get_facecolor())

    # ── Figure 4: 질서 매개변수 ─────────────────────────────────────
    # Pfeuty (1970) T=0 해석적 해
    vmap_mz = jax.vmap(order_parameter)
    mz_vals = vmap_mz(h_sweep)

    fig4, ax4 = plt.subplots(figsize=(9, 5.5))
    fig4.patch.set_facecolor(BG)
    fig4.canvas.manager.set_window_title('Fig 4 — Order Parameter')
    style_ax(ax4,
             r'1D TFIM — Order Parameter $\langle\sigma^z\rangle$  (T=0 Analytic)',
             'External Transverse Field  $h / J$',
             r'Magnetization  $m_z = \langle\sigma^z\rangle$')
    ax4.plot(h_sweep, mz_vals, color='#ff6688', lw=2.2,
             label=r'$(1-h^2)^{1/8}$ — Pfeuty (1970)')
    ax4.fill_between(h_sweep, mz_vals, alpha=0.12, color='#ff6688')
    ax4.annotate('Ordered\nphase', xy=(0.5, float(order_parameter(jnp.array(0.5)))),
                 xytext=(0.15, 0.55), color='#ff9aaa', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='#ff9aaa', lw=1.2))
    ax4.annotate('Disordered\nphase', xy=(1.5, 0.0), xytext=(1.55, 0.2),
                 color='#aaaaff', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='#aaaaff', lw=1.2))
    add_legend(ax4)
    fig4.tight_layout()
    fig4.savefig(HERE / 'fig4_order_parameter.png', dpi=200, bbox_inches='tight',
                 facecolor=fig4.get_facecolor())

    print("Saved: fig1_free_energy.png, fig2_susceptibility.png, "
          "fig3_energy_gap.png, fig4_order_parameter.png")
    plt.show()   # 4개 창 동시 표시
