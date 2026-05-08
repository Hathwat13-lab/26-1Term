import time
from pathlib import Path
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import matplotlib.pyplot as plt
from jax import random
from exact_solution import exact_free_energy

HERE = Path(__file__).parent   # 항상 testJin/ 기준으로 저장

# ==========================================
# 1. 물리량 계산 (Exact Labels)
# ==========================================
def _free_energy_raw(h, T, num_k=2000):
    beta = 1.0 / T
    k = jnp.linspace(0, jnp.pi, num_k)
    arg = 1.0 + h**2 - 2.0 * h * jnp.cos(k) + 1e-8
    eps_k = 2.0 * jnp.sqrt(arg)
    x = beta * eps_k / 2.0
    return -T * jnp.mean(jnp.logaddexp(x, -x))

_dF_dh_raw    = jax.grad(_free_energy_raw, argnums=0)
_d2F_dh2_raw  = jax.grad(_dF_dh_raw,      argnums=0)
_chi_raw      = lambda h, T: -_d2F_dh2_raw(h, T)

vmap_F   = jax.jit(jax.vmap(_free_energy_raw, in_axes=(0, 0)))
vmap_chi = jax.jit(jax.vmap(_chi_raw,         in_axes=(0, 0)))

# ==========================================
# 2. 모델 정의 (멀티태스크 Surrogate)
# ==========================================
T_SCALE = 8.0
H_SCALE = 2.5

class TFIMSurrogate(nn.Module):
    hidden_dims: int = 128

    @nn.compact
    def __call__(self, x):
        x = x / jnp.array([T_SCALE, H_SCALE])
        z = nn.Dense(self.hidden_dims)(x)
        z = nn.tanh(z)
        z = nn.Dense(self.hidden_dims)(z)
        z = nn.tanh(z)
        z = nn.Dense(self.hidden_dims)(z)
        z = nn.tanh(z)
        
        F_head   = nn.Dense(self.hidden_dims // 2)(z)
        F_head   = nn.tanh(F_head)
        F_pred   = nn.Dense(1)(F_head)

        chi_head = nn.Dense(self.hidden_dims // 2)(z)
        chi_head = nn.tanh(chi_head)
        chi_pred = nn.Dense(1)(chi_head)

        return jnp.concatenate([F_pred, chi_pred], axis=-1)

# ==========================================
# 3. 데이터셋 생성
# ==========================================
def generate_dataset(num_samples: int = 20000):
    print(f"\n--- 데이터 생성 ---")
    n_unif = int(num_samples * 0.6)
    n_crit = num_samples - n_unif
    
    key = random.PRNGKey(42)
    key_t1, key_h1, key_t2, key_h2, key_shuf = random.split(key, 5)

    T_unif = random.uniform(key_t1, (n_unif,), minval=0.05, maxval=8.0)
    h_unif = random.uniform(key_h1, (n_unif,), minval=0.05, maxval=2.5)

    T_crit = random.uniform(key_t2, (n_crit,), minval=0.05, maxval=8.0)
    h_crit = random.normal( key_h2, (n_crit,)) * 0.15 + 1.0
    h_crit = jnp.clip(h_crit, 0.05, 2.5)

    T_samp = jnp.concatenate([T_unif, T_crit])
    h_samp = jnp.concatenate([h_unif, h_crit])

    _ = vmap_F(  h_samp[:2], T_samp[:2])
    _ = vmap_chi(h_samp[:2], T_samp[:2])

    t0 = time.time()
    F_vals   = vmap_F(  h_samp, T_samp)
    chi_vals = vmap_chi(h_samp, T_samp)
    jax.block_until_ready(F_vals); jax.block_until_ready(chi_vals)
    print(f"데이터 생성 소요 시간: {time.time()-t0:.4f} 초")

    X = jnp.stack([T_samp, h_samp], axis=1)
    Y = jnp.stack([F_vals, chi_vals], axis=1)

    # PINN label at h=1.0 for the generated T_samp
    h_crit_array = jnp.full_like(T_samp, 1.0)
    chi_crit_vals = vmap_chi(h_crit_array, T_samp)

    idx = random.permutation(key_shuf, num_samples)
    X, Y, chi_crit_vals = X[idx], Y[idx], chi_crit_vals[idx]
    split = int(num_samples * 0.8)
    X_tr, Y_tr, Y_pinn_tr = X[:split],  Y[:split], chi_crit_vals[:split]
    X_te, Y_te, Y_pinn_te = X[split:],  Y[split:], chi_crit_vals[split:]

    print(f"Train: {X_tr.shape[0]}개 | Test: {X_te.shape[0]}개")
    return X_tr, Y_tr, Y_pinn_tr, X_te, Y_te, Y_pinn_te

# ==========================================
# 4. 학습 함수 (PINN 추가)
# ==========================================
def get_train_step_fn(lambda_pinn=1.0):
    def train_step(state, model, X, Y, Y_pinn):
        def loss_fn(params):
            # 1. Data Loss
            pred = model.apply(params, X)          # (batch, 2)
            data_loss = jnp.mean((pred - Y) ** 2)
            
            # 2. PINN Loss (Hessian constraint at h=1.0)
            T_samp = X[:, 0]
            h_crit = 1.0
            
            # F(h, T)를 예측하여 h에 대한 2차 미분 계산 (-chi)
            def F_only(h_val, T_val):
                inp = jnp.array([T_val, h_val])
                return model.apply(params, inp)[0]
                
            d2F_dh2_fn = jax.grad(jax.grad(F_only, argnums=0), argnums=0)
            vmap_d2F_dh2 = jax.vmap(d2F_dh2_fn, in_axes=(None, 0))
            
            nn_d2F = vmap_d2F_dh2(h_crit, T_samp)
            
            # PINN constraint: NN's -d2F/dh2 should match exact chi (Y_pinn)
            pinn_loss = jnp.mean((-nn_d2F - Y_pinn) ** 2)
            
            # Total Loss
            total_loss = data_loss + lambda_pinn * pinn_loss
            return total_loss, (data_loss, pinn_loss)
            
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, aux[0], aux[1]
    return train_step

def eval_loss_fn(state, model, X, Y):
    pred = model.apply(state.params, X)
    return jnp.mean((pred - Y) ** 2)

# ==========================================
# 5. 메인 실행부
# ==========================================
if __name__ == "__main__":
    from flax.training.train_state import TrainState

    X_tr, Y_tr, Y_pinn_tr, X_te, Y_te, Y_pinn_te = generate_dataset(num_samples=15000)

    model = TFIMSurrogate(hidden_dims=128)
    key   = random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, 2)))

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=5e-4)
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # PINN weight 설정
    LAMBDA_PINN = 1.0
    train_step_jit = jax.jit(get_train_step_fn(LAMBDA_PINN), static_argnums=(1,))
    eval_loss_jit = jax.jit(eval_loss_fn, static_argnums=(1,))

    num_epochs  = 8000
    log_every   = 50
    train_losses, test_losses, pinn_losses, log_epochs = [], [], [], []

    print(f"\n--- PINN 모델 학습 ({num_epochs} epochs, Full-batch) ---")
    t0 = time.time()

    for epoch in range(num_epochs):
        state, tr_loss, data_l, pinn_l = train_step_jit(state, model, X_tr, Y_tr, Y_pinn_tr)

        if (epoch + 1) % log_every == 0:
            te_loss = eval_loss_jit(state, model, X_te, Y_te)
            train_losses.append(float(tr_loss))
            test_losses.append(float(te_loss))
            pinn_losses.append(float(pinn_l))
            log_epochs.append(epoch + 1)
            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1:5d} | Total Loss: {tr_loss:.6f} | Data MSE: {data_l:.6f} | PINN Loss: {pinn_l:.6f} | Test MSE: {te_loss:.6f}")

    jax.block_until_ready(state.params)
    train_time = time.time() - t0
    print(f"학습 소요 시간: {train_time:.4f} 초")

    # ── 최종 예측 ─────────────────────────────────────────
    @jax.jit
    def predict(params, x):
        return model.apply(params, x)

    _ = predict(state.params, X_te[:1])

    t0 = time.time()
    Y_pred = predict(state.params, X_te)
    jax.block_until_ready(Y_pred)
    nn_time = time.time() - t0

    _ = vmap_F(X_te[:1, 1], X_te[:1, 0])
    t0 = time.time()
    F_exact   = vmap_F(  X_te[:, 1], X_te[:, 0])
    chi_exact = vmap_chi(X_te[:, 1], X_te[:, 0])
    jax.block_until_ready(F_exact); jax.block_until_ready(chi_exact)
    exact_time = time.time() - t0

    F_pred_te   = Y_pred[:, 0]
    chi_pred_te = Y_pred[:, 1]

    mse_F   = float(jnp.mean((F_pred_te   - Y_te[:, 0]) ** 2))
    mae_F   = float(jnp.mean(jnp.abs(F_pred_te   - Y_te[:, 0])))
    mse_chi = float(jnp.mean((chi_pred_te - Y_te[:, 1]) ** 2))
    mae_chi = float(jnp.mean(jnp.abs(chi_pred_te - Y_te[:, 1])))

    print("\n--- 평가 결과 (Test set) ---")
    print(f"  F   | MSE: {mse_F:.6f}  MAE: {mae_F:.6f}")
    print(f"  chi | MSE: {mse_chi:.6f}  MAE: {mae_chi:.6f}")
    print(f"Exact 연산 시간 : {exact_time:.4f} 초")
    print(f"NN 추론 시간    : {nn_time:.4f} 초")
    if nn_time > 0:
        print(f"→ NN이 Exact보다 {exact_time/nn_time:.1f}배 빠름")

    # ==========================================
    # 6. 시각화 — 3개 창 동시 출력
    # ==========================================
    BG    = '#0f0f1a'
    PANEL = '#13132a'
    WHITE = '#e8e8ff'
    GRID  = '#2a2a55'

    def dark_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=WHITE, fontsize=12, pad=9)
        ax.set_xlabel(xlabel, color=WHITE, fontsize=10)
        ax.set_ylabel(ylabel, color=WHITE, fontsize=10)
        ax.tick_params(colors='#9999cc', labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor('#333366')
        ax.grid(True, alpha=0.18, color=GRID, linestyle=':')
        ax.axvline(1.0, color='white', linestyle=':', alpha=0.25, lw=1.0)

    figA, axA = plt.subplots(figsize=(9, 5))
    figA.patch.set_facecolor(BG)
    figA.canvas.manager.set_window_title('Fig A — PINN Learning Curve')
    dark_ax(axA, 'Learning Curve (Total / Test / PINN Loss)', 'Epoch', 'Loss (log scale)')
    axA.semilogy(log_epochs, train_losses, color='#7788ff', lw=1.8, label='Total Train Loss')
    axA.semilogy(log_epochs, test_losses,  color='#ff8844', lw=1.8, linestyle='--', label='Test Data MSE')
    axA.semilogy(log_epochs, pinn_losses,  color='#88ff88', lw=1.8, linestyle=':', label='PINN Hessian Loss')
    axA.legend(framealpha=0.25, labelcolor=WHITE, facecolor='#1a1a3a', edgecolor='#444466', fontsize=9)
    figA.tight_layout()
    figA.savefig(HERE / 'figA_pinn_learning_curve.png', dpi=200, bbox_inches='tight', facecolor=figA.get_facecolor())

    T_plot_list = [0.1, 0.5, 1.0, 2.0, 4.0]
    cmap_plot   = plt.get_cmap('plasma', len(T_plot_list))
    plot_colors = [cmap_plot(i) for i in range(len(T_plot_list))]

    h_line = jnp.linspace(0.05, 2.5, 400)
    h_dots = jnp.linspace(0.05, 2.5, 55)

    figB, axB = plt.subplots(figsize=(10, 6))
    figB.patch.set_facecolor(BG)
    figB.canvas.manager.set_window_title('Fig B — Free Energy: Exact vs PINN')
    dark_ax(axB,
            f'PINN Helmholtz Free Energy\nMAE$_F$ = {mae_F:.4f}  |  MSE$_F$ = {mse_F:.6f}',
            'External Transverse Field  $h / J$',
            'Helmholtz Free Energy  $F$')

    figC, axC = plt.subplots(figsize=(10, 6))
    figC.patch.set_facecolor(BG)
    figC.canvas.manager.set_window_title('Fig C — Susceptibility: Exact vs PINN')
    dark_ax(axC,
            r'PINN Transverse Susceptibility $\chi = -\partial^2 F / \partial h^2$'
            f'\nMAE$_{{\\chi}}$ = {mae_chi:.4f}  |  MSE$_{{\\chi}}$ = {mse_chi:.6f}',
            'External Transverse Field  $h / J$',
            r'Susceptibility  $\chi$')

    for i, T_val in enumerate(T_plot_list):
        col   = plot_colors[i]
        label = f'T = {T_val}'

        T_vec_line  = jnp.full_like(h_line, T_val)
        F_line_ex   = vmap_F(  h_line, T_vec_line)
        chi_line_ex = vmap_chi(h_line, T_vec_line)
        axB.plot(h_line, F_line_ex,   color=col, lw=1.8, label=label)
        axC.plot(h_line, chi_line_ex, color=col, lw=1.8, label=label)

        T_vec_dots = jnp.full((len(h_dots),), T_val)
        X_dots     = jnp.stack([T_vec_dots, h_dots], axis=1)
        Y_dots_nn  = predict(state.params, X_dots)
        axB.scatter(h_dots, Y_dots_nn[:, 0], facecolors='none', edgecolors=col, s=30, lw=1.3, alpha=0.9, marker='o', zorder=3, label='_nolegend_')
        axC.scatter(h_dots, Y_dots_nn[:, 1], facecolors='none', edgecolors=col, s=30, lw=1.3, alpha=0.9, marker='o', zorder=3, label='_nolegend_')

    for ax in (axB, axC):
        ax.legend(framealpha=0.25, labelcolor=WHITE, facecolor='#1a1a3a', edgecolor='#444466', fontsize=9, loc='best')

    figB.tight_layout()
    figB.savefig(HERE / 'figB_pinn_free_energy_sweep.png', dpi=200, bbox_inches='tight', facecolor=figB.get_facecolor())
    figC.tight_layout()
    figC.savefig(HERE / 'figC_pinn_susceptibility_sweep.png', dpi=200, bbox_inches='tight', facecolor=figC.get_facecolor())

    print("\nSaved: figA_pinn_learning_curve.png, figB_pinn_free_energy_sweep.png, figC_pinn_susceptibility_sweep.png")
    # plt.show()
