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
    """JIT 없는 raw 버전 — grad/vmap 조합 시 사용
    NOTE: eps_k의 sqrt 안에 1e-12를 더해 h≈1(임계점)에서 gradient NaN 방지
    """
    beta = 1.0 / T
    k = jnp.linspace(0, jnp.pi, num_k)
    arg = 1.0 + h**2 - 2.0 * h * jnp.cos(k) + 1e-12  # ← numerical stability
    eps_k = 2.0 * jnp.sqrt(arg)
    x = beta * eps_k / 2.0
    return -T * jnp.mean(jnp.logaddexp(x, -x))

# M_z = -∂F/∂h  (열역학 관계식) — raw 위에 grad 걸어야 NaN 없음
_dF_dh_raw    = jax.grad(_free_energy_raw, argnums=0)
_mz_raw       = lambda h, T: -_dF_dh_raw(h, T)

# vmap 먼저, jit은 마지막에
vmap_F  = jax.jit(jax.vmap(_free_energy_raw, in_axes=(0, 0)))
vmap_Mz = jax.jit(jax.vmap(_mz_raw,         in_axes=(0, 0)))

# ==========================================
# 2. 모델 정의 (멀티태스크 Surrogate)
#    입력: (T, h)  →  출력: (F, M_z)
# ==========================================
# 입력 정규화 상수 (학습 안정성)
T_SCALE = 8.0    # T  ∈ [0.05, 8.0]  → [~0, 1]
H_SCALE = 2.5    # h  ∈ [0.05, 2.5]  → [~0, 1]

class TFIMSurrogate(nn.Module):
    hidden_dims: int = 128

    @nn.compact
    def __call__(self, x):
        # 입력 정규화: [T, h] → [T/T_SCALE, h/H_SCALE]
        x = x / jnp.array([T_SCALE, H_SCALE])
        # 공유 backbone
        z = nn.Dense(self.hidden_dims)(x)
        z = nn.softplus(z)
        z = nn.Dense(self.hidden_dims)(z)
        z = nn.softplus(z)
        z = nn.Dense(self.hidden_dims)(z)
        z = nn.softplus(z)
        # 각 물리량별 head
        F_head  = nn.Dense(self.hidden_dims // 2)(z)
        F_head  = nn.softplus(F_head)
        F_pred  = nn.Dense(1)(F_head)                    # 자유에너지

        Mz_head = nn.Dense(self.hidden_dims // 2)(z)
        Mz_head = nn.softplus(Mz_head)
        Mz_pred = nn.Dense(1)(Mz_head)                   # 자화

        return jnp.concatenate([F_pred, Mz_pred], axis=-1)  # (batch, 2)

# ==========================================
# 3. 데이터셋 생성
#    X: (N, 2)  = [T, h]
#    Y: (N, 2)  = [F, M_z]
# ==========================================
def generate_dataset(num_samples: int = 15000):
    print(f"\n--- 데이터 생성 ---")
    print(f"총 {num_samples}개의 레이블 데이터 (T, h) → (F, Mz) 생성 중...")

    key = random.PRNGKey(42)
    key_t, key_h, key_shuf = random.split(key, 3)

    # 물리적으로 의미 있는 범위
    #   T: 0.05 ~ 8.0  (극저온 ~ 고온)
    #   h: 0.05 ~ 2.5  (임계점 h=1 충분히 포함)
    T_samp = random.uniform(key_t, (num_samples,), minval=0.05, maxval=8.0)
    h_samp = random.uniform(key_h, (num_samples,), minval=0.05, maxval=2.5)

    # JIT 워밍업
    _ = vmap_F( h_samp[:2], T_samp[:2])
    _ = vmap_Mz(h_samp[:2], T_samp[:2])

    t0 = time.time()
    F_vals  = vmap_F( h_samp, T_samp)
    Mz_vals = vmap_Mz(h_samp, T_samp)
    jax.block_until_ready(F_vals); jax.block_until_ready(Mz_vals)
    print(f"데이터 생성 소요 시간: {time.time()-t0:.4f} 초")

    X = jnp.stack([T_samp, h_samp], axis=1)              # (N, 2)
    Y = jnp.stack([F_vals, Mz_vals], axis=1)             # (N, 2)

    # 셔플 후 8:2 분할
    idx = random.permutation(key_shuf, num_samples)
    X, Y = X[idx], Y[idx]
    split = int(num_samples * 0.8)
    X_tr, Y_tr = X[:split],  Y[:split]
    X_te, Y_te = X[split:],  Y[split:]

    print(f"Train: {X_tr.shape[0]}개 | Test: {X_te.shape[0]}개")
    return X_tr, Y_tr, X_te, Y_te

# ==========================================
# 4. 학습 함수
# ==========================================
def train_step(state, model, X, Y):
    def loss_fn(params):
        pred = model.apply(params, X)          # (batch, 2)
        loss = jnp.mean((pred - Y) ** 2)
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

train_step_jit = jax.jit(train_step, static_argnums=(1,))

def eval_loss_jit(state, model, X, Y):
    pred = model.apply(state.params, X)
    return jnp.mean((pred - Y) ** 2)

eval_loss_jit = jax.jit(eval_loss_jit, static_argnums=(1,))

# ==========================================
# 5. 메인 실행부
# ==========================================
if __name__ == "__main__":
    from flax.training.train_state import TrainState

    # 데이터 준비
    X_tr, Y_tr, X_te, Y_te = generate_dataset(num_samples=15000)

    # 모델 초기화
    model = TFIMSurrogate(hidden_dims=128)
    key   = random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, 2)))

    # 안정적인 고정 LR + gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=1e-3)
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # ── 학습 루프 ─────────────────────────────────────────
    num_epochs  = 5000
    log_every   = 50
    train_losses, test_losses, log_epochs = [], [], []

    print(f"\n--- 모델 학습 ({num_epochs} epochs, Full-batch) ---")
    t0 = time.time()

    for epoch in range(num_epochs):
        state, tr_loss = train_step_jit(state, model, X_tr, Y_tr)

        if (epoch + 1) % log_every == 0:
            te_loss = eval_loss_jit(state, model, X_te, Y_te)
            train_losses.append(float(tr_loss))
            test_losses.append(float(te_loss))
            log_epochs.append(epoch + 1)
            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1:5d} | Train MSE: {tr_loss:.6f} | Test MSE: {te_loss:.6f}")

    jax.block_until_ready(state.params)
    train_time = time.time() - t0
    print(f"학습 소요 시간: {train_time:.4f} 초")

    # ── 최종 예측 ─────────────────────────────────────────
    @jax.jit
    def predict(params, x):
        return model.apply(params, x)

    # JIT 워밍업
    _ = predict(state.params, X_te[:1])

    t0 = time.time()
    Y_pred = predict(state.params, X_te)
    jax.block_until_ready(Y_pred)
    nn_time = time.time() - t0

    # Exact 재계산 시간
    _ = vmap_F(X_te[:1, 1], X_te[:1, 0])           # 워밍업
    t0 = time.time()
    F_exact  = vmap_F( X_te[:, 1], X_te[:, 0])
    Mz_exact = vmap_Mz(X_te[:, 1], X_te[:, 0])
    jax.block_until_ready(F_exact); jax.block_until_ready(Mz_exact)
    exact_time = time.time() - t0

    F_pred_te  = Y_pred[:, 0]
    Mz_pred_te = Y_pred[:, 1]

    mse_F  = float(jnp.mean((F_pred_te  - Y_te[:, 0]) ** 2))
    mae_F  = float(jnp.mean(jnp.abs(F_pred_te  - Y_te[:, 0])))
    mse_Mz = float(jnp.mean((Mz_pred_te - Y_te[:, 1]) ** 2))
    mae_Mz = float(jnp.mean(jnp.abs(Mz_pred_te - Y_te[:, 1])))

    print("\n--- 평가 결과 (Test set) ---")
    print(f"  F  | MSE: {mse_F:.6f}  MAE: {mae_F:.6f}")
    print(f"  Mz | MSE: {mse_Mz:.6f}  MAE: {mae_Mz:.6f}")
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

    # ── Fig A: 학습 곡선 ──────────────────────────────────
    figA, axA = plt.subplots(figsize=(9, 5))
    figA.patch.set_facecolor(BG)
    figA.canvas.manager.set_window_title('Fig A — Learning Curve')
    dark_ax(axA, 'Learning Curve  (Train / Test MSE)',
            'Epoch', 'MSE Loss (log scale)')
    axA.semilogy(log_epochs, train_losses, color='#7788ff', lw=1.8, label='Train MSE')
    axA.semilogy(log_epochs, test_losses,  color='#ff8844', lw=1.8,
                 linestyle='--', label='Test MSE')
    axA.legend(framealpha=0.25, labelcolor=WHITE, facecolor='#1a1a3a',
               edgecolor='#444466', fontsize=9)
    figA.tight_layout()
    figA.savefig(HERE / 'figA_learning_curve.png', dpi=200,
                 bbox_inches='tight', facecolor=figA.get_facecolor())

    # ── Fig B & C: 자기장 스윕  Exact(선) vs NN(점) per T ──
    # 비교할 온도 목록
    T_plot_list = [0.1, 0.5, 1.0, 2.0, 4.0]
    cmap_plot   = plt.get_cmap('plasma', len(T_plot_list))
    plot_colors = [cmap_plot(i) for i in range(len(T_plot_list))]

    # 촘촘한 h 배열 (Exact 선용)
    h_line = jnp.linspace(0.05, 2.5, 400)
    # 성긴 h 배열 (NN 점용 — 지나치게 촘촘하면 선처럼 보임)
    h_dots = jnp.linspace(0.05, 2.5, 55)

    @jax.jit
    def predict(params, x):
        return model.apply(params, x)

    figB, axB = plt.subplots(figsize=(10, 6))
    figB.patch.set_facecolor(BG)
    figB.canvas.manager.set_window_title('Fig B — Free Energy: Exact vs NN')
    dark_ax(axB,
            f'Helmholtz Free Energy — Exact (line) vs NN (dots)\n'
            f'MAE$_F$ = {mae_F:.4f}  |  MSE$_F$ = {mse_F:.6f}',
            'External Transverse Field  $h / J$',
            'Helmholtz Free Energy  $F$')

    figC, axC = plt.subplots(figsize=(10, 6))
    figC.patch.set_facecolor(BG)
    figC.canvas.manager.set_window_title('Fig C — Magnetization: Exact vs NN')
    dark_ax(axC,
            r'Magnetization $M_z = -\partial F / \partial h$ — Exact (line) vs NN (dots)'
            f'\nMAE$_{{Mz}}$ = {mae_Mz:.4f}  |  MSE$_{{Mz}}$ = {mse_Mz:.6f}',
            'External Transverse Field  $h / J$',
            r'Magnetization  $M_z$')

    for i, T_val in enumerate(T_plot_list):
        col   = plot_colors[i]
        label = f'T = {T_val}'

        # ── Exact 선 ────────────────────────────────────────
        T_vec_line = jnp.full_like(h_line, T_val)
        F_line_ex  = vmap_F( h_line, T_vec_line)
        Mz_line_ex = vmap_Mz(h_line, T_vec_line)
        axB.plot(h_line, F_line_ex,  color=col, lw=1.8, label=label)
        axC.plot(h_line, Mz_line_ex, color=col, lw=1.8, label=label)

        # ── NN 예측 점 ───────────────────────────────────────
        T_vec_dots = jnp.full((len(h_dots),), T_val)
        X_dots     = jnp.stack([T_vec_dots, h_dots], axis=1)  # (N,2): [T, h]
        Y_dots_nn  = predict(state.params, X_dots)             # (N,2): [F, Mz]
        axB.scatter(h_dots, Y_dots_nn[:, 0],
                    color=col, s=22, alpha=0.85, marker='o',
                    edgecolors='none', zorder=3, label='_nolegend_')
        axC.scatter(h_dots, Y_dots_nn[:, 1],
                    color=col, s=22, alpha=0.85, marker='o',
                    edgecolors='none', zorder=3, label='_nolegend_')

    # 범례 — 선만 표시 (색=온도 대응)
    for ax in (axB, axC):
        ax.legend(framealpha=0.25, labelcolor=WHITE, facecolor='#1a1a3a',
                  edgecolor='#444466', fontsize=9, loc='best')

    figB.tight_layout()
    figB.savefig(HERE / 'figB_free_energy_sweep.png', dpi=200,
                 bbox_inches='tight', facecolor=figB.get_facecolor())
    figC.tight_layout()
    figC.savefig(HERE / 'figC_magnetization_sweep.png', dpi=200,
                 bbox_inches='tight', facecolor=figC.get_facecolor())

    print("\nSaved: figA_learning_curve.png, figB_free_energy_sweep.png, "
          "figC_magnetization_sweep.png")
    plt.show()  # 3개 창 동시 표시
