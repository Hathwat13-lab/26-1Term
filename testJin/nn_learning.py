import time
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import matplotlib.pyplot as plt
from jax import random
from exact_solution import exact_free_energy

# ==========================================
# 1. 모델 정의 (Surrogate Model)
# ==========================================
class FreeEnergySurrogate(nn.Module):
    hidden_dims: int = 64
    
    @nn.compact
    def __call__(self, x):
        # x shape: (batch_size, 2) -> (T, h)
        x = nn.Dense(self.hidden_dims)(x)
        x = nn.softplus(x)
        x = nn.Dense(self.hidden_dims)(x)
        x = nn.softplus(x)
        x = nn.Dense(self.hidden_dims)(x)
        x = nn.softplus(x)
        F_pred = nn.Dense(1)(x)
        return F_pred

# ==========================================
# 2. 데이터셋 생성 및 분할
# ==========================================
def generate_dataset(num_samples=10000):
    print(f"\n--- 데이터 생성 ---")
    print(f"총 {num_samples}개의 레이블 데이터(T, h -> F)를 생성합니다...")
    
    vmap_free_energy_2d = jax.jit(jax.vmap(exact_free_energy, in_axes=(0, 0)))
    
    key = random.PRNGKey(42)
    key_t, key_h, key_shuffle = random.split(key, 3)
    
    # T는 0.01 ~ 2.0, h는 0.0 ~ 2.0 사이에서 무작위 추출
    T_samples = random.uniform(key_t, shape=(num_samples, 1), minval=0.01, maxval=2.0)
    h_samples = random.uniform(key_h, shape=(num_samples, 1), minval=0.0, maxval=2.0)
    
    X = jnp.concatenate([T_samples, h_samples], axis=1)
    
    # JIT 컴파일 시간 분리 (최초 1회 실행)
    _ = vmap_free_energy_2d(h_samples[:1, 0], T_samples[:1, 0])
    
    # 실제 데이터 생성 (시간 측정)
    start_time = time.time()
    Y = vmap_free_energy_2d(h_samples[:, 0], T_samples[:, 0]).reshape(-1, 1)
    jax.block_until_ready(Y)
    gen_time = time.time() - start_time
    print(f"데이터 생성 소요 시간: {gen_time:.4f} 초")
    
    # 데이터 셔플 및 8:2 분할
    indices = random.permutation(key_shuffle, num_samples)
    X, Y = X[indices], Y[indices]
    
    split_idx = int(num_samples * 0.8)
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_test, Y_test = X[split_idx:], Y[split_idx:]
    
    print(f"Train 데이터: {X_train.shape[0]}개 (80%)")
    print(f"Test 데이터: {X_test.shape[0]}개 (20%)")
    return X_train, Y_train, X_test, Y_test

# ==========================================
# 3. 학습 함수 정의
# ==========================================
@jax.jit
def train_step(state, batch_X, batch_Y):
    def loss_fn(params):
        preds = state.apply_fn(params, batch_X)
        loss = jnp.mean((preds - batch_Y) ** 2)
        return loss, preds
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, preds), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# ==========================================
# 4. 메인 실행부 (학습 및 평가)
# ==========================================
if __name__ == "__main__":
    from flax.training.train_state import TrainState
    
    # 데이터 준비
    X_train, Y_train, X_test, Y_test = generate_dataset(num_samples=10000)
    
    # 모델 및 옵티마이저 초기화
    model = FreeEnergySurrogate()
    key = random.PRNGKey(0)
    dummy_x = jnp.ones((1, 2))
    params = model.init(key, dummy_x)
    
    # Adam 옵티마이저 (학습률 0.005)
    tx = optax.adam(learning_rate=5e-3)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    # 신경망 학습 (Full-batch Gradient Descent for simplicity/speed)
    num_epochs = 3000
    print(f"\n--- 모델 학습 ---")
    print(f"총 {num_epochs} 에포크 동안 신경망 학습을 시작합니다 (Full-batch)...")
    
    start_train_time = time.time()
    for epoch in range(num_epochs):
        state, loss = train_step(state, X_train, Y_train)
        if (epoch+1) % 500 == 0:
            print(f"Epoch {epoch+1:4d} | Training MSE Loss: {loss:.6f}")
    
    jax.block_until_ready(state.params)
    train_time = time.time() - start_train_time
    print(f"학습 소요 시간: {train_time:.4f} 초")
    
    # 연산 시간 비교 및 정확도 평가
    print("\n--- 정확도 및 연산 속도 비교 (Test 데이터 2000개 기준) ---")
    
    # a. Exact Solution 시간 측정 (JIT 워밍업 후)
    vmap_free_energy_2d = jax.jit(jax.vmap(exact_free_energy, in_axes=(0, 0)))
    _ = vmap_free_energy_2d(X_test[:1, 1], X_test[:1, 0])  # JIT 워밍업
    
    start_exact = time.time()
    Y_exact = vmap_free_energy_2d(X_test[:, 1], X_test[:, 0]).reshape(-1, 1)
    jax.block_until_ready(Y_exact)
    exact_time = time.time() - start_exact
    
    # b. 신경망(NN) 추론 시간 측정 (JIT 워밍업 후)
    @jax.jit
    def predict(params, x):
        return model.apply(params, x)
    
    _ = predict(state.params, X_test[:1])  # JIT 워밍업
    
    start_nn = time.time()
    Y_pred = predict(state.params, X_test)
    jax.block_until_ready(Y_pred)
    nn_time = time.time() - start_nn
    
    # 평가 지표 산출
    test_mse = jnp.mean((Y_pred - Y_test) ** 2)
    test_mae = jnp.mean(jnp.abs(Y_pred - Y_test))
    
    print(f"Test MSE (오차 제곱 평균):  {test_mse:.6f}")
    print(f"Test MAE (절대 오차 평균):  {test_mae:.6f}")
    print(f"Exact Solution 연산 시간 : {exact_time:.6f} 초")
    print(f"NN Surrogate 추론 시간   : {nn_time:.6f} 초")
    
    if nn_time > 0:
        speedup = exact_time / nn_time
        print(f"-> NN 모델이 Exact 모델보다 약 {speedup:.1f}배 빠름!")

    # 결과 시각화 플롯 (실제값 vs 예측값)
    plt.figure(figsize=(7, 6))
    plt.scatter(Y_test, Y_pred, alpha=0.3, s=15, color='royalblue', label='Test Data Points')
    
    # y=x 기준선 (완벽한 예측)
    min_val = min(float(jnp.min(Y_test)), float(jnp.min(Y_pred)))
    max_val = max(float(jnp.max(Y_test)), float(jnp.max(Y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction ($y=x$)')
    
    plt.title('Test Set Evaluation: Exact vs NN Predicted Free Energy')
    plt.xlabel('Exact Free Energy (Target)')
    plt.ylabel('NN Predicted Free Energy (Prediction)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('nn_surrogate_accuracy.png', dpi=300)
    print("Saved nn_surrogate_accuracy.png")
    plt.show()
