import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm import tqdm
from functools import partial

# [파트 1: 환경 설정]
# 계산 물리학에서는 정밀도가 생명입니다. 64비트 부동소수점을 활성화하여 
# 극저온에서의 수치적 불안정성과 미분 오차를 최소화합니다.
jax.config.update("jax_enable_x64", True)

# [파트 2: 신경망 아키텍처 - FSSNet]
# 이 네트워크는 (T, h, 1/L)을 입력받아 자유 에너지 F를 예측합니다.
class FSSNet(nn.Module):
    # 은닉층의 뉴런 개수 정의 (깊고 넓은 모델일수록 복잡한 상전이 곡면을 잘 학습함)
    features: tuple = (128, 128, 128, 128)

    @nn.compact
    def __call__(self, x):
        # x: [batch, 3] -> (T, h, inv_L)
        for feat in self.features:
            x = nn.Dense(feat)(x)
            # [핵심] Softplus 활성화 함수 사용
            # ReLU와 달리 모든 구간에서 매끄럽게 미분 가능하므로, 
            # 2차 미분값인 비열(Cv)이나 자화율(Chi)을 정확히 계산할 수 있습니다.
            x = jax.nn.softplus(x)
        
        # 마지막 출력은 스칼라 값인 자유 에너지 밀도 F
        x = nn.Dense(1)(x)
        return jnp.squeeze(x)

# [파트 3: 물리 관찰자 (Physics Observer)]
# JAX의 자동 미분을 활용하여 예측된 F로부터 다른 열역학적 물리량들을 유도합니다.
# 이 함수들이 Sobolev Loss의 핵심인 '기울기 정답지'를 생성합니다.
def derive_physics(model_apply, params, T, h, inv_L):
    """F로부터 S, M, Cv, Chi를 해석적으로 유도"""
    # 특정 (T, h, inv_L) 포인트에서의 F를 계산하는 함수 바인딩
    def f_fn(t, h_val, l_val):
        # 입력 형상을 [3]으로 맞춰서 모델에 전달
        return model_apply({'params': params}, jnp.array([t, h_val, l_val]))

    # 1차 미분: 엔트로피(S)와 자화(M)
    df_dT = jax.grad(f_fn, argnums=0)
    df_dh = jax.grad(f_fn, argnums=1)
    
    # 2차 미분: 비열(Cv)과 자화율(Chi)
    d2f_dT2 = jax.grad(df_dT, argnums=0)
    d2f_dh2 = jax.grad(df_dh, argnums=1)

    # 물리적 정의에 따른 값 계산
    F_pred = f_fn(T, h, inv_L)
    S_pred = -df_dT(T, h, inv_L)
    M_pred = -df_dh(T, h, inv_L)
    Cv_pred = -T * d2f_dT2(T, h, inv_L)
    Chi_pred = -d2f_dh2(T, h, inv_L)
    
    return jnp.array([F_pred, M_pred, Cv_pred, S_pred, Chi_pred])

# [파트 4: 손실 함수 - Sobolev Loss]
# 값(Value)뿐만 아니라 미분값(Derivative)의 오차를 동시에 줄입니다.
@partial(jax.jit, static_argnums=(1,))
def compute_loss(params, model_apply, batch_X, batch_Y, weights):
    # vmap을 사용하여 배치 전체에 대해 물리량 유도 과정을 병렬화
    v_derive = jax.vmap(lambda x: derive_physics(model_apply, params, x[0], x[1], x[2]))
    preds = v_derive(batch_X) # [batch, 5] (F, M, Cv, S, Chi)
    
    # MSE 오차 계산
    sq_errors = jnp.square(preds - batch_Y)
    # 각 물리량별 가중치(weights)를 적용하여 합산
    weighted_mse = jnp.mean(sq_errors * weights, axis=0)
    
    return jnp.sum(weighted_mse), preds

# [파트 5: 학습 상태 및 단계 정의]
def create_train_state(rng, learning_rate):
    model = FSSNet()
    params = model.init(rng, jnp.ones((3,)))['params']
    # Adam 최적화기 사용
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_X, batch_Y, weights):
    """한 번의 역전파를 통해 파라미터를 업데이트"""
    grad_fn = jax.value_and_grad(lambda p: compute_loss(p, state.apply_fn, batch_X, batch_Y, weights)[0])
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# [파트 6: 데이터 로딩 및 실행]
def main():
    # 데이터셋 경로 확인 (이전에 만든 csv 파일이 있어야 함)
    train_path = "dataset/tfim_train.csv"
    test_path = "dataset/tfim_test.csv"
    
    if not os.path.exists(train_path):
        print(f"에러: {train_path} 파일을 찾을 수 없습니다. 데이터 생성기를 먼저 실행하세요.")
        return

    print("데이터셋 로딩 중...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 입력: (T, h, inv_L) / 정답: (F, M, Cv, S, Chi)
    X_train = jnp.array(train_df[['T', 'h', 'inv_L']].values)
    Y_train = jnp.array(train_df[['F', 'M', 'Cv', 'S', 'Chi']].values)
    X_test = jnp.array(test_df[['T', 'h', 'inv_L']].values)
    Y_test = jnp.array(test_df[['F', 'M', 'Cv', 'S', 'Chi']].values)

    # [하이퍼파라미터 설정]
    LR = 1e-3
    EPOCHS = 500
    BATCH_SIZE = 512
    # 손실 함수 가중치 (F, M, Cv, S, Chi)
    # Chi(자화율)의 가중치를 높여 상전이 피크 학습 강도를 조절할 수 있습니다.
    LOSS_WEIGHTS = jnp.array([1.0, 1.0, 0.1, 1.0, 0.1]) 

    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, LR)

    train_size = X_train.shape[0]
    steps_per_epoch = train_size // BATCH_SIZE

    print(f"학습 시작... (총 데이터: {train_size}행)")
    for epoch in range(EPOCHS):
        # 에포크마다 데이터 셔플링
        rng, shuffle_rng = jax.random.split(rng)
        perms = jax.random.permutation(shuffle_rng, train_size)
        X_shuffled, Y_shuffled = X_train[perms], Y_train[perms]

        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            batch_X = X_shuffled[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            batch_Y = Y_shuffled[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            state, loss = train_step(state, batch_X, batch_Y, LOSS_WEIGHTS)
            epoch_loss += loss

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / steps_per_epoch
            # 테스트 세트(L=14, 16, inf)에 대한 외삽 오차 측정
            test_loss, _ = compute_loss(state.params, state.apply_fn, X_test, Y_test, LOSS_WEIGHTS)
            print(f"에포크 {epoch+1:03d} | Train Loss: {avg_loss:.6f} | Test(Extrapolation) Loss: {test_loss:.6f}")

    print("학습 완료! 이제 모델이 1/L -> 0 극한의 물리를 예측할 준비가 되었습니다.")

if __name__ == "__main__":
    main()