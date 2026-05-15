import os
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import numpy as np
from flax.training import train_state
from tqdm import tqdm

# JAX 64비트 활성화
jax.config.update("jax_enable_x64", True)

# 분리한 모델 모듈에서 가져오기
from model import FSSNet, sobolev_loss_fn

def create_train_state(rng, learning_rate):
    """모델 파라미터와 Optax 최적화기를 묶어 TrainState 생성"""
    model = FSSNet()
    dummy_input = jnp.ones((3,)) # (T, h, inv_L)
    params = model.init(rng, dummy_input)['params']
    
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_inputs, batch_targets, weights):
    """한 배치의 학습을 수행하고 파라미터를 업데이트"""
    def loss_fn(params):
        loss, _ = sobolev_loss_fn(params, state.apply_fn, batch_inputs, batch_targets, weights)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def eval_step(state, batch_inputs, batch_targets, weights):
    """평가 데이터에 대한 Loss 측정"""
    loss, _ = sobolev_loss_fn(state.params, state.apply_fn, batch_inputs, batch_targets, weights)
    return loss

def main():
    # 데이터 로드
    print("Loading datasets from dataset folder...")
    train_df = pd.read_csv("dataset/tfim_train.csv")
    test_df = pd.read_csv("dataset/tfim_test.csv")

    X_train = jnp.array(train_df[['T', 'h', 'inv_L']].values)
    Y_train = jnp.array(train_df[['F', 'M', 'Cv', 'S', 'Chi']].values)
    
    X_test = jnp.array(test_df[['T', 'h', 'inv_L']].values)
    Y_test = jnp.array(test_df[['F', 'M', 'Cv', 'S', 'Chi']].values)

    # 물리량별 가중치 (F, M, Cv, S, Chi 순서)
    loss_weights = jnp.array([1.0, 1.0, 0.1, 1.0, 0.1])

    # 하이퍼파라미터
    learning_rate = 1e-3
    num_epochs = 2000
    batch_size = 1024

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, learning_rate)

    train_size = X_train.shape[0]
    steps_per_epoch = train_size // batch_size

    print(f"Starting Training... (Train size: {train_size}, Test size: {X_test.shape[0]})")
    
    import pickle
    history = []
    for epoch in range(num_epochs):
        rng, shuffle_rng = jax.random.split(rng)
        perms = jax.random.permutation(shuffle_rng, train_size)
        X_train_shuffled = X_train[perms]
        Y_train_shuffled = Y_train[perms]

        epoch_loss = 0.0
        for i in range(steps_per_epoch):
            batch_X = X_train_shuffled[i*batch_size : (i+1)*batch_size]
            batch_Y = Y_train_shuffled[i*batch_size : (i+1)*batch_size]
            
            state, loss = train_step(state, batch_X, batch_Y, loss_weights)
            epoch_loss += loss

        # 5 에폭마다 리포트 출력 및 기록
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_train_loss = epoch_loss / steps_per_epoch
            test_loss = eval_step(state, X_test, Y_test, loss_weights)
            print(f"Epoch {epoch+1:04d} | Train Loss: {avg_train_loss:.6f} | Test(Extrapolation) Loss: {test_loss:.6f}")
            history.append([epoch + 1, float(avg_train_loss), float(test_loss)])

        # 50 에폭마다 파라미터 백업 저장
        if (epoch + 1) % 50 == 0:
            with open(f"fss_model_epoch_{epoch+1}.params", "wb") as f:
                pickle.dump(state.params, f)
            print(f"Checkpoint saved: epoch {epoch+1}")

    print("Training Complete!")
    
    # 히스토리 저장
    history_df = pd.DataFrame(history, columns=['epoch', 'train_loss', 'test_loss'])
    history_df.to_csv("loss_history.csv", index=False)
    print("Loss history saved to loss_history.csv")

    # 최종 모델 파라미터 저장
    with open("fss_model.params", "wb") as f:
        pickle.dump(state.params, f)
    print("Final model parameters saved to fss_model.params")

if __name__ == "__main__":
    main()
