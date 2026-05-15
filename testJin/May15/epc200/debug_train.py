import os
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import numpy as np
from flax.training import train_state

# JAX 64비트 활성화
jax.config.update("jax_enable_x64", True)

# 분리한 모델 모듈에서 가져오기
from model import FSSNet, sobolev_loss_fn

def create_train_state(rng, learning_rate):
    model = FSSNet()
    dummy_input = jnp.ones((3,))
    params = model.init(rng, dummy_input)['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_inputs, batch_targets, weights):
    def loss_fn(params):
        loss, _ = sobolev_loss_fn(params, state.apply_fn, batch_inputs, batch_targets, weights)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def main():
    print("DEBUG MODE: Loading 1% of datasets...")
    train_df = pd.read_csv("dataset/tfim_train.csv").sample(frac=0.01, random_state=42)
    test_df = pd.read_csv("dataset/tfim_test.csv").sample(frac=0.01, random_state=42)

    X_train = jnp.array(train_df[['T', 'h', 'inv_L']].values)
    Y_train = jnp.array(train_df[['F', 'M', 'Cv', 'S', 'Chi']].values)
    X_test = jnp.array(test_df[['T', 'h', 'inv_L']].values)
    Y_test = jnp.array(test_df[['F', 'M', 'Cv', 'S', 'Chi']].values)

    loss_weights = jnp.array([1.0, 1.0, 0.1, 1.0, 0.1])
    learning_rate = 1e-3
    num_epochs = 10
    batch_size = 128

    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, learning_rate)

    print("Starting Debug Training...")
    for epoch in range(num_epochs):
        state, loss = train_step(state, X_train, Y_train, loss_weights)
        print(f"Epoch {epoch+1:02d} | Loss: {loss:.6f}")

    print("Debug Training Complete!")

if __name__ == "__main__":
    main()
