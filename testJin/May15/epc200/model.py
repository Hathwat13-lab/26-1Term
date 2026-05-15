import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial

class FSSNet(nn.Module):
    """
    Finite-Size Scaling Neural Network (FSSNet)
    입력: (T, h, inv_L)
    출력: 자유 에너지 밀도 F
    """
    features: tuple = (64, 64, 64, 64)

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = jax.nn.softplus(x) # 2차 미분을 위해 매끄러운 softplus 사용
        x = nn.Dense(1)(x)
        return jnp.squeeze(x)

def get_physics_quantities(model_apply, params, T, h, inv_L):
    """
    JAX 자동 미분을 통해 모델로부터 F, M, Cv, S, Chi를 유도합니다.
    """
    def f_bind(t_val, h_val, l_val):
        return model_apply({'params': params}, jnp.array([t_val, h_val, l_val]))

    # 1차 및 2차 미분 정의
    df_dT = jax.grad(f_bind, argnums=0)
    df_dh = jax.grad(f_bind, argnums=1)
    d2f_dT2 = jax.grad(df_dT, argnums=0)
    d2f_dh2 = jax.grad(df_dh, argnums=1)

    # 예측값 계산
    F_pred = f_bind(T, h, inv_L)
    S_pred = -df_dT(T, h, inv_L)
    M_pred = -df_dh(T, h, inv_L)
    Cv_pred = -T * d2f_dT2(T, h, inv_L)
    Chi_pred = -d2f_dh2(T, h, inv_L)
    
    return jnp.array([F_pred, M_pred, Cv_pred, S_pred, Chi_pred])

@partial(jax.jit, static_argnums=(1,))
def sobolev_loss_fn(params, model_apply, batch_inputs, batch_targets, weights):
    """
    Sobolev Loss: 함수값(F)뿐만 아니라 미분값(M, Cv, S, Chi)의 오차도 함께 최소화합니다.
    """
    # vmap을 통해 배치 처리
    pred_fn = jax.vmap(lambda t, h, l: get_physics_quantities(model_apply, params, t, h, l))
    preds = pred_fn(batch_inputs[:, 0], batch_inputs[:, 1], batch_inputs[:, 2])
    
    errors = jnp.square(preds - batch_targets)
    weighted_mse = jnp.mean(errors * weights, axis=0)
    return jnp.sum(weighted_mse), preds
