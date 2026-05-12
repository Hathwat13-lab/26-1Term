import csv
import importlib.util
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"


def _load_model_module():
    model_path = Path(__file__).resolve().parent / "2_jax_model.py"
    spec = importlib.util.spec_from_file_location("tfim_jax_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_model = _load_model_module()
batched_free_energy = _model.batched_free_energy
batched_observables = _model.batched_observables
init_mlp = _model.init_mlp
save_params = _model.save_params


def _load_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    x = np.array([[float(r["T"]), float(r["h"]), float(r["inv_L"])] for r in rows], dtype=np.float32)
    y = np.array([[float(r["F_exact"]), float(r["Cv_exact"]), float(r["chi_exact"])] for r in rows], dtype=np.float32)
    return jnp.array(x), jnp.array(y)


def _adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0)}


def _clip_grads(grads, max_norm=1.0):
    sq_norm = sum(jnp.sum(g * g) for g in jax.tree.leaves(grads))
    scale = jnp.minimum(1.0, max_norm / (jnp.sqrt(sq_norm) + 1e-8))
    return jax.tree.map(lambda g: g * scale, grads)


def _adam_update(params, grads, opt_state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    t = opt_state["t"] + 1
    m = jax.tree.map(lambda m, g: beta1 * m + (1.0 - beta1) * g, opt_state["m"], grads)
    v = jax.tree.map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), opt_state["v"], grads)
    m_hat = jax.tree.map(lambda a: a / (1.0 - beta1**t), m)
    v_hat = jax.tree.map(lambda a: a / (1.0 - beta2**t), v)
    params = jax.tree.map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat)
    return params, {"m": m, "v": v, "t": t}


def _critical_weight(x):
    T = x[:, 0]
    h = x[:, 1]
    inv_L = x[:, 2]
    h_focus = jnp.exp(-((h - 1.0) / 0.22) ** 2)
    low_t_focus = 1.0 / (1.0 + 10.0 * T)  # Sharper focus on low T
    large_l_focus = 1.0 / (1.0 + 8.0 * inv_L)
    ferro_focus = jax.nn.sigmoid(-10.0 * (h - 0.9)) # Focus on h < 0.9
    w = 1.0 + 6.0 * h_focus * low_t_focus + 3.0 * h_focus * large_l_focus + 3.0 * ferro_focus * low_t_focus
    return w / jnp.mean(w)


def _extrapolation_guardrail(params, xd, yd):
    x_inf = xd.at[:, 2].set(0.0)
    chi_inf = batched_observables(params, x_inf)[:, 2]
    finite_chi = yd[:, 2]
    allowed = jnp.maximum(finite_chi + 0.22, 1.18 * finite_chi + 0.05)
    critical = jnp.exp(-((xd[:, 1] - 1.0) / 0.25) ** 2) / (1.0 + 3.0 * xd[:, 0])
    
    # 1. Overshoot penalty (don't go too high above finite L)
    overshoot = jax.nn.relu(chi_inf - allowed)
    overshoot_loss = jnp.mean(critical * (overshoot / (0.15 + jnp.abs(allowed))) ** 2)
    
    # 2. Curvature Inversion Penalty (chi MUST be positive)
    # If chi_inf < 0, heavily penalize.
    inversion_loss = jnp.mean(jax.nn.relu(-chi_inf) ** 2) * 50.0
    
    return overshoot_loss + inversion_loss


def _loss(params, xb, yb, xd, yd, chi_weight, cv_weight, guardrail_weight):
    f_pred = batched_free_energy(params, xb)
    f_loss = jnp.mean((f_pred - yb[:, 0]) ** 2)

    obs_pred = batched_observables(params, xd)
    w = _critical_weight(xd)
    chi_scale = 0.15 + jnp.abs(yd[:, 2])
    cv_scale = 0.10 + jnp.abs(yd[:, 1])
    chi_loss = jnp.mean(w * ((obs_pred[:, 2] - yd[:, 2]) / chi_scale) ** 2)
    cv_loss = jnp.mean(w * ((obs_pred[:, 1] - yd[:, 1]) / cv_scale) ** 2)
    guardrail_loss = _extrapolation_guardrail(params, xd, yd)
    total = f_loss + chi_weight * chi_loss + cv_weight * cv_loss + guardrail_weight * guardrail_loss
    return total, (f_loss, chi_loss, cv_loss, guardrail_loss)


@jax.jit
def _train_step(params, opt_state, xb, yb, xd, yd, chi_weight, cv_weight, guardrail_weight):
    (loss, parts), grads = jax.value_and_grad(_loss, has_aux=True)(
        params, xb, yb, xd, yd, chi_weight, cv_weight, guardrail_weight
    )
    grads = _clip_grads(grads, max_norm=1.0)
    params, opt_state = _adam_update(params, grads, opt_state)
    return params, opt_state, loss, parts


@jax.jit
def _eval_losses(params, x, y):
    f_pred = batched_free_energy(params, x)
    obs_pred = batched_observables(params, x)
    f_mse = jnp.mean((f_pred - y[:, 0]) ** 2)
    chi_mae = jnp.mean(jnp.abs(obs_pred[:, 2] - y[:, 2]))
    return f_mse, chi_mae


def _sample_mixed(key, n, critical_idx, batch_size, critical_fraction):
    n_crit = int(batch_size * critical_fraction)
    n_any = batch_size - n_crit
    key_any, key_crit = jax.random.split(key)
    any_idx = jax.random.choice(key_any, n, (n_any,), replace=False)
    crit_pick = jax.random.choice(key_crit, critical_idx, (n_crit,), replace=True)
    return jnp.concatenate([any_idx, crit_pick])


def main():
    x_train, y_train = _load_csv(DATA / "ed_train_data.csv")
    x_hold, y_hold = _load_csv(DATA / "ed_holdout_data.csv")

    key = jax.random.PRNGKey(20260511)
    params = init_mlp(key)
    opt_state = _adam_init(params)

    n = x_train.shape[0]
    batch_size = 1536
    deriv_batch_size = 384
    epochs = 3200
    log_every = 50
    chi_weight = 4e-2
    cv_weight = 4e-3
    guardrail_weight = 2e-2
    logs = []
    critical_mask = (jnp.abs(x_train[:, 1] - 1.0) <= 0.25) & (x_train[:, 0] <= 0.35)
    critical_idx = jnp.where(critical_mask, size=int(jnp.sum(critical_mask)))[0]

    print(f"Training rows: {n}, holdout rows: {x_hold.shape[0]}")
    print(f"Critical-region derivative pool: {critical_idx.shape[0]} rows")
    start = time.time()
    for epoch in range(1, epochs + 1):
        key, kb, kd = jax.random.split(key, 3)
        idx = _sample_mixed(kb, n, critical_idx, batch_size, critical_fraction=0.20)
        didx = _sample_mixed(kd, n, critical_idx, deriv_batch_size, critical_fraction=0.55)
        params, opt_state, loss, parts = _train_step(
            params,
            opt_state,
            x_train[idx],
            y_train[idx],
            x_train[didx],
            y_train[didx],
            chi_weight,
            cv_weight,
            guardrail_weight,
        )

        if epoch % log_every == 0 or epoch == 1:
            train_mse, train_chi_mae = _eval_losses(params, x_train, y_train)
            hold_mse, hold_chi_mae = _eval_losses(params, x_hold, y_hold)
            jax.block_until_ready(hold_chi_mae)
            logs.append(
                (
                    epoch,
                    float(train_mse),
                    float(hold_mse),
                    float(train_chi_mae),
                    float(hold_chi_mae),
                    float(loss),
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                )
            )
            if epoch % 200 == 0 or epoch == 1:
                print(
                    f"epoch {epoch:4d} | total {float(loss):.6e} | "
                    f"train F {float(train_mse):.6e} | holdout F {float(hold_mse):.6e} | "
                    f"holdout chi MAE {float(hold_chi_mae):.6e}"
                )

    elapsed = time.time() - start
    metadata = {
        "variant": "hessian_feature_tuned_fixed",
        "story": "Feature Engineering + Convexity Penalty: Fixed negative chi bug by using correct scaling r2*log(r2) and adding a strict convexity loss to prevent curvature inversion.",
        "train_L": [6, 8, 10, 12],
        "holdout_L": [14, 16],
        "epochs": epochs,
        "chi_weight": chi_weight,
        "cv_weight": cv_weight,
        "guardrail_weight": guardrail_weight,
        "critical_sampling": {"batch_fraction": 0.20, "derivative_fraction": 0.55},
        "elapsed_seconds": elapsed,
    }
    save_params(MODELS / "saved_weights.pkl", params, metadata)

    FIGURES.mkdir(parents=True, exist_ok=True)
    arr = np.array(logs)
    np.savetxt(
        MODELS / "training_log.csv",
        arr,
        delimiter=",",
        header="epoch,train_F_mse,holdout_F_mse,train_chi_mae,holdout_chi_mae,total_batch_loss,batch_F_loss,batch_chi_loss,batch_Cv_loss,batch_guardrail_loss",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(arr[:, 0], arr[:, 1], label="train L=6,8,10,12")
    ax.semilogy(arr[:, 0], arr[:, 2], label="holdout L=14,16")
    ax.set_xlabel("epoch")
    ax.set_ylabel("free-energy MSE")
    ax.set_title("1D NN TFIM FSS surrogate learning curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figA_learning_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(arr[:, 0], arr[:, 3], label="train chi MAE")
    ax.semilogy(arr[:, 0], arr[:, 4], label="holdout chi MAE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("chi MAE")
    ax.set_title("Sobolev derivative learning curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figA2_chi_learning_curve.png", dpi=180)
    plt.close(fig)

    print(f"Saved model to {MODELS / 'saved_weights.pkl'}")
    print(f"Training finished in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
