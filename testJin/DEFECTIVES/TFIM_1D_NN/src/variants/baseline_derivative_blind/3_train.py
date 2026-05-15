import csv
import importlib.util
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
MODELS = ROOT / "models" / "baseline_derivative_blind"
FIGURES = ROOT / "figures" / "baseline_derivative_blind"


def _load_model_module():
    model_path = Path(__file__).resolve().parent / "2_jax_model.py"
    spec = importlib.util.spec_from_file_location("tfim_baseline_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = _load_model_module()


def _load_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    x = np.array([[float(r["T"]), float(r["h"]), float(r["inv_L"])] for r in rows], dtype=np.float32)
    y = np.array([[float(r["F_exact"]), float(r["Cv_exact"]), float(r["chi_exact"])] for r in rows], dtype=np.float32)
    return jnp.array(x), jnp.array(y)


def _adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0)}


def _adam_update(params, grads, opt_state, lr=2e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    t = opt_state["t"] + 1
    m = jax.tree.map(lambda m, g: beta1 * m + (1.0 - beta1) * g, opt_state["m"], grads)
    v = jax.tree.map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), opt_state["v"], grads)
    m_hat = jax.tree.map(lambda a: a / (1.0 - beta1**t), m)
    v_hat = jax.tree.map(lambda a: a / (1.0 - beta2**t), v)
    params = jax.tree.map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat)
    return params, {"m": m, "v": v, "t": t}


def _loss(params, xb, yb, xd, yd, chi_weight):
    f_pred = model.batched_free_energy(params, xb)
    f_loss = jnp.mean((f_pred - yb[:, 0]) ** 2)

    obs_pred = model.batched_observables(params, xd)
    chi_scale = jnp.maximum(jnp.std(yd[:, 2]), 1e-3)
    chi_loss = jnp.mean(((obs_pred[:, 2] - yd[:, 2]) / chi_scale) ** 2)
    return f_loss + chi_weight * chi_loss, (f_loss, chi_loss)


@jax.jit
def _train_step(params, opt_state, xb, yb, xd, yd, chi_weight):
    (loss, parts), grads = jax.value_and_grad(_loss, has_aux=True)(params, xb, yb, xd, yd, chi_weight)
    params, opt_state = _adam_update(params, grads, opt_state)
    return params, opt_state, loss, parts


@jax.jit
def _eval_losses(params, x, y):
    f_pred = model.batched_free_energy(params, x)
    obs_pred = model.batched_observables(params, x)
    return jnp.mean((f_pred - y[:, 0]) ** 2), jnp.mean(jnp.abs(obs_pred[:, 2] - y[:, 2]))


def main():
    x_train, y_train = _load_csv(DATA / "ed_train_data.csv")
    x_hold, y_hold = _load_csv(DATA / "ed_holdout_data.csv")

    key = jax.random.PRNGKey(20260511)
    params = model.init_mlp(key)
    opt_state = _adam_init(params)

    n = x_train.shape[0]
    batch_size = 1024
    deriv_batch_size = 160
    epochs = 1800
    chi_weight = 2e-3
    log_every = 50
    logs = []

    print(f"Baseline training rows: {n}, holdout rows: {x_hold.shape[0]}")
    start = time.time()
    for epoch in range(1, epochs + 1):
        key, kb, kd = jax.random.split(key, 3)
        idx = jax.random.choice(kb, n, (batch_size,), replace=False)
        didx = jax.random.choice(kd, n, (deriv_batch_size,), replace=False)
        params, opt_state, loss, parts = _train_step(
            params, opt_state, x_train[idx], y_train[idx], x_train[didx], y_train[didx], chi_weight
        )
        if epoch % log_every == 0 or epoch == 1:
            train_f, train_chi = _eval_losses(params, x_train, y_train)
            hold_f, hold_chi = _eval_losses(params, x_hold, y_hold)
            jax.block_until_ready(hold_chi)
            logs.append((epoch, float(train_f), float(hold_f), float(train_chi), float(hold_chi), float(loss)))
            if epoch % 200 == 0 or epoch == 1:
                print(
                    f"epoch {epoch:4d} | holdout F {float(hold_f):.6e} | "
                    f"holdout chi MAE {float(hold_chi):.6e}"
                )

    metadata = {
        "variant": "baseline_derivative_blind",
        "story": "Smooth MLP mostly trained on F; weak chi penalty, no critical sampling.",
        "train_L": [6, 8, 10, 12],
        "holdout_L": [14, 16],
        "epochs": epochs,
        "chi_weight": chi_weight,
        "elapsed_seconds": time.time() - start,
    }
    model.save_params(MODELS / "saved_weights.pkl", params, metadata)

    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    arr = np.array(logs)
    np.savetxt(
        MODELS / "training_log.csv",
        arr,
        delimiter=",",
        header="epoch,train_F_mse,holdout_F_mse,train_chi_mae,holdout_chi_mae,total_batch_loss",
        comments="",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(arr[:, 0], arr[:, 2], label="holdout F MSE")
    ax.semilogy(arr[:, 0], arr[:, 4], label="holdout chi MAE")
    ax.set_xlabel("epoch")
    ax.set_title("Baseline derivative-blind learning")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curve.png", dpi=180)
    plt.close(fig)
    print(f"Saved baseline model to {MODELS / 'saved_weights.pkl'}")


if __name__ == "__main__":
    main()
