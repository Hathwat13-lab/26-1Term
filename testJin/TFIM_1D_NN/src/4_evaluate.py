import csv
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from tfim_exact import v_thermo_chi, v_thermo_f


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


model = _load_model_module()


def _load_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    x = np.array([[float(r["T"]), float(r["h"]), float(r["inv_L"])] for r in rows], dtype=np.float32)
    y = np.array([[float(r["F_exact"]), float(r["Cv_exact"]), float(r["chi_exact"])] for r in rows], dtype=np.float32)
    L = np.array([int(r["L"]) for r in rows], dtype=np.int32)
    return jnp.array(x), jnp.array(y), L


def _metrics(name, pred, truth):
    err = pred - truth
    return {
        f"{name}_mse": float(jnp.mean(err * err)),
        f"{name}_mae": float(jnp.mean(jnp.abs(err))),
        f"{name}_max_abs": float(jnp.max(jnp.abs(err))),
    }


def _critical_mask(x):
    return (jnp.abs(x[:, 1] - 1.0) <= 0.25) & (x[:, 0] <= 0.35)


def _peak_table(params, temperatures):
    h_line = jnp.linspace(0.1, 2.0, 300)
    rows = []
    for T in temperatures:
        T_vec = jnp.full_like(h_line, T)
        x_inf = jnp.stack([T_vec, h_line, jnp.zeros_like(h_line)], axis=1)
        nn_chi = model.batched_observables(params, x_inf)[:, 2]
        ex_chi = v_thermo_chi(T_vec, h_line)
        nn_idx = int(jnp.argmax(nn_chi))
        ex_idx = int(jnp.argmax(ex_chi))
        rows.append(
            {
                "T": float(T),
                "nn_h_peak": float(h_line[nn_idx]),
                "exact_h_peak": float(h_line[ex_idx]),
                "nn_chi_max": float(nn_chi[nn_idx]),
                "exact_chi_max": float(ex_chi[ex_idx]),
            }
        )
    return rows


def main():
    params, metadata = model.load_params(MODELS / "saved_weights.pkl")
    x_train, y_train, _ = _load_csv(DATA / "ed_train_data.csv")
    x_hold, y_hold, L_hold = _load_csv(DATA / "ed_holdout_data.csv")
    x_truth, y_truth, _ = _load_csv(DATA / "analytic_truth_data.csv")

    pred_train_f = model.batched_free_energy(params, x_train)
    pred_hold_f = model.batched_free_energy(params, x_hold)
    pred_hold_obs = model.batched_observables(params, x_hold)
    pred_truth_obs = model.batched_observables(params, x_truth)
    jax.block_until_ready(pred_truth_obs)

    metrics = {}
    metrics.update(_metrics("train_F", pred_train_f, y_train[:, 0]))
    metrics.update(_metrics("holdout_F", pred_hold_f, y_hold[:, 0]))
    metrics.update(_metrics("holdout_chi", pred_hold_obs[:, 2], y_hold[:, 2]))
    metrics.update(_metrics("thermo_F", pred_truth_obs[:, 0], y_truth[:, 0]))
    metrics.update(_metrics("thermo_chi", pred_truth_obs[:, 2], y_truth[:, 2]))
    hold_crit = _critical_mask(x_hold)
    truth_crit = _critical_mask(x_truth)
    metrics.update(_metrics("holdout_critical_chi", pred_hold_obs[hold_crit, 2], y_hold[hold_crit, 2]))
    metrics.update(_metrics("thermo_critical_chi", pred_truth_obs[truth_crit, 2], y_truth[truth_crit, 2]))

    peak_rows = _peak_table(params, [0.05, 0.1, 0.2, 0.5, 1.0])
    FIGURES.mkdir(parents=True, exist_ok=True)

    h_line = jnp.linspace(0.1, 2.0, 300)
    temps = [0.05, 0.1, 0.2, 0.5, 1.0]
    cmap = plt.get_cmap("viridis", len(temps))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, T in enumerate(temps):
        T_vec = jnp.full_like(h_line, T)
        x_inf = jnp.stack([T_vec, h_line, jnp.zeros_like(h_line)], axis=1)
        ax.plot(h_line, v_thermo_f(T_vec, h_line), color=cmap(i), lw=1.8, label=f"T={T:g} exact")
        ax.scatter(h_line[::12], model.batched_free_energy(params, x_inf)[::12], color=cmap(i), s=14, marker="o")
    ax.axvline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_xlabel("h")
    ax.set_ylabel("F")
    ax.set_title("Thermodynamic-limit extrapolation: exact line vs NN dots")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figB_free_energy_extrapolation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, T in enumerate(temps):
        T_vec = jnp.full_like(h_line, T)
        x_inf = jnp.stack([T_vec, h_line, jnp.zeros_like(h_line)], axis=1)
        ax.plot(h_line, v_thermo_chi(T_vec, h_line), color=cmap(i), lw=1.8, label=f"T={T:g} exact")
        ax.scatter(h_line[::12], model.batched_observables(params, x_inf)[::12, 2], color=cmap(i), s=14, marker="o")
    ax.axvline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_xlabel("h")
    ax.set_ylabel("chi")
    ax.set_title("Derivative check: chi from NN Hessian")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figC_susceptibility_derivative_check.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for L in sorted(set(L_hold.tolist())):
        mask = L_hold == L
        err = np.asarray(pred_hold_f[mask] - y_hold[mask, 0])
        ax.hist(err, bins=40, alpha=0.55, label=f"L={L}")
    ax.set_xlabel("F_pred - F_exact")
    ax.set_ylabel("count")
    ax.set_title("Holdout finite-size error distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "figD_holdout_error_hist.png", dpi=180)
    plt.close(fig)

    T_grid = jnp.linspace(0.05, 1.0, 120)
    h_grid = jnp.linspace(0.1, 2.0, 180)
    TT, HH = jnp.meshgrid(T_grid, h_grid, indexing="ij")
    x_inf = jnp.stack([TT.reshape(-1), HH.reshape(-1), jnp.zeros(TT.size)], axis=1)
    chi_nn = model.batched_observables(params, x_inf)[:, 2].reshape(TT.shape)
    chi_ex = v_thermo_chi(TT.reshape(-1), HH.reshape(-1)).reshape(TT.shape)
    vmax = float(jnp.percentile(chi_ex, 98))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, zz, title in zip(axes, [chi_ex, chi_nn], ["Exact chi", "NN Hessian chi"]):
        im = ax.pcolormesh(h_grid, T_grid, zz, shading="auto", vmin=0.0, vmax=vmax, cmap="magma")
        ax.axvline(1.0, color="white", ls=":", alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel("h")
        ax.grid(False)
    axes[0].set_ylabel("T")
    fig.colorbar(im, ax=axes, label="chi")
    fig.suptitle("Critical fan check in the thermodynamic-limit extrapolation")
    fig.savefig(FIGURES / "figE_chi_critical_fan_heatmap.png", dpi=180)
    plt.close(fig)

    with (ROOT / "results_summary.md").open("w", encoding="utf-8") as f:
        f.write("# 1D NN TFIM FSS 신경망 결과 요약\n\n")
        f.write(f"설정: `{metadata}`\n\n")
        f.write("## 주요 지표\n\n")
        for k, v in metrics.items():
            f.write(f"- `{k}`: {v:.8e}\n")
        f.write("\n## 자기감수율 피크 위치\n\n")
        f.write("| T | NN h_peak | 해석해 h_peak | NN chi_max | 해석해 chi_max |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for row in peak_rows:
            f.write(
                f"| {row['T']:.3f} | {row['nn_h_peak']:.5f} | {row['exact_h_peak']:.5f} | "
                f"{row['nn_chi_max']:.5f} | {row['exact_chi_max']:.5f} |\n"
            )

    print("평가 지표")
    for k, v in metrics.items():
        print(f"{k}: {v:.8e}")
    print(f"요약 저장: {ROOT / 'results_summary.md'}")


if __name__ == "__main__":
    main()
