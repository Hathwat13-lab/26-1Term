import csv
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"

sys.path.append(str(SRC))
import tfim_exact

def _load_model_module():
    model_path = SRC / "2_jax_model.py"
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

def plot_learning_curves():
    log_path = MODELS / "training_log.csv"
    if not log_path.exists():
        print("Training log not found.")
        return
    
    data = np.genfromtxt(log_path, delimiter=",", names=True)
    epochs = data["epoch"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: MSE and MAE
    ax = axes[0]
    ax.plot(epochs, data["train_F_mse"], label="Train F MSE", alpha=0.8)
    ax.plot(epochs, data["holdout_F_mse"], label="Holdout F MSE", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Free Energy Learning Curve")
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    # Right: Chi MAE
    ax = axes[1]
    ax.plot(epochs, data["train_chi_mae"], label="Train chi MAE", alpha=0.8)
    ax.plot(epochs, data["holdout_chi_mae"], label="Holdout chi MAE", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.set_title("Susceptibility Learning Curve")
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    fig.tight_layout()
    fig.savefig(FIGURES / "diag_A_learning_curves.png", dpi=150)
    plt.close(fig)

def plot_thermo_observables(params):
    h_line = jnp.linspace(0.1, 2.0, 400)
    temps = [0.05, 0.1, 0.2, 0.5]
    
    # Observables return format: [F, Cv, chi, M, S]
    # Indices: 0:F, 1:Cv, 2:chi, 3:M, 4:S
    labels = ["Free Energy (F)", "Specific Heat (Cv)", "Susceptibility (chi)", "Magnetization (M)", "Entropy (S)"]
    indices = [0, 1, 2, 3, 4]
    
    fig, axes = plt.subplots(len(indices), 1, figsize=(8, 15), sharex=True)
    cmap = plt.get_cmap("tab10")
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        for j, T in enumerate(temps):
            T_vec = jnp.full_like(h_line, T)
            x_inf = jnp.stack([T_vec, h_line, jnp.zeros_like(h_line)], axis=1)
            
            # NN predictions
            preds = model.batched_observables(params, x_inf)
            y_nn = preds[:, idx]
            
            # Exact
            if idx == 0: y_ex = tfim_exact.v_thermo_f(T_vec, h_line)
            elif idx == 1: y_ex = tfim_exact.v_thermo_cv(T_vec, h_line)
            elif idx == 2: y_ex = tfim_exact.v_thermo_chi(T_vec, h_line)
            elif idx == 3: y_ex = tfim_exact.v_thermo_m(T_vec, h_line)
            elif idx == 4: y_ex = tfim_exact.v_thermo_s(T_vec, h_line)
            
            ax.plot(h_line, y_ex, color="black", lw=1.0, alpha=0.5, ls="--")
            ax.plot(h_line, y_nn, color=cmap(j), label=f"T={T:g}" if i==0 else None)
            
        ax.set_ylabel(labels[i])
        ax.axvline(1.0, color="k", ls=":", alpha=0.3)
        ax.grid(True, alpha=0.2)
        if i == 0: ax.legend(loc="upper right", fontsize=8)
        
    axes[-1].set_xlabel("Magnetic Field (h)")
    fig.suptitle("Thermodynamic Observables: NN (colors) vs Exact (dashed black)")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(FIGURES / "diag_B_thermo_observables.png", dpi=150)
    plt.close(fig)

def plot_chi_heatmap(params):
    T_grid = jnp.linspace(0.01, 0.8, 100)
    h_grid = jnp.linspace(0.1, 2.0, 150)
    TT, HH = jnp.meshgrid(T_grid, h_grid, indexing="ij")
    x_inf = jnp.stack([TT.reshape(-1), HH.reshape(-1), jnp.zeros(TT.size)], axis=1)
    
    chi_nn = model.batched_observables(params, x_inf)[:, 2].reshape(TT.shape)
    chi_ex = tfim_exact.v_thermo_chi(TT.reshape(-1), HH.reshape(-1)).reshape(TT.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    vmax = float(jnp.percentile(chi_ex, 98))
    
    for ax, data, title in zip(axes, [chi_ex, chi_nn], ["Exact chi", "NN Hessian chi"]):
        im = ax.pcolormesh(h_grid, T_grid, data, shading="auto", vmin=0, vmax=vmax, cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("h")
        ax.axvline(1.0, color="white", ls=":", alpha=0.5)
        
    axes[0].set_ylabel("T")
    fig.colorbar(im, ax=axes, label="chi")
    fig.suptitle("Critical Fan Extrapolation Heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES / "diag_C_chi_fan_heatmap.png", dpi=150)
    plt.close(fig)

def main():
    params, metadata = model.load_params(MODELS / "saved_weights.pkl")
    FIGURES.mkdir(parents=True, exist_ok=True)
    
    print("Generating learning curves...")
    plot_learning_curves()
    
    print("Generating thermodynamic observables...")
    plot_thermo_observables(params)
    
    print("Generating chi heatmap...")
    plot_chi_heatmap(params)
    
    print("Diagnostics complete.")

if __name__ == "__main__":
    main()
