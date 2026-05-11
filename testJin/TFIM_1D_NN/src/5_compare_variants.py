import csv
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tfim_exact import v_thermo_chi, v_thermo_f


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
FIGURES = ROOT / "figures" / "variant_comparison"

VARIANTS = {
    "baseline_derivative_blind": ROOT / "src" / "variants" / "baseline_derivative_blind" / "2_jax_model.py",
    "sobolev_critical": ROOT / "src" / "variants" / "sobolev_critical" / "2_jax_model.py",
    "balanced_peak_guard": ROOT / "src" / "variants" / "balanced_peak_guard" / "2_jax_model.py",
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    x = np.array([[float(r["T"]), float(r["h"]), float(r["inv_L"])] for r in rows], dtype=np.float32)
    y = np.array([[float(r["F_exact"]), float(r["Cv_exact"]), float(r["chi_exact"])] for r in rows], dtype=np.float32)
    return jnp.array(x), jnp.array(y)


def _base_metrics(prefix, pred, truth):
    err = pred - truth
    mse = jnp.mean(err * err)
    rmse = jnp.sqrt(mse)
    mae = jnp.mean(jnp.abs(err))
    centered_pred = pred - jnp.mean(pred)
    centered_truth = truth - jnp.mean(truth)
    corr = jnp.sum(centered_pred * centered_truth) / (
        jnp.sqrt(jnp.sum(centered_pred * centered_pred) * jnp.sum(centered_truth * centered_truth)) + 1e-12
    )
    truth_var = jnp.var(truth)
    r2 = 1.0 - mse / (truth_var + 1e-12)
    nrmse = rmse / (jnp.std(truth) + 1e-12)
    smape = jnp.mean(2.0 * jnp.abs(err) / (jnp.abs(pred) + jnp.abs(truth) + 1e-6))
    return {
        f"{prefix}_mse": float(mse),
        f"{prefix}_mae": float(mae),
        f"{prefix}_nrmse": float(nrmse),
        f"{prefix}_corr": float(corr),
        f"{prefix}_r2": float(r2),
        f"{prefix}_smape": float(smape),
        f"{prefix}_max_abs": float(jnp.max(jnp.abs(err))),
    }


def _critical_mask(x):
    return (jnp.abs(x[:, 1] - 1.0) <= 0.25) & (x[:, 0] <= 0.35)


def _peak_metrics(model, params, temperatures):
    h_line = jnp.linspace(0.1, 2.0, 400)
    rows = []
    for T in temperatures:
        T_vec = jnp.full_like(h_line, T)
        x_inf = jnp.stack([T_vec, h_line, jnp.zeros_like(h_line)], axis=1)
        pred = model.batched_observables(params, x_inf)[:, 2]
        truth = v_thermo_chi(T_vec, h_line)
        pred_idx = int(jnp.argmax(pred))
        truth_idx = int(jnp.argmax(truth))
        pred_peak = float(pred[pred_idx])
        truth_peak = float(truth[truth_idx])
        rows.append(
            {
                "T": float(T),
                "h_peak_pred": float(h_line[pred_idx]),
                "h_peak_exact": float(h_line[truth_idx]),
                "h_peak_abs_error": abs(float(h_line[pred_idx] - h_line[truth_idx])),
                "chi_peak_pred": pred_peak,
                "chi_peak_exact": truth_peak,
                "chi_peak_rel_error": abs(pred_peak - truth_peak) / (abs(truth_peak) + 1e-6),
            }
        )
    return rows


def _evaluate_variant(name, model, params, x_hold, y_hold, x_truth, y_truth):
    hold_obs = model.batched_observables(params, x_hold)
    truth_obs = model.batched_observables(params, x_truth)
    jax.block_until_ready(truth_obs)

    metrics = {}
    metrics.update(_base_metrics("holdout_F", hold_obs[:, 0], y_hold[:, 0]))
    metrics.update(_base_metrics("holdout_chi", hold_obs[:, 2], y_hold[:, 2]))
    metrics.update(_base_metrics("thermo_F", truth_obs[:, 0], y_truth[:, 0]))
    metrics.update(_base_metrics("thermo_chi", truth_obs[:, 2], y_truth[:, 2]))

    hold_crit = _critical_mask(x_hold)
    truth_crit = _critical_mask(x_truth)
    metrics.update(_base_metrics("holdout_critical_chi", hold_obs[hold_crit, 2], y_hold[hold_crit, 2]))
    metrics.update(_base_metrics("thermo_critical_chi", truth_obs[truth_crit, 2], y_truth[truth_crit, 2]))

    peak_rows = _peak_metrics(model, params, [0.05, 0.1, 0.2, 0.5, 1.0])
    metrics["mean_h_peak_abs_error"] = float(np.mean([r["h_peak_abs_error"] for r in peak_rows]))
    metrics["mean_chi_peak_rel_error"] = float(np.mean([r["chi_peak_rel_error"] for r in peak_rows]))
    low_t_peaks = [r for r in peak_rows if r["T"] <= 0.2]
    metrics["lowT_mean_h_peak_abs_error"] = float(np.mean([r["h_peak_abs_error"] for r in low_t_peaks]))
    metrics["lowT_mean_chi_peak_rel_error"] = float(np.mean([r["chi_peak_rel_error"] for r in low_t_peaks]))
    return metrics, peak_rows


def _write_summary(results):
    out = ROOT / "results_variant_comparison.md"
    headline_metrics = [
        "holdout_F_mse",
        "holdout_chi_mae",
        "holdout_chi_corr",
        "holdout_critical_chi_mae",
        "thermo_chi_mae",
        "thermo_chi_corr",
        "thermo_critical_chi_mae",
        "lowT_mean_h_peak_abs_error",
        "lowT_mean_chi_peak_rel_error",
        "mean_h_peak_abs_error",
        "mean_chi_peak_rel_error",
    ]
    with out.open("w", encoding="utf-8") as f:
        f.write("# TFIM NN 분기 비교 결과\n\n")
        f.write("MSE도 유지하지만, 상전이 학습 여부를 보려면 chi MAE/correlation과 피크 위치/높이 오차를 함께 봐야 합니다.\n\n")
        f.write("## 핵심 비교 지표\n\n")
        f.write("| 분기 | " + " | ".join(headline_metrics) + " |\n")
        f.write("|---|" + "|".join(["---:" for _ in headline_metrics]) + "|\n")
        for name, payload in results.items():
            metrics = payload["metrics"]
            vals = [f"{metrics[k]:.6e}" for k in headline_metrics]
            f.write(f"| {name} | " + " | ".join(vals) + " |\n")

        f.write("\n## 피크 진단\n\n")
        for name, payload in results.items():
            f.write(f"### {name}\n\n")
            f.write("| T | 예측 h_peak | 해석해 h_peak | h 절대오차 | 예측 chi_peak | 해석해 chi_peak | chi 상대오차 |\n")
            f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in payload["peaks"]:
                f.write(
                    f"| {row['T']:.3f} | {row['h_peak_pred']:.5f} | {row['h_peak_exact']:.5f} | "
                    f"{row['h_peak_abs_error']:.5f} | {row['chi_peak_pred']:.5f} | "
                    f"{row['chi_peak_exact']:.5f} | {row['chi_peak_rel_error']:.5f} |\n"
                )
            f.write("\n")
    return out


def _plot_comparison(loaded):
    FIGURES.mkdir(parents=True, exist_ok=True)
    h_line = jnp.linspace(0.1, 2.0, 360)
    temps = [0.05, 0.1, 0.2]
    fig, axes = plt.subplots(1, len(temps), figsize=(14, 4.2), sharey=True)
    for ax, T in zip(axes, temps):
        T_vec = jnp.full_like(h_line, T)
        exact = v_thermo_chi(T_vec, h_line)
        ax.plot(h_line, exact, color="black", lw=2.0, label="exact")
        for name, (model, params) in loaded.items():
            x_inf = jnp.stack([T_vec, h_line, jnp.zeros_like(h_line)], axis=1)
            pred = model.batched_observables(params, x_inf)[:, 2]
            ax.plot(h_line, pred, lw=1.5, label=name)
        ax.axvline(1.0, color="gray", ls=":", alpha=0.7)
        ax.set_title(f"T={T:g}")
        ax.set_xlabel("h")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("chi")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Susceptibility comparison by variant")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_variant_chi_curves.png", dpi=180)
    plt.close(fig)


def main():
    x_hold, y_hold = _load_csv(DATA / "ed_holdout_data.csv")
    x_truth, y_truth = _load_csv(DATA / "analytic_truth_data.csv")

    results = {}
    loaded = {}
    for name, model_path in VARIANTS.items():
        weights = MODELS / name / "saved_weights.pkl"
        if not weights.exists():
            print(f"{name} 건너뜀: weight 파일 없음 - {weights}")
            continue
        model = _load_module(f"model_{name}", model_path)
        params, metadata = model.load_params(weights)
        metrics, peaks = _evaluate_variant(name, model, params, x_hold, y_hold, x_truth, y_truth)
        results[name] = {"metadata": metadata, "metrics": metrics, "peaks": peaks}
        loaded[name] = (model, params)

    if not results:
        raise SystemExit("분기 weight 파일이 없습니다. 먼저 각 분기를 학습하세요.")

    summary = _write_summary(results)
    _plot_comparison(loaded)
    print(f"비교 요약 저장: {summary}")
    print(f"비교 그림 저장: {FIGURES / 'fig_variant_chi_curves.png'}")


if __name__ == "__main__":
    main()
