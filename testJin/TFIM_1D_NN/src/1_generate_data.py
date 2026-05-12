import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from tfim_exact import vectorized_finite_observables, v_thermo_chi, v_thermo_cv, v_thermo_f


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TRAIN_L = (6, 8, 10, 12)
HOLDOUT_L = (14, 16)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["T", "h", "L", "inv_L", "F_exact", "Cv_exact", "chi_exact", "split"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _grid():
    h_base = jnp.linspace(0.1, 2.0, 49)
    h_crit = jnp.linspace(0.82, 1.18, 25)
    h_vals = jnp.unique(jnp.concatenate([h_base, h_crit]))

    T_base = jnp.linspace(0.05, 1.0, 33)
    T_low = jnp.linspace(0.05, 0.25, 13)
    T_extra = jnp.array([0.001, 0.01])
    T_vals = jnp.unique(jnp.concatenate([T_base, T_low, T_extra]))
    return T_vals, h_vals


def _rows_for_L(L, split, T_vals, h_vals):
    TT, HH = jnp.meshgrid(T_vals, h_vals, indexing="ij")
    T_flat = TT.reshape(-1)
    h_flat = HH.reshape(-1)

    vf, vcv, vchi = vectorized_finite_observables(L)
    F = vf(T_flat, h_flat)
    Cv = vcv(T_flat, h_flat)
    chi = vchi(T_flat, h_flat)
    jax.block_until_ready(F)

    rows = []
    for T, h, f, cv, c in zip(T_flat, h_flat, F, Cv, chi):
        rows.append(
            {
                "T": float(T),
                "h": float(h),
                "L": int(L),
                "inv_L": 1.0 / float(L),
                "F_exact": float(f),
                "Cv_exact": float(cv),
                "chi_exact": float(c),
                "split": split,
            }
        )
    return rows


def _truth_rows(T_vals, h_vals):
    TT, HH = jnp.meshgrid(T_vals, h_vals, indexing="ij")
    T_flat = TT.reshape(-1)
    h_flat = HH.reshape(-1)
    F = v_thermo_f(T_flat, h_flat)
    Cv = v_thermo_cv(T_flat, h_flat)
    chi = v_thermo_chi(T_flat, h_flat)
    jax.block_until_ready(F)

    rows = []
    for T, h, f, cv, c in zip(T_flat, h_flat, F, Cv, chi):
        rows.append(
            {
                "T": float(T),
                "h": float(h),
                "L": 0,
                "inv_L": 0.0,
                "F_exact": float(f),
                "Cv_exact": float(cv),
                "chi_exact": float(c),
                "split": "analytic",
            }
        )
    return rows


def main():
    T_vals, h_vals = _grid()
    print(f"Grid: {len(T_vals)} T values x {len(h_vals)} h values")

    train_rows = []
    holdout_rows = []
    for L in TRAIN_L:
        print(f"Generating train labels for L={L}")
        train_rows.extend(_rows_for_L(L, "train", T_vals, h_vals))
    for L in HOLDOUT_L:
        print(f"Generating holdout labels for L={L}")
        holdout_rows.extend(_rows_for_L(L, "holdout", T_vals, h_vals))

    print("Generating thermodynamic-limit analytic truth")
    truth_rows = _truth_rows(T_vals, h_vals)

    _write_csv(DATA / "ed_train_data.csv", train_rows)
    _write_csv(DATA / "ed_holdout_data.csv", holdout_rows)
    _write_csv(DATA / "analytic_truth_data.csv", truth_rows)
    np.savez(
        DATA / "grid_metadata.npz",
        T_values=np.array(T_vals),
        h_values=np.array(h_vals),
        train_L=np.array(TRAIN_L),
        holdout_L=np.array(HOLDOUT_L),
    )

    print(f"Saved {len(train_rows)} train rows")
    print(f"Saved {len(holdout_rows)} holdout rows")
    print(f"Saved {len(truth_rows)} analytic rows")


if __name__ == "__main__":
    main()
