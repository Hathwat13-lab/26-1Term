import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import os
import sys

# 상위 폴더의 exact.py를 불러오기 위해 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 또한 기존 May15 폴더의 exact.py를 참조할 수 있도록 폴더를 추가로 등록합니다 (존재 시)
may15_sibling = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "May15"))
if os.path.isdir(may15_sibling):
    sys.path.append(may15_sibling)

# JAX의 64비트 정밀도 강제 활성화
jax.config.update("jax_enable_x64", True)

from exact import (
    vectorized_finite_observables,
    v_thermo_f, v_thermo_m, v_thermo_cv, v_thermo_s, v_thermo_chi
)

def create_parameter_grid(num_h=100, num_T=100):
    h_vals = np.linspace(0.5, 1.5, num_h)
    T_vals = np.geomspace(0.0001, 2.0, num_T)
    H_grid, T_grid = np.meshgrid(h_vals, T_vals)
    return jnp.array(T_grid.flatten()), jnp.array(H_grid.flatten())

def generate_data_for_L(L, T_flat, h_flat):
    print(f"Generating data for L = {L}...")
    if L == np.inf:
        inv_L = 0.0
        F = v_thermo_f(T_flat, h_flat)
        M = v_thermo_m(T_flat, h_flat)
        Cv = v_thermo_cv(T_flat, h_flat)
        S = v_thermo_s(T_flat, h_flat)
        Chi = v_thermo_chi(T_flat, h_flat)
    else:
        inv_L = 1.0 / float(L)
        v_f, v_m, v_cv, v_s, v_chi = vectorized_finite_observables(L)
        F = v_f(T_flat, h_flat)
        M = v_m(T_flat, h_flat)
        Cv = v_cv(T_flat, h_flat)
        S = v_s(T_flat, h_flat)
        Chi = v_chi(T_flat, h_flat)
    return pd.DataFrame({
        'T': np.array(T_flat),
        'h': np.array(h_flat),
        'inv_L': np.full_like(np.array(T_flat), inv_L),
        'F': np.array(F),
        'M': np.array(M),
        'Cv': np.array(Cv),
        'S': np.array(S),
        'Chi': np.array(Chi)
    })

def main():
    out_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.makedirs(out_root, exist_ok=True)
    T_flat, h_flat = create_parameter_grid(num_h=100, num_T=100)

    # Sparse-span training Ls and test Ls per request
    train_L_list = [8, 16, 24]
    test_L_list = [32, 48, np.inf]

    print("--- Building Sparse Train Dataset (L=8,16,24) ---")
    train_dfs = []
    for L in train_L_list:
        df = generate_data_for_L(L, T_flat, h_flat)
        train_dfs.append(df)
    train_dataset = pd.concat(train_dfs, ignore_index=True)
    train_path = os.path.join(out_root, "tfim_train_L8_24_sparse.csv")
    train_dataset.to_csv(train_path, index=False)
    print(f"Train dataset saved: {train_path} (Shape: {train_dataset.shape})")

    print("\n--- Building Test (Extrapolation) Dataset v2 (L=32,48,inf) ---")
    test_dfs = []
    for L in test_L_list:
        df = generate_data_for_L(L, T_flat, h_flat)
        test_dfs.append(df)
    test_dataset = pd.concat(test_dfs, ignore_index=True)
    test_path = os.path.join(out_root, "tfim_test_extrapolation_v2.csv")
    test_dataset.to_csv(test_path, index=False)
    print(f"Test dataset saved: {test_path} (Shape: {test_dataset.shape})")

if __name__ == "__main__":
    main()
