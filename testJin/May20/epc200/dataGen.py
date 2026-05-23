import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import importlib.util
import os
import sys

# JAX의 64비트 정밀도 강제 활성화 (극저온에서의 데이터 품질 보장)
jax.config.update("jax_enable_x64", True)


def _load_exact_module():
    """
    May15/exact.py를 파일 경로로 직접 로드합니다.
    """
    exact_candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "May15", "exact.py")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "exact.py")),
    ]

    for exact_path in exact_candidates:
        if os.path.isfile(exact_path):
            spec = importlib.util.spec_from_file_location("tfim_exact", exact_path)
            module = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(module)
            return module

    raise FileNotFoundError("Could not locate exact.py in the expected workspace paths.")


_exact = _load_exact_module()
vectorized_finite_observables = _exact.vectorized_finite_observables
v_thermo_f = _exact.v_thermo_f
v_thermo_m = _exact.v_thermo_m
v_thermo_cv = _exact.v_thermo_cv
v_thermo_s = _exact.v_thermo_s
v_thermo_chi = _exact.v_thermo_chi

def create_parameter_grid(num_h=240, num_T=240):
    """
    QPT/FSS용 (h, T) 파라미터 그리드를 생성합니다.

    - h: [0.0, 2.0] 선형 그리드
    - T: [0.001, 2.0] 로그 그리드
    """
    h_vals = np.linspace(0.0, 2.0, num_h, dtype=np.float64)
    T_vals = np.geomspace(0.001, 0.5, num_T, dtype=np.float64)

    H_grid, T_grid = np.meshgrid(h_vals, T_vals, indexing="xy")
    return jnp.asarray(T_grid.ravel(), dtype=jnp.float64), jnp.asarray(H_grid.ravel(), dtype=jnp.float64)


def _as_float64(array_like):
    return np.asarray(array_like, dtype=np.float64)


def _evaluate_in_chunks(fn, T_flat, h_flat, chunk_size=4096):
    """
    전체 그리드를 한 번에 JAX에 넣지 않고, 메모리 사용량을 제한하기 위해
    작은 청크 단위로 계산한 뒤 numpy 배열로 결합합니다.
    """
    total = int(T_flat.shape[0])
    pieces = []

    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        chunk_values = fn(T_flat[start:stop], h_flat[start:stop])
        pieces.append(np.asarray(jax.device_get(chunk_values), dtype=np.float64))

    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)

def generate_data_for_L(L, T_flat, h_flat):
    """
    특정 L에 대한 물리량 정답을 계산하고, 저장 규격에 맞는 DataFrame으로 반환합니다.
    """
    print(f"Generating data for L = {L}...")

    if L == np.inf:
        inv_L = np.float64(0.0)
        F = _evaluate_in_chunks(v_thermo_f, T_flat, h_flat)
        M = _evaluate_in_chunks(v_thermo_m, T_flat, h_flat)
        Cv = _evaluate_in_chunks(v_thermo_cv, T_flat, h_flat)
        S = _evaluate_in_chunks(v_thermo_s, T_flat, h_flat)
        Chi = _evaluate_in_chunks(v_thermo_chi, T_flat, h_flat)
    else:
        inv_L = np.float64(1.0 / float(L))
        v_f, v_m, v_cv, v_s, v_chi = vectorized_finite_observables(L)

        F = _evaluate_in_chunks(v_f, T_flat, h_flat)
        M = _evaluate_in_chunks(v_m, T_flat, h_flat)
        Cv = _evaluate_in_chunks(v_cv, T_flat, h_flat)
        S = _evaluate_in_chunks(v_s, T_flat, h_flat)
        Chi = _evaluate_in_chunks(v_chi, T_flat, h_flat)

    return pd.DataFrame({
        'T': _as_float64(T_flat),
        'h': _as_float64(h_flat),
        'inv_L': np.full(_as_float64(T_flat).shape, inv_L, dtype=np.float64),
        'F': _as_float64(F),
        'M': _as_float64(M),
        'Cv': _as_float64(Cv),
        'S': _as_float64(S),
        'Chi': _as_float64(Chi),
    })


def validate_group_cardinality(df, expected_n_l, label):
    """
    저장 직전, 각 (h, T) 그룹이 정확히 expected_n_l개의 L 샘플을 갖는지 확인합니다.
    """
    group_sizes = df.groupby(['h', 'T'], sort=False).size()
    unique_sizes = group_sizes.unique()
    assert len(unique_sizes) == 1 and unique_sizes[0] == expected_n_l, (
        f"{label}: each (h, T) group must contain exactly {expected_n_l} rows, "
        f"but observed group sizes were {unique_sizes.tolist()}"
    )


def build_and_save_dataset(L_list, output_path, T_flat, h_flat, label):
    """
    여러 L에 대한 데이터를 결합하고, 정렬 및 정합성 검사를 수행한 뒤 CSV로 저장합니다.
    """
    dfs = [generate_data_for_L(L, T_flat, h_flat) for L in L_list]
    dataset = pd.concat(dfs, ignore_index=True)

    dataset = dataset[['T', 'h', 'inv_L', 'F', 'M', 'Cv', 'S', 'Chi']]
    dataset = dataset.sort_values(by=['h', 'T', 'inv_L'], kind='mergesort').reset_index(drop=True)

    validate_group_cardinality(dataset, len(L_list), label)
    dataset.to_csv(output_path, index=False, float_format='%.17g')
    print(f"{label} dataset saved: {output_path} (Shape: {dataset.shape})")

    return dataset

def main():
    """
    Train/Test 2분할 FSS-PINN 데이터셋을 생성하고 CSV로 저장합니다.
    """
    out_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.makedirs(out_root, exist_ok=True)

    T_flat, h_flat = create_parameter_grid(num_h=240, num_T=240)

    train_L_list = [8, 12, 16, 24, 32]
    test_L_list = [48, 64, 96, 128, np.inf]

    train_path = os.path.join(out_root, "tfim_train_L8_12_16_24_32_Tmax05.csv")
    test_path = os.path.join(out_root, "tfim_test_L48_64_96_128_inf_Tmax05.csv")

    print("--- Building Train Dataset (L=8,12,16,24,32) ---")
    build_and_save_dataset(train_L_list, train_path, T_flat, h_flat, label="Train")

    print("\n--- Building Test Dataset (L=48,64,96,128,inf) ---")
    build_and_save_dataset(test_L_list, test_path, T_flat, h_flat, label="Test")
    
    print("\nDataset generation complete!")

if __name__ == "__main__":
    main()
