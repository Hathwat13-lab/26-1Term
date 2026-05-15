import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os

# JAX 64비트 활성화
jax.config.update("jax_enable_x64", True)

# training 폴더 내의 model.py를 참조하기 위해 경로 추가
sys.path.append(os.path.join(os.getcwd(), "training"))
from model import FSSNet, get_physics_quantities

# 정답 비교를 위한 exact 모듈 import
from exact import (
    vectorized_finite_observables,
    v_thermo_f, v_thermo_m, v_thermo_cv, v_thermo_s, v_thermo_chi
)

def run_test_case(T_fixed, L_val, params, model):
    """특정 T와 L에 대해 정답과 예측값을 비교하여 시각화하고 저장합니다."""
    h_vals = jnp.linspace(0.5, 1.5, 300)
    T_jax = jnp.full_like(h_vals, T_fixed)
    
    # 1. 정답(Exact Solution) 계산
    if L_val == np.inf:
        inv_L_val = 0.0
        F_true = v_thermo_f(T_jax, h_vals)
        M_true = v_thermo_m(T_jax, h_vals)
        Chi_true = v_thermo_chi(T_jax, h_vals)
        Cv_true = v_thermo_cv(T_jax, h_vals)
        S_true = v_thermo_s(T_jax, h_vals)
        label_str = "L=inf"
    else:
        inv_L_val = 1.0 / float(L_val)
        v_f, v_m, v_cv, v_s, v_chi = vectorized_finite_observables(int(L_val))
        F_true = v_f(T_jax, h_vals)
        M_true = v_m(T_jax, h_vals)
        Chi_true = v_chi(T_jax, h_vals)
        Cv_true = v_cv(T_jax, h_vals)
        S_true = v_s(T_jax, h_vals)
        label_str = f"L={L_val}"

    # 2. 신경망(NN) 예측 계산
    predict_fn = jax.vmap(lambda t, h, l: get_physics_quantities(model.apply, params, t, h, l))
    preds = predict_fn(T_jax, h_vals, jnp.full_like(h_vals, inv_L_val))
    
    F_pred, M_pred, Cv_pred, S_pred, Chi_pred = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3], preds[:, 4]

    # 3. 시각화
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    titles = ['Free Energy (F)', 'Magnetization (M)', 'Susceptibility (Chi)', 'Specific Heat (Cv)', 'Entropy (S)']
    true_data = [F_true, M_true, Chi_true, Cv_true, S_true]
    pred_data = [F_pred, M_pred, Chi_pred, Cv_pred, S_pred]

    for i in range(5):
        ax = axes[i]
        ax.plot(h_vals, true_data[i], 'k-', label=f'Exact ({label_str})', linewidth=2)
        ax.plot(h_vals, pred_data[i], 'r--', label='NN Predict (Extrapolated)', linewidth=2)
        ax.set_title(titles[i], fontsize=14)
        ax.set_xlabel('Transverse Field (h)')
        ax.grid(True, alpha=0.3)
        ax.axvline(1.0, color='gray', linestyle=':', alpha=0.7)
        if i == 0: ax.legend()

    axes[5].axis('off')
    plt.tight_layout()
    plt.suptitle(f'NN vs Exact: Extrapolation to {label_str} (at T={T_fixed})', y=1.02, fontsize=16)
    
    filename = f'NN_Test_{label_str}_T{str(T_fixed).replace(".", "")}.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def main():
    # 학습된 파라미터 로드
    params_path = "training/fss_model.params"
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    print("Model parameters loaded.")

    model = FSSNet()
    
    # 테스트 케이스 설정
    temps = [0.1, 0.001]
    Ls = [14, 16, np.inf]

    for T in temps:
        for L in Ls:
            run_test_case(T, L, params, model)

if __name__ == "__main__":
    main()
