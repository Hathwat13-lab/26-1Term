import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import sys
import os

# JAX 64비트 활성화
jax.config.update("jax_enable_x64", True)

# training 폴더 경로 추가
sys.path.append(os.path.join(os.getcwd(), "training"))
from model import FSSNet, get_physics_quantities
from exact import vectorized_finite_observables, v_thermo_f, v_thermo_m, v_thermo_cv, v_thermo_s, v_thermo_chi

def plot_learning_curve():
    """학습 곡선을 그립니다."""
    history_path = "training/loss_history.csv"
    if not os.path.exists(history_path):
        print("Waiting for training to generate loss_history.csv...")
        return False
    
    df = pd.read_csv(history_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
    plt.plot(df['epoch'], df['test_loss'], label='Test (Extrapolation) Loss', color='red', linestyle='--')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('1D TFIM Learning Curve (Sobolev Loss)', fontsize=15)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.savefig('Final_Learning_Curve.png', bbox_inches='tight')
    plt.close()
    print("Learning curve saved as Final_Learning_Curve.png")
    return True

def run_vis_case(T_fixed, L_val, params, model):
    """예측 비교 그래프 생성"""
    h_vals = jnp.linspace(0.4, 1.6, 400) # 범위를 조금 더 넓히고 해상도 높임
    T_jax = jnp.full_like(h_vals, T_fixed)
    
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

    predict_fn = jax.vmap(lambda t, h, l: get_physics_quantities(model.apply, params, t, h, l))
    preds = predict_fn(T_jax, h_vals, jnp.full_like(h_vals, inv_L_val))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    titles = ['Free Energy (F)', 'Magnetization (M)', 'Susceptibility (Chi)', 'Specific Heat (Cv)', 'Entropy (S)']
    true_data = [F_true, M_true, Chi_true, Cv_true, S_true]
    pred_data = [preds[:, 0], preds[:, 1], preds[:, 4], preds[:, 2], preds[:, 3]] # 순서: F, M, Chi, Cv, S

    for i in range(5):
        ax = axes[i]
        ax.plot(h_vals, true_data[i], 'k-', label=f'Exact ({label_str})', linewidth=2.5)
        ax.plot(h_vals, pred_data[i], 'r--', label='NN Predict', linewidth=2)
        ax.set_title(titles[i], fontsize=14, fontweight='bold')
        ax.set_xlabel('Transverse Field (h)')
        ax.grid(True, alpha=0.3)
        ax.axvline(1.0, color='gray', linestyle=':', alpha=0.7)
        if i == 0: ax.legend()

    axes[5].axis('off')
    plt.tight_layout()
    plt.suptitle(f'Extrapolation Performance: {label_str} at T={T_fixed}', y=1.02, fontsize=18)
    
    filename = f'Final_Test_{label_str}_T{str(T_fixed).replace(".", "")}.png'
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Prediction plot saved: {filename}")

def main():
    # 1. 학습 곡선 그리기
    if not plot_learning_curve():
        return

    # 2. 모델 로드 및 예측 시각화
    params_path = "training/fss_model.params"
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    
    model = FSSNet()
    temps = [0.1, 0.001]
    Ls = [14, 16, np.inf]

    print("Generating prediction comparison plots...")
    for T in temps:
        for L in Ls:
            run_vis_case(T, L, params, model)
    
    print("\n--- Final Report Generation Complete ---")

if __name__ == "__main__":
    main()
