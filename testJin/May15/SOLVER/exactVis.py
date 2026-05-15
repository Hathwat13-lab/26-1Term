import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
# 반드시 다른 JAX 연산이 시작되기 전에 선언해야 합니다.
jax.config.update("jax_enable_x64", True)

# 앞서 만든 exact.py 모듈에서 함수들을 불러옵니다.
from exact import (
    vectorized_finite_observables,
    v_thermo_f, v_thermo_m, v_thermo_cv, v_thermo_s, v_thermo_chi
)

def main():
    # 1. 관찰할 시스템 크기 L의 리스트
    L_list = [4, 6, 8, 12, 14, 16]
    
    # 2. 하이퍼파라미터 그리드 설정
    # 양자 상전이를 보기 위해 초저온(T=0.001)으로 고정하고, h를 0.5 ~ 1.5까지 스윕
    num_points = 300
    h_vals = jnp.linspace(0.5, 1.5, num_points)
    T_fixed = 0.001
    T_vals = jnp.full_like(h_vals, T_fixed) # h_vals와 같은 크기의 T 배열 생성

    # 3. 플롯 준비 (2행 3열의 서브플롯)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    obs_names = ['Free Energy (F)', 'Magnetization (M)', 'Susceptibility (Chi)', 'Specific Heat (Cv)', 'Entropy (S)']
    
    # 색상 맵 설정 (L이 커질수록 색이 진해지도록)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_list) + 1))

    # 4. 유한 크기(Finite-L) 데이터 계산 및 플로팅
    for i, L in enumerate(L_list):
        print(f"Calculating for L = {L}...")
        # L에 대한 vmap된 함수들을 가져옵니다.
        v_f, v_m, v_cv, v_s, v_chi = vectorized_finite_observables(L)
        
        # 계산 수행 (JAX 배열 반환)
        F = v_f(T_vals, h_vals)
        M = v_m(T_vals, h_vals)
        Cv = v_cv(T_vals, h_vals)
        S = v_s(T_vals, h_vals)
        Chi = v_chi(T_vals, h_vals)
        
        observables = [F, M, Chi, Cv, S]
        
        # 각 물리량을 서브플롯에 그리기
        for j, (ax, obs_data) in enumerate(zip(axes[:5], observables)):
            ax.plot(h_vals, obs_data, color=colors[i], label=f'L={L}')

    # 5. 무한 사슬 (L -> inf) 데이터 계산 및 플로팅
    print("Calculating for L = inf...")
    F_inf = v_thermo_f(T_vals, h_vals)
    M_inf = v_thermo_m(T_vals, h_vals)
    Cv_inf = v_thermo_cv(T_vals, h_vals)
    S_inf = v_thermo_s(T_vals, h_vals)
    Chi_inf = v_thermo_chi(T_vals, h_vals)
    
    observables_inf = [F_inf, M_inf, Chi_inf, Cv_inf, S_inf]
    
    for j, (ax, obs_data) in enumerate(zip(axes[:5], observables_inf)):
        # 무한대 데이터는 빨간색 점선으로 표시하여 명확히 구분
        ax.plot(h_vals, obs_data, color='red', linestyle='--', linewidth=2, label='L=inf')
        
        # 그래프 꾸미기
        ax.set_title(obs_names[j])
        ax.set_xlabel('Transverse Field (h)')
        ax.grid(True, alpha=0.3)
        # 임계점 h=1.0 에 수직선 긋기
        ax.axvline(1.0, color='gray', linestyle=':', alpha=0.7)
        if j == 0: # 첫 번째 그래프에만 범례 표시
            ax.legend()

    # 빈 서브플롯(6번째) 숨기기
    axes[5].axis('off')

    plt.tight_layout()
    plt.suptitle(f'1D TFIM Exact Observables at T={T_fixed}', y=1.02, fontsize=16)
    plt.savefig('exact_observables_T0001_x64.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()