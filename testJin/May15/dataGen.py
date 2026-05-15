import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import os

# JAX의 64비트 정밀도 강제 활성화 (극저온에서의 데이터 품질 보장)
jax.config.update("jax_enable_x64", True)

# 완성된 exact.py 엔진에서 함수들을 불러옵니다.
from exact import (
    vectorized_finite_observables,
    v_thermo_f, v_thermo_m, v_thermo_cv, v_thermo_s, v_thermo_chi
)

def create_parameter_grid(num_h=300, num_T=300):
    """
    [블록 1: 하이퍼파라미터 그리드 생성]
    h와 T의 탐색 범위를 설정합니다.
    """
    # h (자기장): 0.5 ~ 1.5 구간을 선형(Linear)으로 쪼갭니다.
    # 양자 임계점(h=1) 부근의 상전이를 잘 포착할 수 있습니다.
    h_vals = np.linspace(0.5, 1.5, num_h)
    
    # T (온도): 0.0001 ~ 2.0 구간을 로그 스케일(Log-space)로 쪼갭니다.
    # 극저온(0.0001~0.1) 영역에 데이터 포인트를 집중시켜 Schottky bump를 신경망이 잘 보게 합니다.
    T_vals = np.geomspace(0.0001, 2.0, num_T)
    
    # h와 T의 모든 가능한 조합(Meshgrid)을 만듭니다.
    H_grid, T_grid = np.meshgrid(h_vals, T_vals)
    
    # 1차원 배열로 펼쳐서 JAX 함수에 넣기 좋게 만듭니다.
    return jnp.array(T_grid.flatten()), jnp.array(H_grid.flatten())

def generate_data_for_L(L, T_flat, h_flat):
    """
    [블록 2: 특정 크기 L에 대한 데이터 생성]
    주어진 L, T, h 배열에 대해 물리량 정답(Ground Truth)을 계산합니다.
    """
    print(f"Generating data for L = {L}...")
    
    # L이 무한대(inf)인 경우와 유한한 경우를 분기 처리
    if L == np.inf:
        inv_L = 0.0 # 열역학적 극한에서는 1/L = 0
        F = v_thermo_f(T_flat, h_flat)
        M = v_thermo_m(T_flat, h_flat)
        Cv = v_thermo_cv(T_flat, h_flat)
        S = v_thermo_s(T_flat, h_flat)
        Chi = v_thermo_chi(T_flat, h_flat)
    else:
        inv_L = 1.0 / float(L) # 신경망 입력용 1/L 특징(Feature)
        v_f, v_m, v_cv, v_s, v_chi = vectorized_finite_observables(L)
        
        F = v_f(T_flat, h_flat)
        M = v_m(T_flat, h_flat)
        Cv = v_cv(T_flat, h_flat)
        S = v_s(T_flat, h_flat)
        Chi = v_chi(T_flat, h_flat)
        
    # Pandas DataFrame으로 묶기 위해 딕셔너리로 반환
    # 당장은 F만 학습하더라도, 추후를 위해 모든 파생 물리량을 함께 저장해둡니다.
    return pd.DataFrame({
        'T': np.array(T_flat),
        'h': np.array(h_flat),
        'inv_L': np.full_like(np.array(T_flat), inv_L), # 모든 행에 1/L 값 복사
        'F': np.array(F),
        'M': np.array(M),
        'Cv': np.array(Cv),
        'S': np.array(S),
        'Chi': np.array(Chi)
    })

def main():
    """
    [블록 3: 메인 실행 및 Train/Test 분할 저장]
    데이터셋을 생성하고 CSV 파일로 저장합니다.
    """
    # 저장할 폴더 생성
    os.makedirs("dataset", exist_ok=True)
    
    # 1. 그리드 생성 (총 300 x 300 = 90,000개의 (T, h) 포인트)
    T_flat, h_flat = create_parameter_grid(num_h=300, num_T=300)
    
    # 2. 시스템 크기 정의
    train_L_list = [4, 6, 8, 10, 12]
    test_L_list = [14, 16, np.inf]
    
    # 3. Train Data 생성 및 결합
    print("--- Building Train Dataset ---")
    train_dfs = []
    for L in train_L_list:
        df = generate_data_for_L(L, T_flat, h_flat)
        train_dfs.append(df)
    
    train_dataset = pd.concat(train_dfs, ignore_index=True)
    train_path = os.path.join("dataset", "tfim_train.csv")
    train_dataset.to_csv(train_path, index=False)
    print(f"Train dataset saved: {train_path} (Shape: {train_dataset.shape})")
    
    # 4. Test Data 생성 및 결합
    print("\n--- Building Test (Extrapolation) Dataset ---")
    test_dfs = []
    for L in test_L_list:
        df = generate_data_for_L(L, T_flat, h_flat)
        test_dfs.append(df)
        
    test_dataset = pd.concat(test_dfs, ignore_index=True)
    test_path = os.path.join("dataset", "tfim_test.csv")
    test_dataset.to_csv(test_path, index=False)
    print(f"Test dataset saved: {test_path} (Shape: {test_dataset.shape})")
    
    print("\nDataset generation complete!")

if __name__ == "__main__":
    main()