# 1D NN TFIM FSS 신경망 결과 요약

설정: `{'variant': 'hessian_feature_tuned_fixed', 'story': 'Feature Engineering + Convexity Penalty: Fixed negative chi bug by using correct scaling r2*log(r2) and adding a strict convexity loss to prevent curvature inversion.', 'train_L': [6, 8, 10, 12], 'holdout_L': [14, 16], 'epochs': 3200, 'chi_weight': 0.04, 'cv_weight': 0.004, 'guardrail_weight': 0.02, 'critical_sampling': {'batch_fraction': 0.2, 'derivative_fraction': 0.55}, 'elapsed_seconds': 69.98059487342834}`

## 주요 지표

- `train_F_mse`: 3.14826957e-06
- `train_F_mae`: 1.40973052e-03
- `train_F_max_abs`: 8.79758596e-03
- `holdout_F_mse`: 4.31401986e-06
- `holdout_F_mae`: 1.35167292e-03
- `holdout_F_max_abs`: 1.08399987e-02
- `holdout_chi_mse`: 5.84931811e-04
- `holdout_chi_mae`: 1.72993876e-02
- `holdout_chi_max_abs`: 1.43962979e-01
- `thermo_F_mse`: 5.42281814e-05
- `thermo_F_mae`: 6.67147478e-03
- `thermo_F_max_abs`: 1.93608403e-02
- `thermo_chi_mse`: 3.80017282e-03
- `thermo_chi_mae`: 4.61087301e-02
- `thermo_chi_max_abs`: 1.15494287e+00
- `holdout_critical_chi_mse`: 9.20390652e-04
- `holdout_critical_chi_mae`: 2.16609351e-02
- `holdout_critical_chi_max_abs`: 1.43962979e-01
- `thermo_critical_chi_mse`: 6.57911785e-03
- `thermo_critical_chi_mae`: 6.01779222e-02
- `thermo_critical_chi_max_abs`: 1.15494287e+00

## 자기감수율 피크 위치

| T | NN h_peak | 해석해 h_peak | NN chi_max | 해석해 chi_max |
|---:|---:|---:|---:|---:|
| 0.050 | 0.99599 | 0.99599 | 1.19366 | 1.34301 |
| 0.100 | 0.98963 | 0.98328 | 1.10945 | 1.12734 |
| 0.200 | 0.97692 | 0.95151 | 0.96826 | 0.92028 |
| 0.500 | 0.44950 | 0.78629 | 0.63324 | 0.67707 |
| 1.000 | 0.10000 | 0.10000 | 0.64997 | 0.59014 |
