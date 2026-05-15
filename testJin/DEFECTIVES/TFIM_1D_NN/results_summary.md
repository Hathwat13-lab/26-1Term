# 1D NN TFIM FSS 신경망 결과 요약

설정: `{'variant': 'thermo_consistency_fixed', 'story': 'Global Thermodynamic Consistency: Added explicit Sobolev supervision for Magnetization (M) and Entropy (S) to fix curvature errors in temperature derivatives, while maintaining chi-Hessian performance.', 'train_L': [6, 8, 10, 12], 'holdout_L': [14, 16], 'epochs': 3600, 'chi_weight': 0.04, 'cv_weight': 0.01, 'entropy_weight': 0.01, 'm_weight': 0.01, 'guardrail_weight': 0.02, 'critical_sampling': {'batch_fraction': 0.2, 'derivative_fraction': 0.55}, 'elapsed_seconds': 97.07023739814758}`

## 주요 지표

- `train_F_mse`: 6.61561160e-07
- `train_F_mae`: 6.17925194e-04
- `train_F_max_abs`: 3.70502472e-03
- `holdout_F_mse`: 2.13232238e-06
- `holdout_F_mae`: 1.28369324e-03
- `holdout_F_max_abs`: 3.07810307e-03
- `holdout_chi_mse`: 4.77319205e-04
- `holdout_chi_mae`: 1.68974288e-02
- `holdout_chi_max_abs`: 1.08099937e-01
- `thermo_F_mse`: 5.98826955e-05
- `thermo_F_mae`: 6.98270928e-03
- `thermo_F_max_abs`: 1.20536089e-02
- `thermo_chi_mse`: 3.63003253e-03
- `thermo_chi_mae`: 4.53905277e-02
- `thermo_chi_max_abs`: 1.08806396e+00
- `holdout_critical_chi_mse`: 6.95510244e-04
- `holdout_critical_chi_mae`: 2.10971739e-02
- `holdout_critical_chi_max_abs`: 1.08099937e-01
- `thermo_critical_chi_mse`: 5.87095506e-03
- `thermo_critical_chi_mae`: 5.58711737e-02
- `thermo_critical_chi_max_abs`: 1.08806396e+00

## 자기감수율 피크 위치

| T | NN h_peak | 해석해 h_peak | NN chi_max | 해석해 chi_max |
|---:|---:|---:|---:|---:|
| 0.050 | 0.99599 | 0.99599 | 1.25784 | 1.34301 |
| 0.100 | 0.98963 | 0.98328 | 1.15439 | 1.12734 |
| 0.200 | 0.97692 | 0.95151 | 0.97544 | 0.92028 |
| 0.500 | 0.45585 | 0.78629 | 0.65462 | 0.67707 |
| 1.000 | 0.57023 | 0.10000 | 0.57158 | 0.59014 |
