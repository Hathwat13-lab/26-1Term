# 1D NN TFIM FSS 신경망 결과 요약

설정: `{'variant': 'balanced_peak_guard', 'story': 'Balanced tuning: keep Sobolev chi supervision, but add an inv_L=0 overshoot guardrail.', 'train_L': [6, 8, 10, 12], 'holdout_L': [14, 16], 'epochs': 2600, 'chi_weight': 0.04, 'cv_weight': 0.004, 'guardrail_weight': 0.02, 'critical_sampling': {'batch_fraction': 0.2, 'derivative_fraction': 0.55}, 'elapsed_seconds': 56.95398473739624}`

## 주요 지표

- `train_F_mse`: 6.32338867e-07
- `train_F_mae`: 5.94544574e-04
- `train_F_max_abs`: 4.65476513e-03
- `holdout_F_mse`: 1.88924855e-06
- `holdout_F_mae`: 1.01365556e-03
- `holdout_F_max_abs`: 6.73824549e-03
- `holdout_chi_mse`: 4.48615669e-04
- `holdout_chi_mae`: 1.65741853e-02
- `holdout_chi_max_abs`: 7.94345140e-02
- `thermo_F_mse`: 2.55010291e-05
- `thermo_F_mae`: 4.52262675e-03
- `thermo_F_max_abs`: 1.68820620e-02
- `thermo_chi_mse`: 3.45181162e-03
- `thermo_chi_mae`: 3.97851616e-02
- `thermo_chi_max_abs`: 2.44114161e-01
- `holdout_critical_chi_mse`: 5.89121773e-04
- `holdout_critical_chi_mae`: 1.96352229e-02
- `holdout_critical_chi_max_abs`: 7.74660707e-02
- `thermo_critical_chi_mse`: 9.23154503e-03
- `thermo_critical_chi_mae`: 7.53218681e-02
- `thermo_critical_chi_max_abs`: 2.44114161e-01

## 자기감수율 피크 위치

| T | NN h_peak | 해석해 h_peak | NN chi_max | 해석해 chi_max |
|---:|---:|---:|---:|---:|
| 0.050 | 1.00234 | 0.99599 | 1.22772 | 1.34301 |
| 0.100 | 0.99599 | 0.98328 | 1.14208 | 1.12734 |
| 0.200 | 0.98963 | 0.95151 | 0.97701 | 0.92028 |
| 0.500 | 0.92609 | 0.78629 | 0.68974 | 0.67707 |
| 1.000 | 0.10000 | 0.10000 | 0.61903 | 0.59014 |
