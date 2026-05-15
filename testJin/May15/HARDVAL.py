import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import time

# 64-bit precision 강제 활성화 (Numerical Stability)
jax.config.update("jax_enable_x64", True)

# 완성된 solver 모듈 import
from exact import vectorized_finite_observables, dispersion

class TFIMValidator:
    def __init__(self):
        self.report = {}
        
    # ========================================================
    # [Task 4 & 3] Exact Diagonalization & T->0 Consistency
    # ========================================================
    def build_pbc_spin_hamiltonian(self, L, h):
        """
        Spin-basis Exact Diagonalization용 해밀토니안 (PBC)
        H = - \sum (X_i X_{i+1}) - h \sum Z_i
        """
        sig_x = np.array([[0, 1], [1, 0]])
        sig_z = np.array([[1, 0], [0, -1]])
        eye = np.eye(2)
        
        H = np.zeros((2**L, 2**L))
        
        # XX term (PBC)
        for i in range(L):
            j = (i + 1) % L
            term = 1
            for k in range(L):
                if k == i or k == j:
                    term = np.kron(term, sig_x) if not isinstance(term, int) else sig_x
                else:
                    term = np.kron(term, eye) if not isinstance(term, int) else eye
            H -= term
            
        # Z term (Transverse field)
        for i in range(L):
            term = 1
            for k in range(L):
                if k == i:
                    term = np.kron(term, sig_z) if not isinstance(term, int) else sig_z
                else:
                    term = np.kron(term, eye) if not isinstance(term, int) else eye
            H -= h * term
            
        return H

    def run_ed_crosscheck(self, L_test=[4, 6], h=0.8):
        print("\n--- [Task 3 & 4] ED Cross-check & GS Consistency ---")
        T_vals = np.linspace(0.05, 2.0, 50)
        T_zero = np.array([1e-4]) # T -> 0 limit
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for L in L_test:
            # 1. ED 계산
            H = self.build_pbc_spin_hamiltonian(L, h)
            evals = eigh(H, eigvals_only=True)
            E0 = evals[0] / L # Ground state energy density
            
            F_ed = []
            for T in T_vals:
                beta = 1.0 / T
                # LogSumExp trick for ED
                Z_shifted = np.sum(np.exp(-beta * (evals - evals[0])))
                F_ed.append((evals[0] - T * np.log(Z_shifted)) / L)
                
            # 2. JW Solver 계산
            v_f, _, _, _, _ = vectorized_finite_observables(L)
            F_jw = np.array(v_f(jnp.array(T_vals), jnp.full_like(T_vals, h)))
            F_zero_jw = np.array(v_f(T_zero, jnp.array([h])))[0]
            
            # Validation Assertions
            max_err = np.max(np.abs(F_ed - F_jw))
            gs_err = np.abs(F_zero_jw - E0)
            
            print(f"[L={L}] Max F error (ED vs JW): {max_err:.2e}")
            print(f"[L={L}] T->0 GS Energy error:   {gs_err:.2e}")
            
            self.report[f'ED_Match_L{L}'] = max_err < 1e-10
            self.report[f'GS_Consistency_L{L}'] = gs_err < 1e-10
            
            axes[0].plot(T_vals, F_ed, 'o', label=f'ED L={L}', alpha=0.5)
            axes[0].plot(T_vals, F_jw, '-', label=f'JW L={L}')
            axes[1].plot(T_vals, F_ed - F_jw, label=f'Error L={L}')
            
        axes[0].set_title(f"Free Energy: ED vs JW (h={h})")
        axes[0].legend()
        axes[1].set_title("Absolute Error")
        plt.tight_layout()
        plt.savefig('Valid_Hard_ED_Comparison.png', bbox_inches='tight')
        plt.close()

    # ========================================================
    # [Task 1 & 2] Finite-Size Scaling & Pseudocritical Drift
    # ========================================================
    def run_scaling_tests(self):
        print("\n--- [Task 1 & 2] FSS & Pseudocritical Drift ---")
        L_list = np.array([6, 8, 10, 12, 14, 16, 20])
        h_vals = np.linspace(0.8, 1.2, 2000)
        T_fixed = 0.01 # Deep critical regime
        
        chi_max = []
        h_star = []
        
        for L in L_list:
            _, _, _, _, v_chi = vectorized_finite_observables(int(L))
            chi = np.array(v_chi(jnp.full_like(h_vals, T_fixed), jnp.array(h_vals)))
            idx = np.argmax(chi)
            chi_max.append(chi[idx])
            h_star.append(h_vals[idx])
            
        chi_max = np.array(chi_max)
        h_star = np.array(h_star)
        
        # Linear fits
        log_L = np.log(L_list)
        inv_L = 1.0 / L_list
        
        # chi_max ~ a*log(L) + b 확인 (1D TFIM specific)
        fit_chi = np.polyfit(log_L, chi_max, 1)
        # h* - 1 ~ a*(1/L) + b 확인 (nu = 1)
        fit_h = np.polyfit(inv_L, np.abs(h_star - 1.0), 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(log_L, chi_max, 'bo-', label='Data')
        axes[0].plot(log_L, np.polyval(fit_chi, log_L), 'r--', label=f'Fit: slope={fit_chi[0]:.2f}')
        axes[0].set_title(r"Susceptibility Peak Scaling: $\chi_{max} \sim \log L$")
        axes[0].set_xlabel(r"$\log L$")
        axes[0].legend()
        
        axes[1].plot(inv_L, np.abs(h_star - 1.0), 'go-', label='Data')
        axes[1].plot(inv_L, np.polyval(fit_h, inv_L), 'r--', label=f'Fit: slope={fit_h[0]:.2f}')
        axes[1].set_title(r"Pseudocritical Drift: $|h^* - 1| \sim L^{-1/\nu} \ (\nu=1)$")
        axes[1].set_xlabel(r"$1/L$")
        axes[1].legend()
        plt.tight_layout()
        plt.savefig('Valid_Hard_Scaling_Analysis.png', bbox_inches='tight')
        plt.close()

    # ========================================================
    # [Task 5] Sector-Resolved Analysis
    # ========================================================
    def extract_sectors(self, T, h, L):
        """솔버 로직을 분해하여 각 Z 파티션의 Weight를 추출"""
        n = jnp.arange(L)
        k_A = (2.0 * n + 1.0) * jnp.pi / L
        k_P = 2.0 * n * jnp.pi / L

        x_A = dispersion(k_A, h) / (2.0 * T)
        x_P_rest = dispersion(k_P[1:], h) / (2.0 * T)
        x_0 = (1.0 - h) / T

        ln_Z1 = jnp.sum(x_A + jnp.log1p(jnp.exp(-2.0 * x_A)))
        ln_Z2 = jnp.sum(x_A + jnp.log1p(-jnp.exp(-2.0 * x_A)))
        ln_Z3_rest = jnp.sum(x_P_rest + jnp.log1p(jnp.exp(-2.0 * x_P_rest)))
        ln_Z4_rest = jnp.sum(x_P_rest + jnp.log1p(-jnp.exp(-2.0 * x_P_rest)))

        max_ln_Z = jnp.maximum(ln_Z1, ln_Z3_rest + jnp.abs(x_0))
        term1 = jnp.exp(ln_Z1 - max_ln_Z)
        term2 = jnp.exp(ln_Z2 - max_ln_Z)
        term3 = jnp.exp(ln_Z3_rest + x_0 - max_ln_Z) + jnp.exp(ln_Z3_rest - x_0 - max_ln_Z)
        term4 = jnp.exp(ln_Z4_rest + x_0 - max_ln_Z) - jnp.exp(ln_Z4_rest - x_0 - max_ln_Z)

        Z_tot = term1 + term2 + term3 + term4
        # Relative weights
        W1 = term1 / Z_tot
        W2 = term2 / Z_tot
        W3 = term3 / Z_tot
        W4 = term4 / Z_tot 
        
        return W1, W2, W3, W4

    def run_sector_analysis(self, L=10):
        print("\n--- [Task 5] Sector-Resolved Contribution ---")
        T_vals = np.linspace(0.01, 1.5, 300)
        h_ferro = 0.8
        
        W1, W2, W3, W4 = jax.vmap(self.extract_sectors, in_axes=(0, None, None))(T_vals, h_ferro, L)
        
        plt.figure(figsize=(8, 5))
        plt.plot(T_vals, W1, label='Z1 (APBC Cosh)')
        plt.plot(T_vals, W2, label='Z2 (APBC Sinh)')
        plt.plot(T_vals, W3, label='Z3 (PBC Cosh)')
        plt.plot(T_vals, W4, label='-Z4 (PBC Sinh)')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.title(f"Parity Sector Weights at h={h_ferro} (Ferro, L={L})")
        plt.xlabel("Temperature T")
        plt.ylabel("Relative Weight ($W_i / Z_{tot}$)")
        plt.legend()
        plt.savefig('Valid_Hard_Sector_Weights.png', bbox_inches='tight')
        plt.close()

    # ========================================================
    # [Task 6 & 7] Stability & Thermodynamic Consistency
    # ========================================================
    def run_consistency_stability(self, L=12):
        print("\n--- [Task 6 & 7] AutoDiff Stability & Thermodynamics ---")
        h_crit = 1.0000001 # Critical point edge
        T_vals = np.logspace(-4, 0, 100) # 1e-4 to 1.0
        dT = 1e-6
        dh = 1e-6
        
        v_f, v_m, v_cv, v_s, v_chi = vectorized_finite_observables(L)
        
        # 1. Stability Check (NaN 검열)
        cv = np.array(v_cv(jnp.array(T_vals), jnp.full_like(T_vals, h_crit)))
        chi = np.array(v_chi(jnp.array(T_vals), jnp.full_like(T_vals, h_crit)))
        
        nan_safe = not (np.isnan(cv).any() or np.isnan(chi).any())
        self.report['Stability_Critical'] = nan_safe
        print(f"Numerical Stability at h~1, T<<1: {'PASS' if nan_safe else 'FAIL'}")

        # 2. Consistency Check (Numerical vs AutoDiff)
        T_test = np.array([0.1, 0.5, 1.0])
        h_test = np.array([0.8, 1.0, 1.2])
        
        T_grid, h_grid = np.meshgrid(T_test, h_test)
        T_flat, h_flat = T_grid.flatten(), h_grid.flatten()
        
        F = v_f(T_flat, h_flat)
        F_Tplus = v_f(T_flat + dT, h_flat)
        F_Tminus = v_f(T_flat - dT, h_flat)
        F_hplus = v_f(T_flat, h_flat + dh)
        F_hminus = v_f(T_flat, h_flat - dh)
        
        S_num = -(F_Tplus - F_Tminus) / (2 * dT)
        M_num = -(F_hplus - F_hminus) / (2 * dh)
        
        S_ad = v_s(T_flat, h_flat)
        M_ad = v_m(T_flat, h_flat)
        
        err_S = np.max(np.abs(S_num - S_ad))
        err_M = np.max(np.abs(M_num - M_ad))
        
        self.report['Consistency_S'] = err_S < 1e-5
        self.report['Consistency_M'] = err_M < 1e-5
        
        print(f"Thermodynamic Consistency S (Num vs AD) Error: {err_S:.2e}")
        print(f"Thermodynamic Consistency M (Num vs AD) Error: {err_M:.2e}")

if __name__ == "__main__":
    validator = TFIMValidator()
    
    # 순차적으로 모든 검증 수행
    validator.run_ed_crosscheck()
    validator.run_scaling_tests()
    validator.run_sector_analysis()
    validator.run_consistency_stability()
    
    print("\n--- FINAL REPORT ---")
    for k, v in validator.report.items():
        print(f"{k.ljust(25)} : {'✅ PASS' if v else '❌ FAIL'}")