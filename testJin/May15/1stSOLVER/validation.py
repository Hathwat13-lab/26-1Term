import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.signal import find_peaks

# 앞서 완성한 완벽한 솔버를 불러옵니다.
from exact import vectorized_finite_observables, finite_free_energy

class TFIMValidator:
    def __init__(self):
        self.kb = 1.0
        # 검증 통과 여부를 기록할 딕셔너리
        self.report = {}

    # ==========================================
    # [1] Exact Diagonalization (독립 검증용 그라운드 트루스)
    # ==========================================
    def build_spin_hamiltonian(self, L, h):
        """JW 변환 없이 순수 스핀 행렬 텐서곱으로 해밀토니안 구성"""
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        eye = np.eye(2)
        
        H = np.zeros((2**L, 2**L))
        
        # - \sum Z_i Z_{i+1} (PBC)
        for i in range(L):
            j = (i + 1) % L
            term = 1
            for k in range(L):
                if k == i or k == j:
                    term = np.kron(term, sigma_z) if not isinstance(term, int) else sigma_z
                else:
                    term = np.kron(term, eye) if not isinstance(term, int) else eye
            H -= term
            
        # - h \sum X_i
        for i in range(L):
            term = 1
            for k in range(L):
                if k == i:
                    term = np.kron(term, sigma_x) if not isinstance(term, int) else sigma_x
                else:
                    term = np.kron(term, eye) if not isinstance(term, int) else eye
            H -= h * term
            
        return H

    def ed_thermodynamics(self, L, h, T_vals):
        """ED를 통한 정확한 분배함수 및 물리량 계산"""
        H = self.build_spin_hamiltonian(L, h)
        evals = eigh(H, eigvals_only=True)
        
        F_ed = []
        for T in T_vals:
            beta = 1.0 / T
            # 수치 안정성을 위해 최대 에너지를 빼고 계산 (LogSumExp 원리)
            E0 = evals[0]
            Z_shifted = np.sum(np.exp(-beta * (evals - E0)))
            F = E0 - T * np.log(Z_shifted)
            F_ed.append(F / L)
        return np.array(F_ed), evals[0] / L

    # ==========================================
    # [2] Test Suites (검증 항목들)
    # ==========================================
    def test_ed_comparison(self, L=6, h=1.0):
        """(3) 작은 시스템(L=6)에 대한 ED 결과와 JW 솔버 직접 비교"""
        print(f"Running Test: ED vs JW Comparison (L={L}, h={h})")
        T_vals = np.linspace(0.05, 2.0, 50)
        
        F_ed, GS_ed = self.ed_thermodynamics(L, h, T_vals)
        
        v_f, _, _, _, _ = vectorized_finite_observables(L)
        F_jw = np.array(v_f(jnp.array(T_vals), jnp.full_like(T_vals, h)))
        
        max_error = np.max(np.abs(F_ed - F_jw))
        passed = max_error < 1e-6
        self.report['ED_Match'] = passed
        print(f"  -> Max Error: {max_error:.2e} [{'PASS' if passed else 'FAIL'}]")
        
        return T_vals, F_ed, F_jw, GS_ed

    def test_fss_pseudocritical_drift(self):
        """(1) Susceptibility peak의 Finite-size scaling 및 Drift 확인"""
        print("Running Test: FSS Pseudocritical Drift")
        L_list = [4, 6, 8, 10, 12, 14, 16]
        T_fixed = 0.05
        h_vals = np.linspace(0.8, 1.2, 500)
        
        h_max_list = []
        chi_max_list = []
        
        for L in L_list:
            _, _, _, _, v_chi = vectorized_finite_observables(L)
            chi_vals = v_chi(jnp.full_like(h_vals, T_fixed), jnp.array(h_vals))
            idx_max = np.argmax(chi_vals)
            h_max_list.append(h_vals[idx_max])
            chi_max_list.append(chi_vals[idx_max])
            
        # h_max가 L이 커짐에 따라 1.0으로 수렴하는지 확인
        trend_is_correct = np.abs(h_max_list[-1] - 1.0) < np.abs(h_max_list[0] - 1.0)
        self.report['FSS_Drift'] = trend_is_correct
        print(f"  -> Drift towards h=1.0: [{'PASS' if trend_is_correct else 'FAIL'}]")
        
        return L_list, h_max_list, chi_max_list

    def test_thermodynamic_consistency(self, L=10):
        """(6) F, S, Cv 간의 열역학적 일관성 수치 미분 교차 검증"""
        print(f"Running Test: Thermodynamic Consistency (L={L})")
        T = 0.5
        h = 1.0
        dT = 1e-4
        
        f_fn, _, _, s_fn, _ = vectorized_finite_observables(L)
        
        # S = -dF/dT 확인 (수치 미분 vs 자동 미분)
        F_plus = f_fn(jnp.array([T + dT]), jnp.array([h]))[0]
        F_minus = f_fn(jnp.array([T - dT]), jnp.array([h]))[0]
        num_S = -(F_plus - F_minus) / (2 * dT)
        
        auto_S = s_fn(jnp.array([T]), jnp.array([h]))[0]
        
        error = np.abs(num_S - auto_S)
        passed = error < 1e-5
        self.report['Consistency_S'] = passed
        print(f"  -> S consistency error: {error:.2e} [{'PASS' if passed else 'FAIL'}]")

    def test_low_T_stability(self, L=16):
        """(5) h=1, 극저온에서의 Auto-diff 안정성 점검"""
        print(f"Running Test: Low-T AutoDiff Stability (L={L})")
        T_extreme = np.array([1e-4, 1e-3, 1e-2])
        h_crit = np.array([1.0, 1.0, 1.0])
        
        _, _, v_cv, _, v_chi = vectorized_finite_observables(L)
        cv_vals = np.array(v_cv(T_extreme, h_crit))
        chi_vals = np.array(v_chi(T_extreme, h_crit))
        
        # NaN이나 Inf가 발생했는지 확인
        passed = not (np.isnan(cv_vals).any() or np.isinf(cv_vals).any() or 
                      np.isnan(chi_vals).any() or np.isinf(chi_vals).any())
        
        self.report['Stability_LowT'] = passed
        print(f"  -> NaN/Inf Check: [{'PASS' if passed else 'FAIL'}]")

    # ==========================================
    # [3] Run & Visualize
    # ==========================================
    def run_all_and_plot(self):
        # 1. ED vs JW Comparison
        plt.figure(figsize=(10, 7))
        T_vals, F_ed, F_jw, GS_ed = self.test_ed_comparison()
        plt.plot(T_vals, F_ed, 'ko', label='ED (Exact)', alpha=0.5)
        plt.plot(T_vals, F_jw, 'r-', label='JW AutoDiff')
        plt.axhline(GS_ed, color='b', linestyle=':', label='Ground State Energy')
        plt.xlabel("Temperature (T)")
        plt.ylabel("Free Energy Density (F/L)")
        plt.title("ED vs JW AutoDiff (L=6, h=1.0)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('Valid_ED_Comparison.png', bbox_inches='tight')
        plt.close()
        print("  -> Saved Valid_ED_Comparison.png")

        # 2. FSS Drift
        plt.figure(figsize=(10, 7))
        L_list, h_max_list, chi_max_list = self.test_fss_pseudocritical_drift()
        plt.plot(1.0 / np.array(L_list), h_max_list, 'o-g')
        plt.xlabel("1 / L")
        plt.ylabel("Pseudocritical Field ($h_{max}$)")
        plt.title("Pseudocritical Drift towards Quantum Critical Point")
        plt.grid(True, alpha=0.3)
        plt.savefig('Valid_FSS_Drift.png', bbox_inches='tight')
        plt.close()
        print("  -> Saved Valid_FSS_Drift.png")

        # 3. Thermodynamic consistency & Stability
        self.test_thermodynamic_consistency()
        self.test_low_T_stability()
        
        # Print Summary
        print("\n" + "="*30)
        print(" VALIDATION SUMMARY REPORT ")
        print("="*30)
        for test, result in self.report.items():
            print(f"{test.ljust(20)}: {'PASS' if result else 'FAIL'}")

if __name__ == "__main__":
    validator = TFIMValidator()
    validator.run_all_and_plot()