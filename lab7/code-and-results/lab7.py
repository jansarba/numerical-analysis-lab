# Lab7.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from numpy.polynomial.legendre import leggauss
import os

# Utwórz folder na wykresy jeśli nie istnieje
if not os.path.exists('wykresy_lab7'):
    os.makedirs('wykresy_lab7')


# --- Implementacja rekursywnej kwadratury adaptacyjnej trapezów ---
class FuncCounter:
    def __init__(self, func_to_call):
        self.func = func_to_call
        self.eval_count = 0

    def __call__(self, x):
        if np.isscalar(x):
            self.eval_count += 1
        else:
            self.eval_count += np.size(x)
        return self.func(x)

    def reset_count(self):
        self.eval_count = 0


def recursive_adaptive_trapezoid(counter, a, b, tol, S_ab, fa, fb, max_depth, current_depth=0,
                                 global_max_evals=float('inf')):
    """
    Rekursywna kwadratura adaptacyjna trapezów.
    ... (reszta docstring bez zmian) ...
    global_max_evals: globalny limit ewaluacji dla tej całki adaptacyjnej.
    """
    if current_depth > max_depth:
        return S_ab

    if counter.eval_count > global_max_evals:  # Dodatkowe sprawdzenie globalnego limitu
        # print(f"Adaptive Trapezoid: Osiągnięto global_max_evals ({global_max_evals}) przy tol={tol}")
        return S_ab  # Zwracamy co mamy

    c = (a + b) / 2.0
    fc = counter(c)

    if counter.eval_count > global_max_evals:  # Sprawdzenie po ewaluacji fc
        return S_ab

    S_ac = (fa + fc) * (c - a) / 2.0
    S_cb = (fc + fb) * (b - c) / 2.0
    S_acb = S_ac + S_cb

    if np.abs(S_acb - S_ab) < tol:
        return S_acb
    else:
        left_integral = recursive_adaptive_trapezoid(counter, a, c, tol / 2.0, S_ac, fa, fc, max_depth,
                                                     current_depth + 1, global_max_evals)
        if counter.eval_count > global_max_evals: return left_integral  # Jeśli przekroczono w lewej gałęzi

        right_integral = recursive_adaptive_trapezoid(counter, c, b, tol / 2.0, S_cb, fc, fb, max_depth,
                                                      current_depth + 1, global_max_evals)
        return left_integral + right_integral


# --- Główna funkcja do przetwarzania i rysowania dla danego zadania ---
def process_integral(f_inspect, exact_value_inspect, integral_limits, plot_filename_suffix, integral_name,
                     m_values_override=None, n_gauss_range_override=None,
                     num_tolerances_override=None, max_evals_adaptive_trap=10 ** 7,
                     max_evals_adaptive_gk=10 ** 7):  # Dodane limity dla metod adaptacyjnych
    print(f"\n--- Przetwarzanie całki: {integral_name} ---")

    m_values = m_values_override if m_values_override is not None else range(1, 21)
    n_gauss_range = n_gauss_range_override if n_gauss_range_override is not None else range(2, 101)
    num_tolerances = num_tolerances_override if num_tolerances_override is not None else 30

    eval_counts_lab6 = []
    errors_midpoint = []
    errors_trapezoid = []
    errors_simpson = []

    print(f"  Metody NC: m_values = {list(m_values)}")
    for m in m_values:
        n = 2 ** m
        a, b = integral_limits
        h = (b - a) / n
        x_full = np.linspace(a, b, n + 1)
        y_full = f_inspect(x_full)
        x_mid = np.linspace(a + h / 2, b - h / 2, n)
        y_mid = f_inspect(x_mid)
        midpoint_integral = h * np.sum(y_mid)
        trapezoid_integral = integrate.trapezoid(y_full, x_full)
        simpson_integral = integrate.simpson(y_full, x_full)
        eval_count_nc = n + 1
        eval_counts_lab6.append(eval_count_nc)
        errors_midpoint.append(np.abs((midpoint_integral - exact_value_inspect) / exact_value_inspect))
        errors_trapezoid.append(np.abs((trapezoid_integral - exact_value_inspect) / exact_value_inspect))
        errors_simpson.append(np.abs((simpson_integral - exact_value_inspect) / exact_value_inspect))

    gauss_errors = []
    gauss_evals = []
    a_gl, b_gl = integral_limits
    map_nodes = lambda nodes: 0.5 * (b_gl - a_gl) * nodes + 0.5 * (b_gl + a_gl)
    map_weights = lambda weights: 0.5 * (b_gl - a_gl) * weights

    print(f"  Gauss-Legendre: n_gauss_range = {list(n_gauss_range)}")
    for n_g in n_gauss_range:
        nodes, weights = leggauss(n_g)
        x_mapped = map_nodes(nodes)
        w_mapped = map_weights(weights)
        integral_gauss = np.sum(w_mapped * f_inspect(x_mapped))
        error = np.abs((integral_gauss - exact_value_inspect) / exact_value_inspect)
        gauss_errors.append(error)
        gauss_evals.append(n_g)

    tolerances = np.logspace(0, -14, num_tolerances)

    errors_adaptive_trap = []
    evals_adaptive_trap = []
    func_counter_trap = FuncCounter(f_inspect)
    max_recursion_depth = 25  # Można trochę zmniejszyć, jeśli nadal za wolno

    print(
        f"  Adaptacyjne Trapezy: {num_tolerances} tolerancji, max_recursion_depth={max_recursion_depth}, max_evals_per_tol={max_evals_adaptive_trap}")
    for tol_idx, tol in enumerate(tolerances):
        func_counter_trap.reset_count()
        fa = func_counter_trap(integral_limits[0])
        fb = func_counter_trap(integral_limits[1])
        S_ab = (fa + fb) * (integral_limits[1] - integral_limits[0]) / 2.0

        try:
            integral_val = recursive_adaptive_trapezoid(func_counter_trap,
                                                        integral_limits[0], integral_limits[1],
                                                        tol, S_ab, fa, fb, max_recursion_depth,
                                                        global_max_evals=max_evals_adaptive_trap)  # Przekazanie limitu

            current_evals = func_counter_trap.eval_count
            if current_evals > max_evals_adaptive_trap and tol_idx > 0:  # Jeśli przekroczono limit
                print(
                    f"    Adaptive Trapezoid: Przekroczono max_evals ({max_evals_adaptive_trap}) dla tol={tol}. Przerywam dla tej metody.")
                break  # Przerywamy pętlę po tolerancjach dla tej metody

            err = np.abs((integral_val - exact_value_inspect) / exact_value_inspect)
            if np.isfinite(err) and err > 1e-16:
                errors_adaptive_trap.append(err)
                evals_adaptive_trap.append(current_evals)

        except RecursionError:
            print(f"    Adaptive Trapezoid: Osiągnięto maksymalną głębokość rekursji dla tol={tol}.")
            if func_counter_trap.eval_count > max_evals_adaptive_trap and tol_idx > 0: break
            pass

    errors_gauss_kronrod = []
    evals_gauss_kronrod = []

    print(f"  Adaptacyjne Gauss-Kronrod: {num_tolerances} tolerancji, max_evals_per_tol={max_evals_adaptive_gk}")
    for tol_idx, tol in enumerate(tolerances):
        # quad_vec nie ma bezpośredniego parametru maxfev jak quad
        # Możemy symulować limit przez sprawdzanie info.neval, ale to mniej eleganckie
        # Dla quad_vec trudniej jest narzucić twardy limit ewaluacji *dla pojedynczego wywołania*
        # Zamiast tego, możemy przerwać pętlę po tolerancjach, jeśli całkowita liczba ewaluacji (sumaryczna) rośnie zbyt szybko
        # lub jeśli pojedyncze wywołanie zużyje za dużo

        # Tutaj zrobimy prostsze podejście: jeśli po kilku tolerancjach liczba ewaluacji jest bardzo duża, przerwiemy
        if sum(evals_gauss_kronrod) > max_evals_adaptive_gk * 2 and tol_idx > 5:  # Prosty heurystyczny limit sumaryczny
            print(
                f"    Adaptive Gauss-Kronrod: Sumaryczna liczba ewaluacji przekracza limit. Przerywam dla tej metody.")
            break

        val_gk_arr, err_est_gk_arr, info_gk = integrate.quad_vec(
            f_inspect, integral_limits[0], integral_limits[1],
            epsabs=tol, epsrel=tol, full_output=True
            # limit= można by użyć z quad, ale quad_vec nie ma tego bezpośrednio
        )
        integral_val_gk = val_gk_arr
        num_evals_gk = info_gk.neval

        if num_evals_gk > max_evals_adaptive_gk and tol_idx > 0:  # Limit na pojedyncze wywołanie
            print(
                f"    Adaptive Gauss-Kronrod: Przekroczono max_evals ({max_evals_adaptive_gk}) dla tol={tol}. Przerywam dla tej metody.")
            break

        err_gk = np.abs((integral_val_gk - exact_value_inspect) / exact_value_inspect)
        if np.isfinite(err_gk):
            errors_gauss_kronrod.append(err_gk)
            evals_gauss_kronrod.append(num_evals_gk)

    if evals_adaptive_trap:
        sorted_indices_trap = np.argsort(evals_adaptive_trap)
        evals_adaptive_trap = np.array(evals_adaptive_trap)[sorted_indices_trap]
        errors_adaptive_trap = np.array(errors_adaptive_trap)[sorted_indices_trap]

    if evals_gauss_kronrod:
        sorted_indices_gk = np.argsort(evals_gauss_kronrod)
        evals_gauss_kronrod = np.array(evals_gauss_kronrod)[sorted_indices_gk]
        errors_gauss_kronrod = np.array(errors_gauss_kronrod)[sorted_indices_gk]

    plt.figure(figsize=(12, 8))
    if len(eval_counts_lab6) > 0:  # Dodatkowe sprawdzenie, czy są dane
        plt.loglog(eval_counts_lab6, errors_midpoint, 'o-', label='Mid-point Rule (Lab6)')
        plt.loglog(eval_counts_lab6, errors_trapezoid, 's-', label='Trapezoidal Rule (Lab6)')
        plt.loglog(eval_counts_lab6, errors_simpson, '^-', label="Simpson's Rule (Lab6)")
    if len(gauss_evals) > 0:
        plt.loglog(gauss_evals, gauss_errors, 'd-', label="Gauss-Legendre (Lab6)")

    if len(evals_adaptive_trap) > 0:
        plt.loglog(evals_adaptive_trap, errors_adaptive_trap, 'x-', label='Adaptive Trapezoid (Lab7)')
    if len(evals_gauss_kronrod) > 0:
        plt.loglog(evals_gauss_kronrod, errors_gauss_kronrod, 'p-', label="Adaptive Gauss-Kronrod (Lab7)")

    plt.xlabel('Liczba ewaluacji funkcji')
    plt.ylabel('Bezwzględny błąd względny')
    plt.title(f'Porównanie metod całkowania dla: {integral_name}')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.7))
    plt.grid(True, which="both", ls="--")
    plt.ylim(bottom=1e-16)
    plt.tight_layout()
    plt.savefig(f'wykresy_lab7/wykres_zbiorczy_{plot_filename_suffix}.png', dpi=300)
    plt.close()
    print(f"Wykres zapisano jako: wykresy_lab7/wykres_zbiorczy_{plot_filename_suffix}.png")


# --- Definicje funkcji i wartości dokładnych dla poszczególnych zadań ---
def f1(x): return 4 / (1 + x ** 2)


exact_value1 = np.pi


def f2a(x):
    if isinstance(x, (np.ndarray, list)):
        x_arr = np.asarray(x);
        res = np.zeros_like(x_arr, dtype=float)
        mask = x_arr > 0;
        res[mask] = np.sqrt(x_arr[mask]) * np.log(x_arr[mask]);
        return res
    else:
        return 0.0 if x == 0 else (np.sqrt(x) * np.log(x) if x > 0 else np.nan)


exact_value2a = -4 / 9

a_const = 0.001;
b_const = 0.004;
x0_1 = 0.3;
x0_2 = 0.9


def f2b(x): return 1 / ((x - x0_1) ** 2 + a_const) + 1 / ((x - x0_2) ** 2 + b_const) - 6


def exact_integral_term(x0, c, l=(0, 1)): v = np.sqrt(c); return (1 / v) * (
            np.arctan((l[1] - x0) / v) - np.arctan((l[0] - x0) / v))


exact_value2b = exact_integral_term(x0_1, a_const) + exact_integral_term(x0_2, b_const) - 6

# --- Główny blok wykonawczy ---
if __name__ == '__main__':
    process_integral(f_inspect=f1,
                     exact_value_inspect=exact_value1,
                     integral_limits=(0, 1),
                     plot_filename_suffix="zad1_pi",
                     integral_name="∫₀¹ 4/(1+x²) dx")

    process_integral(f_inspect=f2a,
                     exact_value_inspect=exact_value2a,
                     integral_limits=(0, 1),
                     plot_filename_suffix="zad2a_sqrtxlogx",
                     integral_name="∫₀¹ √x log x dx")

    print("\nUruchamianie ostatniej całki z ograniczonymi parametrami...")
    process_integral(f_inspect=f2b,
                     exact_value_inspect=exact_value2b,
                     integral_limits=(0, 1),
                     plot_filename_suffix="zad2b_terms_minus_6_optimized",
                     integral_name="∫₀¹ [1/((x-0.3)²+a) + ... - 6] dx (opt.)",
                     m_values_override=range(1, 16),  # Zmniejszone m dla NC
                     n_gauss_range_override=range(2, 51),  # Zmniejszony zakres dla Gaussa
                     num_tolerances_override=20,  # Mniej tolerancji
                     max_evals_adaptive_trap=5 * 10 ** 5,  # Limit ewaluacji dla adapt. trap.
                     max_evals_adaptive_gk=10 ** 6)  # Limit ewaluacji dla adapt. GK

    print("\nZakończono Lab7.")