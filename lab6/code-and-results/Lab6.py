import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from numpy.polynomial.legendre import leggauss
import os

# Utwórz folder na wykresy jeśli nie istnieje
if not os.path.exists('wykresy'):
    os.makedirs('wykresy')

# Funkcja podcałkowa
def f(x):
    return 4 / (1 + x**2)

# Dokładna wartość całki
exact_value = np.pi

# Zakres m
m_values = range(1, 26)

# Przechowywanie wyników
eval_counts = []
errors_midpoint = []
errors_trapezoid = []
errors_simpson = []

for m in m_values:
    # Liczba przedziałów i krok
    n = 2**m
    h = 1 / n

    # Węzły do trapezów i Simpsona (n+1 punktów)
    x_full = np.linspace(0, 1, n + 1)
    y_full = f(x_full)

    # Mid-point: środki przedziałów (n punktów)
    x_mid = np.linspace(h/2, 1 - h/2, n)
    y_mid = f(x_mid)
    midpoint_integral = h * np.sum(y_mid)

    # Trapezowa
    trapezoid_integral = np.trapezoid(y_full, x_full)

    # Simpson
    simpson_integral = integrate.simpson(y_full, x_full)

    # Obliczanie błędów względnych
    eval_count = n + 1
    eval_counts.append(eval_count)
    errors_midpoint.append(np.abs((midpoint_integral - exact_value) / exact_value))
    errors_trapezoid.append(np.abs((trapezoid_integral - exact_value) / exact_value))
    errors_simpson.append(np.abs((simpson_integral - exact_value) / exact_value))

# Wykres 1 - metody podstawowe
plt.figure(figsize=(10, 6))
plt.loglog(eval_counts, errors_midpoint, 'o-', label='Mid-point Rule')
plt.loglog(eval_counts, errors_trapezoid, 's-', label='Trapezoidal Rule')
plt.loglog(eval_counts, errors_simpson, '^-', label='Simpson\'s Rule')
plt.xlabel('Liczba ewaluacji funkcji ($n+1$)')
plt.ylabel('Bezwzględny błąd względny')
plt.title('Porównanie błędów numerycznego całkowania dla ∫₀¹ 4/(1+x²) dx')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('wykresy/wykres1_metody_podstawowe.png', dpi=300)
plt.close()

# 1.B - Znajdowanie minimalnego błędu
min_idx_mid = np.argmin(errors_midpoint)
min_idx_trap = np.argmin(errors_trapezoid)
min_idx_simp = np.argmin(errors_simpson)

h_min_mid = 1 / (2**m_values[min_idx_mid])
h_min_trap = 1 / (2**m_values[min_idx_trap])
h_min_simp = 1 / (2**m_values[min_idx_simp])

print(f"Midpoint: h_min = {h_min_mid:.1e}, min error = {errors_midpoint[min_idx_mid]:.1e}")
print(f"Trapezoid: h_min = {h_min_trap:.1e}, min error = {errors_trapezoid[min_idx_trap]:.1e}")
print(f"Simpson: h_min = {h_min_simp:.1e}, min error = {errors_simpson[min_idx_simp]:.1e}")

# 1.C - Empiryczny rząd zbieżności
errors_midpoint = np.array(errors_midpoint)
errors_trapezoid = np.array(errors_trapezoid)
errors_simpson = np.array(errors_simpson)
hs = np.array([1 / (2**m) for m in m_values])

# Zakres m (np. 5 do 15)
start = 4   # m = 5
end = 14    # m = 15 (czyli indeksy 4:15)

log_h = np.log(hs[start:end])
log_err_mid = np.log(errors_midpoint[start:end])
log_err_trap = np.log(errors_trapezoid[start:end])
log_err_simp = np.log(errors_simpson[start:end])

# Regresja liniowa log-log
p_mid, _ = np.polyfit(log_h, log_err_mid, 1)
p_trap, _ = np.polyfit(log_h, log_err_trap, 1)
p_simp, _ = np.polyfit(log_h, log_err_simp, 1)

print(f"Empiryczny rząd zbieżności:")
print(f"  Mid-point rule:     {abs(p_mid):.2f} (teoria: 2)")
print(f"  Trapezoidal rule:   {abs(p_trap):.2f} (teoria: 2)")
print(f"  Simpson's rule:     {abs(p_simp):.2f} (teoria: 4)")

# 2 - Gauss-Legendre
n_gauss_range = range(2, 101)

gauss_errors = []
gauss_evals = []

for n in n_gauss_range:
    nodes, weights = leggauss(n)
    x_mapped = 0.5 * (nodes + 1)
    w_mapped = 0.5 * weights
    integral_gauss = np.sum(w_mapped * f(x_mapped))
    error = np.abs((integral_gauss - exact_value) / exact_value)
    gauss_errors.append(error)
    gauss_evals.append(n)

# Wykres 2 - wszystkie metody
plt.figure(figsize=(10, 6))
plt.loglog(eval_counts, errors_midpoint, 'o-', label='Mid-point Rule')
plt.loglog(eval_counts, errors_trapezoid, 's-', label='Trapezoidal Rule')
plt.loglog(eval_counts, errors_simpson, '^-', label="Simpson's Rule")
plt.loglog(gauss_evals, gauss_errors, 'd-', label="Gauss-Legendre")

plt.xlabel('Liczba ewaluacji funkcji ($n+1$ lub $n$ dla Gaussa)')
plt.ylabel('Bezwzględny błąd względny')
plt.title('Porównanie błędów metod całkowania ∫₀¹ 4/(1+x²) dx')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('wykresy/wykres2_wszystkie_metody.png', dpi=300)
plt.close()

print("Wykresy zostały zapisane w folderze 'wykresy/'")