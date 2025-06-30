# Lab8.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import os

# Utwórz folder na wykresy jeśli nie istnieje
if not os.path.exists('wykresy_lab8'):
    os.makedirs('wykresy_lab8')

# --- Zadanie 1: Problemy z metodą Newtona ---
print("--- Zadanie 1: Problemy z metodą Newtona ---")


# (a) f(x) = x^3 - 5*x, x0 = 1
def f1a(x): return x ** 3 - 5 * x


def df1a(x): return 3 * x ** 2 - 5


print("\n(a) f(x) = x^3 - 5*x, x0 = 1")
print("Wyjaśnienie: Metoda Newtona wpada w cykl. x1 = 1 - f(1)/f'(1) = 1 - (-4)/(-2) = 1 - 2 = -1.")
print("x2 = -1 - f(-1)/f'(-1) = -1 - (4)/(-2) = -1 + 2 = 1. Cykl: 1, -1, 1, ...")
try:
    root_newton = optimize.newton(f1a, 1, fprime=df1a, maxiter=10)  # maxiter niski by pokazać problem
    print(f"Newton (standard): {root_newton} (może nie zbiec lub zbiec wolno/do dalekiego)")
except RuntimeError:
    print("Newton (standard): Nie zbiegł (zgodnie z oczekiwaniami).")

# Modyfikacja: użycie innej metody lub innego x0
# Pierwiastki to 0, +/-sqrt(5) (+/-2.236)
root_brentq_a1 = optimize.brentq(f1a, -3, -1)  # przedział dla -sqrt(5)
root_brentq_a2 = optimize.brentq(f1a, -0.5, 0.5)  # przedział dla 0
root_brentq_a3 = optimize.brentq(f1a, 1, 3)  # przedział dla sqrt(5)
print(f"Brentq pierwiastki: {root_brentq_a1}, {root_brentq_a2}, {root_brentq_a3}")
root_newton_mod_a = optimize.newton(f1a, 2, fprime=df1a)  # Inny x0
print(f"Newton (x0=2): {root_newton_mod_a} (zbiega do sqrt(5))")


# (b) f(x) = x^3 - 3*x + 1, x0 = 1
def f1b(x): return x ** 3 - 3 * x + 1


def df1b(x): return 3 * x ** 2 - 3


print("\n(b) f(x) = x^3 - 3*x + 1, x0 = 1")
print("Wyjaśnienie: f'(1) = 3*(1)^2 - 3 = 0. Pochodna w punkcie startowym jest zerowa, dzielenie przez zero.")
try:
    root_newton = optimize.newton(f1b, 1, fprime=df1b)
    print(f"Newton (standard): {root_newton}")
except RuntimeError as e:
    print(f"Newton (standard): Nie zbiegł - {e}")

# Modyfikacja: użycie metody siecznych (fprime=None) lub innego x0
root_newton_secant_b = optimize.newton(f1b, 1)  # domyślnie metoda siecznych jeśli fprime=None
print(f"Newton (metoda siecznych, x0=1): {root_newton_secant_b}")
root_newton_mod_b = optimize.newton(f1b, 0.5, fprime=df1b)  # Inny x0
print(f"Newton (x0=0.5): {root_newton_mod_b}")


# Pierwiastki: ok. -1.879, 0.347, 1.532


# (c) f(x) = 2 - x^5, x0 = 0.01
def f1c(x): return 2 - x ** 5


def df1c(x): return -5 * x ** 4


print("\n(c) f(x) = 2 - x^5, x0 = 0.01")
print("Wyjaśnienie: f'(x0) jest bardzo małe (-5e-8). Może prowadzić do dużego kroku i 'przestrzelenia' pierwiastka.")
# x1 = x0 - f(x0)/f'(x0) = 0.01 - (2 - 0.01^5)/(-5*0.01^4) = 0.01 - 2/(-5e-8) = 0.01 + 4e7 - ogromny krok.
try:
    # Domyślne tol może być zbyt małe dla tak dużego kroku początkowego
    root_newton = optimize.newton(f1c, 0.01, fprime=df1c, tol=1e-2, maxiter=50)
    print(f"Newton (standard, większe tol): {root_newton}")
except RuntimeError as e:
    print(f"Newton (standard): Nie zbiegł - {e}")

# Modyfikacja: inny x0 lub metoda bez pochodnej
root_actual_c = 2 ** (1 / 5)  # ok 1.148
print(f"Dokładny pierwiastek: {root_actual_c}")
root_newton_mod_c = optimize.newton(f1c, 1, fprime=df1c)  # x0 bliżej pierwiastka
print(f"Newton (x0=1): {root_newton_mod_c}")
root_brentq_c = optimize.brentq(f1c, 0, 2)
print(f"Brentq: {root_brentq_c}")


# (d) f(x) = x^4 - 4.29*x^2 - 5.29, x0 = 0.8
def f1d(x): return x ** 4 - 4.29 * x ** 2 - 5.29


def df1d(x): return 4 * x ** 3 - 2 * 4.29 * x


print("\n(d) f(x) = x^4 - 4.29*x^2 - 5.29, x0 = 0.8")
# Pierwiastki rzeczywiste: x^2 = 5.29 => x = +/-2.3
# f'(x) = x(4x^2 - 8.58). Miejsca zerowe pochodnej to x=0 i x=+/-(8.58/4) ~ +/-1.4645
# x0=0.8 jest dość daleko od pierwiastków +/-2.3. Może zbiegać wolno lub do innego pierwiastka.
# f(0.8) = 0.8^4 - 4.29*0.8^2 - 5.29 = 0.4096 - 2.7456 - 5.29 = -7.626
# f'(0.8) = 0.8 * (4*0.8^2 - 8.58) = 0.8 * (2.56 - 8.58) = 0.8 * (-6.02) = -4.816
# x1 = 0.8 - (-7.626 / -4.816) = 0.8 - 1.583... = -0.783...
# Wygląda na to, że może zbiec.
try:
    root_newton = optimize.newton(f1d, 0.8, fprime=df1d, maxiter=50)
    print(f"Newton (standard, x0=0.8): {root_newton}")  # Powinien zbiec do -2.3
except RuntimeError as e:
    print(f"Newton (standard): Nie zbiegł - {e}")

# Modyfikacja (jeśli standardowa zawodzi lub dla demonstracji):
root_brentq_d1 = optimize.brentq(f1d, 2, 3)  # dla 2.3
root_brentq_d2 = optimize.brentq(f1d, -3, -2)  # dla -2.3
print(f"Brentq pierwiastki: {root_brentq_d1}, {root_brentq_d2}")
print(
    "Wyjaśnienie: W tym przypadku Newton z x0=0.8 powinien zbiec do -2.3. Jeśli nie, może to być kwestia liczby iteracji lub tolerancji. Ogólnie, start daleko od pierwiastka lub blisko ekstremum lokalnego może być problemem.")

# --- Zadanie 2: Schematy iteracyjne ---
print("\n--- Zadanie 2: Schematy iteracyjne dla f(x) = x^2 - 3*x + 2 = 0 ---")


# Pierwiastki: x=1, x=2. Badamy zbieżność do alpha = 2.

def phi1(x): return (x ** 2 + 2) / 3


def dphi1(x): return 2 * x / 3


def phi2(x): return np.sqrt(3 * x - 2) if (3 * x - 2) >= 0 else np.nan  # Zabezpieczenie przed ujemnym pierwiastkiem


def dphi2(x): return 3 / (2 * np.sqrt(3 * x - 2)) if (3 * x - 2) > 0 else np.nan


def phi3(x): return 3 - 2 / x if x != 0 else np.nan


def dphi3(x): return 2 / x ** 2 if x != 0 else np.nan


def phi4(x):  # Metoda Newtona
    # f(x) = x^2 - 3x + 2
    # f'(x) = 2x - 3
    # phi(x) = x - f(x)/f'(x) = x - (x^2 - 3x + 2)/(2x - 3)
    #        = (x(2x-3) - (x^2-3x+2))/(2x-3)
    #        = (2x^2-3x - x^2+3x-2)/(2x-3) = (x^2-2)/(2x-3)
    if (2 * x - 3) == 0: return np.nan
    return (x ** 2 - 2) / (2 * x - 3)


def dphi4(x):  # Pochodna metody Newtona
    if (2 * x - 3) == 0: return np.nan
    # (2x(2x-3)-(x^2-2)*2)/(2x-3)^2 = (4x^2-6x-2x^2+4)/(2x-3)^2 = (2x^2-6x+4)/(2x-3)^2
    return (2 * x ** 2 - 6 * x + 4) / (2 * x - 3) ** 2


alpha = 2.0  # Dokładny pierwiastek

print("\n(a) Analiza teoretyczna zbieżności do alpha = 2:")
print(f"phi1(x): |phi1'(2)| = |{dphi1(alpha):.4f}|. > 1, więc dywergencja.")
print(f"phi2(x): |phi2'(2)| = |{dphi2(alpha):.4f}|. < 1, więc zbieżność liniowa.")
print(f"phi3(x): |phi3'(2)| = |{dphi3(alpha):.4f}|. < 1, więc zbieżność liniowa.")
print(f"phi4(x): |phi4'(2)| = |{dphi4(alpha):.4f}|. = 0, więc zbieżność co najmniej kwadratowa (faktycznie kwadratowa).")

# (b) Implementacja i weryfikacja
x0_zad2 = 1.5  # Punkt startowy (blisko 2, ale nie 2)
num_iterations = 10
schemes = {
    "fi1(x)": phi1, # Zmieniono nazwy dla legendy, aby były bardziej polskie
    "fi2(x)": phi2,
    "fi3(x)": phi3,
    "fi4(x) (Newton)": phi4
}
results = {}

print(f"\n(b) Iteracje (x0 = {x0_zad2}):")
for name, phi_func in schemes.items():
    xs = [x0_zad2]
    errors = [np.abs(x0_zad2 - alpha)]
    print(f"\nMetoda: {name}")
    print(f"Iter 0: x = {x0_zad2:.8f}, błąd = {errors[0]:.2e}")
    for k in range(num_iterations):
        xk_plus_1 = phi_func(xs[-1])
        if np.isnan(xk_plus_1) or np.isinf(xk_plus_1):
            print(f"Iter {k + 1}: x = {xk_plus_1} - przerwanie iteracji.")
            break
        xs.append(xk_plus_1)
        err = np.abs(xk_plus_1 - alpha)
        errors.append(err)
        print(f"Iter {k + 1}: x = {xs[-1]:.8f}, błąd = {errors[-1]:.2e}")
        if err < 1e-15:  # Zbiegło do dokładności maszynowej
            print("Osiągnięto dokładność maszynową.")
            break
    results[name] = {'xs': xs, 'errors': errors}

# Wyznaczanie eksperymentalnego rzędu zbieżności
print("\nEksperymentalny rząd zbieżności (r):")
for name, data in results.items():
    errs = np.array(data['errors'])
    if len(errs) < 3:
        print(f"{name}: Za mało iteracji do obliczenia r.")
        continue

    valid_indices = np.where((errs > 1e-16) & (np.isfinite(errs)))[0]
    if len(valid_indices) < 3:
        print(f"{name}: Za mało poprawnych błędów do obliczenia r.")
        continue

    errs_valid = errs[valid_indices]

    if len(errs_valid) >= 3:
        eps_kp1 = errs_valid[-1]
        eps_k = errs_valid[-2]
        eps_km1 = errs_valid[-3]

        if eps_k > 1e-17 and eps_km1 > 1e-17 and eps_kp1 > 1e-17 and \
                (eps_k / eps_kp1) > 0 and (eps_km1 / eps_k) > 0 and \
                np.abs(np.log(eps_km1 / eps_k)) > 1e-9:

            log_eps_kp1 = np.log(eps_kp1)
            log_eps_k = np.log(eps_k)
            log_eps_km1 = np.log(eps_km1)

            numerator = log_eps_k - log_eps_kp1
            denominator = log_eps_km1 - log_eps_k

            if np.abs(denominator) > 1e-9:
                r_exp = numerator / denominator
                print(f"{name}: r approx {r_exp:.2f}")
            else:
                print(f"{name}: Nie można obliczyć r (dzielnik bliski zera).")
        else:
            print(f"{name}: Nie można obliczyć r (problemy z log lub wartościami błędów).")

# (c) Wykresy błędów
plt.figure(figsize=(10, 7))
for name, data in results.items():
    iterations_plot = range(len(data['errors']))
    plt.semilogy(iterations_plot, data['errors'], 'o-', label=name.replace("φ", "fi")) # Zmiana dla legendy na wykresie

plt.xlabel("Numer iteracji (k)")
plt.ylabel("Błąd bezwzględny |x_k - alpha| (skala log)")
plt.title("Zbieżność schematów iteracyjnych do alpha=2 (wszystkie metody)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.ylim(bottom=1e-5)
plt.savefig('wykresy_lab8/zad2c_wszystkie_metody.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 7))
convergent_methods = ["fi2(x)", "fi3(x)", "fi4(x) (Newton)"]
for name in convergent_methods:
    if name in results: # Sprawdzenie czy klucz istnieje
        data = results[name]
        iterations_plot = range(len(data['errors']))
        plt.semilogy(iterations_plot, data['errors'], 'o-', label=name.replace("φ", "fi")) # Zmiana dla legendy

plt.xlabel("Numer iteracji (k)")
plt.ylabel("Błąd bezwzględny |x_k - alpha| (skala log)")
plt.title("Zbieżność schematów iteracyjnych do alpha=2 (metody zbieżne)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.ylim(bottom=1e-5)
plt.savefig('wykresy_lab8/zad2c_zbiezne_metody.png', dpi=300)
plt.close()
print("Wykresy dla Zadania 2 zapisano.")

# --- Zadanie 3: Schemat Newtona i precyzja ---
print("\n--- Zadanie 3: Schemat Newtona i precyzja ---")
print("(a) f(x) = x^3 - 2x - 5 = 0")
print("   f'(x) = 3x^2 - 2")
print("   x_{k+1} = x_k - (x_k^3 - 2x_k - 5) / (3x_k^2 - 2)")

print("(b) f(x) = e^-x - x = 0")
print("   f'(x) = -e^-x - 1")
print("   x_{k+1} = x_k - (e^-x_k - x_k) / (-e^-x_k - 1)")

print("(c) f(x) = x sin(x) - 1 = 0")
print("   f'(x) = sin(x) + x cos(x)")
print("   x_{k+1} = x_k - (x_k sin(x_k) - 1) / (sin(x_k) + x_k cos(x_k))")

print("\nLiczba iteracji dla precyzji (teoretycznie, dla kwadratowej zbieżności):")
print("Zakładając, że liczba poprawnych bitów z grubsza podwaja się w każdej iteracji:")
print("Start: 4 bity")
print("Iter 1: ~8 bitów")
print("Iter 2: ~16 bitów")
print("Iter 3: ~32 bity")
print("Iter 4: ~64 bity")
print("Aby osiągnąć 24-bitową dokładność: potrzeba około 3 iteracji.")
print("Aby osiągnąć 53-bitową dokładność: potrzeba około 4 iteracji.")

# --- Zadanie 4: Metoda Newtona dla układu równań ---
print("\n--- Zadanie 4: Metoda Newtona dla układu równań ---")
# Układ z treści zadania (zmodyfikowany zgodnie z zapytaniem):
# f1(x1,x2) = x1^2 + x2^2 - 1 = 0
# f2(x1,x2) = x1^2 - x2 = 0
print("Układ równań:")
print("f1(x1, x2) = x1^2 + x2^2 - 1 = 0")
print("f2(x1, x2) = x1^2 - x2 = 0")

def F_system_zad4(X):
    x1, x2 = X
    f1 = x1**2 + x2**2 - 1
    f2 = x1**2 - x2
    return np.array([f1, f2])

def J_system_zad4(X):
    x1, x2 = X
    # df1/dx1 = 2*x1, df1/dx2 = 2*x2
    # df2/dx1 = 2*x1, df2/dx2 = -1
    return np.array([
        [2 * x1, 2 * x2],
        [2 * x1, -1]
    ])

# Dokładne rozwiązania (są dwa symetryczne dla x1):
# x2 = (-1 + sqrt(5))/2
# x1 = +/- sqrt(x2)
x2_exact_zad4 = (np.sqrt(5) - 1) / 2
x1_exact_zad4_pos = np.sqrt(x2_exact_zad4)
x1_exact_zad4_neg = -np.sqrt(x2_exact_zad4)

# Wybieramy jedno z rozwiązań do testowania zbieżności, np. z x1 > 0
X_exact_zad4 = np.array([x1_exact_zad4_pos, x2_exact_zad4])
print(f"Oczekiwane rozwiązanie (jedno z dwóch, x1 > 0): x1 approx {X_exact_zad4[0]:.6f}, x2 approx {X_exact_zad4[1]:.6f}")

# Iteracja Newtona
# Punkty startowe powinny być rozsądnie blisko rozwiązania
# np. x1 ~ sqrt(0.6) ~ 0.77, x2 ~ 0.6
X0_zad4 = np.array([0.7, 0.7])  # Punkt startowy
print(f"Punkt startowy: {X0_zad4}")

Xk = X0_zad4
num_iter_system = 10
errors_system_zad4 = []

print("Iteracje Newtona dla układu z Zadania 4:")
for k in range(num_iter_system):
    Fk = F_system_zad4(Xk)
    Jk = J_system_zad4(Xk)

    if np.abs(np.linalg.det(Jk)) < 1e-10:
        print(f"Iter {k}: Jacobian osobliwy. Zatrzymanie.")
        break

    delta_Xk = np.linalg.solve(Jk, -Fk)
    Xk_plus_1 = Xk + delta_Xk

    if np.linalg.norm(X_exact_zad4) > 1e-9: # Unikamy dzielenia przez zero jeśli X_exact jest [0,0]
        error_norm = np.linalg.norm(Xk_plus_1 - X_exact_zad4) / np.linalg.norm(X_exact_zad4)
    else:
        error_norm = np.linalg.norm(Xk_plus_1 - X_exact_zad4) # Błąd absolutny jeśli X_exact jest [0,0]

    errors_system_zad4.append(error_norm)

    print(f"Iter {k + 1}: X = [{Xk_plus_1[0]:.6f}, {Xk_plus_1[1]:.6f}], błąd względny = {error_norm:.2e}")

    if error_norm < 1e-10:
        print("Osiągnięto zbieżność.")
        break
    Xk = Xk_plus_1

if not errors_system_zad4:
    print("Nie udało się wykonać iteracji.")
elif k == num_iter_system - 1 and errors_system_zad4[-1] > 1e-8 : # Jeśli ostatni błąd jest duży
    print(f"Metoda Newtona dla układu nie zbiegła wystarczająco szybko do X_exact w {num_iter_system} iteracjach.")


# Użycie scipy.optimize.root
sol_scipy_zad4 = optimize.root(F_system_zad4, X0_zad4, jac=J_system_zad4, method='hybr') # Można też 'lm'
print(f"\nRozwiązanie z scipy.optimize.root dla Zadania 4: {sol_scipy_zad4.x}")
if sol_scipy_zad4.success:
    if np.linalg.norm(X_exact_zad4) > 1e-9:
        error_scipy_zad4 = np.linalg.norm(sol_scipy_zad4.x - X_exact_zad4) / np.linalg.norm(X_exact_zad4)
    else:
        error_scipy_zad4 = np.linalg.norm(sol_scipy_zad4.x - X_exact_zad4)
    print(f"Błąd względny rozwiązania scipy dla Zadania 4: {error_scipy_zad4:.2e}")
else:
    print("scipy.optimize.root nie znalazło rozwiązania dla Zadania 4.")

print("\nZakończono Lab8.")