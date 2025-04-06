import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Dane wejściowe (lata i populacje)
years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980])
population = np.array([76212168, 92228496, 106021537, 123202624, 132164569,
                       151325798, 179323175, 203302031, 226542199])

# 1. φ(t) = t^(j−1)
V1 = np.vander(years, increasing=True)

# 2. φ(t) = (t−1900)^(j−1)
V2 = np.vander(years - 1900, increasing=True)

# 3. φ(t) = (t−1940)^(j−1)
V3 = np.vander(years - 1940, increasing=True)

# 4. φ(t) = ((t−1940)/40)^(j−1)
V4 = np.vander((years - 1940) / 40, increasing=True)

# Wyświetlenie rozmiaru i przykładowych wartości (opcjonalnie)
print("Macierz V1 (bazowa t^j):\n", V1)
print("Macierz V2 (bazowa (t-1900)^j):\n", V2)
print("Macierz V3 (bazowa (t-1940)^j):\n", V3)
print("Macierz V4 (bazowa ((t-1940)/40)^j):\n", V4)

# B

# Obliczenie współczynników uwarunkowania
cond_V1 = np.linalg.cond(V1)
cond_V2 = np.linalg.cond(V2)
cond_V3 = np.linalg.cond(V3)
cond_V4 = np.linalg.cond(V4)

print(f"Współczynnik uwarunkowania V1 (t^j): {cond_V1:.2e}")
print(f"Współczynnik uwarunkowania V2 ((t−1900)^j): {cond_V2:.2e}")
print(f"Współczynnik uwarunkowania V3 ((t−1940)^j): {cond_V3:.2e}")
print(f"Współczynnik uwarunkowania V4 (((t−1940)/40)^j): {cond_V4:.2e}")

# Baza najlepiej uwarunkowana: ((t - 1940)/40)^(j-1)
t_base = (years - 1940) / 40
V_best = np.vander(t_base, increasing=True)

# Rozwiązanie układu: V · a = population
coeffs = np.linalg.solve(V_best, population)


# Horner do oceny wartości wielomianu na przedziale
def horner_eval(x_scaled, coeffs):
    result = np.zeros_like(x_scaled)
    for c in reversed(coeffs):
        result = result * x_scaled + c
    return result


# Przedział 1900–1990
t_full = np.arange(1900, 1991)
x_scaled = (t_full - 1940) / 40  # przeskalowanie do najlepiej uwarunkowanej bazy
poly_vals = horner_eval(x_scaled, coeffs)  # wartości wielomianu

# Styl seaborn
sns.set_theme(style="whitegrid")

# Dane do DataFrame (Seaborn lubi pandas)
df = pd.DataFrame({
    'Rok': t_full,
    'Populacja interpolowana': poly_vals
})
df_nodes = pd.DataFrame({
    'Rok': years,
    'Populacja': population
})

# Wykres
import matplotlib.pyplot as plt
import seaborn as sns

# Ustawienie tematu, palety kolorów i czcionek
sns.set_theme(style="white", palette="pastel", font="DejaVu Sans", rc={"axes.labelsize": 12, "axes.titlesize": 16})

# Utworzenie wykresu
plt.figure(figsize=(10, 6))

# Rysowanie linii (interpolacja)
sns.lineplot(data=df, x='Rok', y='Populacja interpolowana', label='Wielomian interpolacyjny', color='steelblue', linewidth=2)

# Rysowanie punktów (węzły interpolacji)
sns.scatterplot(data=df_nodes, x='Rok', y='Populacja', color='red', s=100, label='Węzły interpolacji', zorder=5, linewidth=1.5)

# Tytuł wykresu
plt.title("Interpolacja populacji USA (1900–1990)", fontsize=18, fontweight='bold')

# Etykiety osi
plt.xlabel("Rok", fontsize=14)
plt.ylabel("Populacja", fontsize=14)

# Ustawienie siatki
plt.grid(True, linestyle='--', alpha=0.6)

# Legendy
plt.legend(title="Legenda", title_fontsize=13, fontsize=11, loc='upper left')

# Dostosowanie układu wykresu
plt.tight_layout()

# Wyświetlenie wykresu
plt.show()


# Rok 1990 w przeskalowanej bazie
x_1990 = (1990 - 1940) / 40
pop_1990_est = horner_eval(np.array([x_1990]), coeffs)[0]

# Prawdziwa wartość
pop_1990_true = 248_709_873

# Obliczenie błędu względnego
rel_error = abs(pop_1990_est - pop_1990_true) / pop_1990_true

print(f"Estymowana populacja w 1990 roku: {pop_1990_est:.0f}")
print(f"Rzeczywista populacja w 1990 roku: {pop_1990_true}")
print(f"Błąd względny: {rel_error:.6f} ({rel_error * 100:.4f}%)")


def lagrange_basis(x, xi, i):
    """Oblicza i-tą funkcję bazową Lagrange’a w punktach x"""
    terms = [(x - xi[j]) / (xi[i] - xi[j]) for j in range(len(xi)) if j != i]
    return np.prod(terms, axis=0)


def lagrange_interp(x, xi, yi):
    """Oblicza wartości wielomianu Lagrange’a"""
    result = np.zeros_like(x, dtype=np.float64)  # Typ wynikowy jako float64
    for i in range(len(xi)):
        Li = np.ones_like(x, dtype=np.float64)  # Również Li musi być float64
        for j in range(len(xi)):
            if i != j:
                Li *= (x - xi[j]) / (xi[i] - xi[j])  # Używaj float64 w operacjach
        result += yi[i] * Li
    return result


# Przedział 1900–1990 (co rok)
t_full = np.arange(1900, 1991)
lagrange_vals = lagrange_interp(t_full, years, population)

# Wykres
plt.figure(figsize=(12, 6))
sns.lineplot(x=t_full, y=lagrange_vals, label="Lagrange", color="green")
sns.scatterplot(x=years, y=population, color="crimson", s=100, label="Węzły interpolacji", zorder=5)
plt.title("Wielomian interpolacyjny Lagrange’a", fontsize=16)
plt.xlabel("Rok")
plt.ylabel("Populacja")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


def newton_coeffs(xi, yi):
    """Oblicza współczynniki wielomianu Newtona za pomocą różnic skończonych."""
    n = len(xi)
    coeffs = np.copy(yi).astype(np.float64)  # Kopia wartości y (populacja) jako współczynniki
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coeffs[i] = (coeffs[i] - coeffs[i - 1]) / (xi[i] - xi[i - j])
    return coeffs


def newton_eval(x, xi, coeffs):
    """Oblicza wartości wielomianu Newtona w punktach x."""
    n = len(xi)
    result = coeffs[-1]
    for i in range(n - 2, -1, -1):
        result = coeffs[i] + (x - xi[i]) * result
    return result


# Obliczenie współczynników Newtona
coeffs_newton = newton_coeffs(years, population)

# Przedział 1900–1990 (co rok)
t_full = np.arange(1900, 1991)

# Obliczenie wartości wielomianu Newtona w tych punktach
newton_vals = newton_eval(t_full, years, coeffs_newton)

# Wykres
plt.figure(figsize=(12, 6))
sns.lineplot(x=t_full, y=newton_vals, label="Wielomian Newtona", color="purple")
sns.scatterplot(x=years, y=population, color="crimson", s=100, label="Węzły interpolacji", zorder=5)
plt.title("Wielomian interpolacyjny Newtona", fontsize=16)
plt.xlabel("Rok")
plt.ylabel("Populacja")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Można również obliczyć wartość populacji w 1990 roku na podstawie tego wielomianu
pop_1990_newton = newton_eval(np.array([1990]), years, coeffs_newton)[0]
print(f"Estymowana populacja w 1990 roku (wielomian Newtona): {pop_1990_newton:.0f}")

# Zaokrąglenie danych do najbliższego miliona
population_rounded = np.round(population / 1_000_000) * 1_000_000

# Wyznaczenie bazy najlepiej uwarunkowanej (V4)
t_base_rounded = (years - 1940) / 40
V_rounded = np.vander(t_base_rounded, increasing=True)

# Rozwiązanie układu V * a = population_rounded
coeffs_rounded = np.linalg.solve(V_rounded, population_rounded)

# Porównanie współczynników z poprzednimi
print("Współczynniki wielomianu z danych zaokrąglonych do miliona:")
print(coeffs_rounded)

# Obliczenie wartości wielomianu w 1990 roku przy pomocy schematu Hornera
x_1990 = (1990 - 1940) / 40
pop_1990_est_rounded = horner_eval(np.array([x_1990]), coeffs_rounded)[0]

# Prawdziwa wartość
pop_1990_true = 248_709_873

# Obliczenie błędu względnego
rel_error_rounded = abs(pop_1990_est_rounded - pop_1990_true) / pop_1990_true

print(f"Estymowana populacja w 1990 roku (po zaokrągleniu): {pop_1990_est_rounded:.0f}")
print(f"Rzeczywista populacja w 1990 roku: {pop_1990_true}")
print(f"Błąd względny (po zaokrągleniu): {rel_error_rounded:.6f} ({rel_error_rounded * 100:.4f}%)")
print("Poprzednie współczynnuki", coeffs)
