import numpy as np
from scipy.optimize import minimize_scalar

# Funkcja dokładna
def exact(t):
    return np.exp(-5 * t)

# Funkcja błędu
def error(h):
    if h <= 0 or h >= 0.4:
        return np.inf
    n = int(0.5 / h)
    approx = (1 - 5 * h)**n
    return abs(approx - exact(0.5))

# Minimalizacja błędu do granicy 0.001
def constrained_error(h):
    return abs((1 - 5 * h)**int(0.5 / h) - np.exp(-5 * 0.5)) - 0.001

res = minimize_scalar(error, bounds=(0.01, 0.4), method='bounded')
h_max = res.x
n_steps = int(0.5 / h_max)

print(f"Maksymalny krok h, by błąd < 0.001: {h_max:.4f}")
print(f"Liczba kroków: {n_steps}")
print(f"Błąd przy tym h: {error(h_max)}")
