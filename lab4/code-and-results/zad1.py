import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator, CubicSpline


def f1(x):
    return 1 / (1 + 25 * x ** 2)


def f2(x):
    return np.exp(np.cos(x))


def chebyshev_nodes(n, a, b):
    theta = np.pi * (2 * np.arange(1, n + 1) - 1) / (2 * n)
    return np.cos(theta) * (b - a) / 2 + (a + b) / 2


def part_a():
    f = f1
    a, b = -1, 1
    n = 12
    # Generowanie gęstych siatek do ewaluacji
    x_uniform_dense = np.linspace(a, b, 10 * n)
    x_cheb_dense = chebyshev_nodes(10 * n, a, b)
    x_cheb_dense_sorted = np.sort(x_cheb_dense)
    x_fine = np.linspace(a, b, 1000)
    y_fine = f(x_fine)

    # Węzły interpolacji
    x_uniform = np.linspace(a, b, n)
    y_uniform = f(x_uniform)
    x_cheb = chebyshev_nodes(n, a, b)
    y_cheb = f(x_cheb)

    # Interpolacje
    lagrange_uni = BarycentricInterpolator(x_uniform, y_uniform)
    y_lagrange_uni = lagrange_uni(x_uniform_dense)

    spline = CubicSpline(x_uniform, y_uniform)
    y_spline = spline(x_uniform_dense)

    lagrange_cheb = BarycentricInterpolator(x_cheb, y_cheb)
    y_lagrange_cheb = lagrange_cheb(x_cheb_dense_sorted)

    # Wykres
    plt.figure(figsize=(10, 5))
    plt.plot(x_fine, y_fine, label="Prawdziwa funkcja", linewidth=2)
    plt.plot(x_uniform_dense, y_lagrange_uni, '--', label="Lagrange równoodległe")
    plt.plot(x_uniform_dense, y_spline, '-.', label="Cubic Spline")
    plt.plot(x_cheb_dense_sorted, y_lagrange_cheb, ':', label="Lagrange Czebyszewa")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"Interpolacja $f_1(x)$ dla n={n}")
    plt.legend()
    plt.savefig("f1_part_a.png", dpi=300)
    plt.close()


def part_b():
    functions = [(f1, -1, 1, "f1"), (f2, 0, 2 * np.pi, "f2")]
    n_values = range(4, 51)
    num_points = 500

    for f, a, b, name in functions:
        errors_uni, errors_spl, errors_cheb = [], [], []

        for n in n_values:
            np.random.seed(42)
            x_rand = np.random.uniform(a, b, num_points)
            x_rand.sort()
            y_true = f(x_rand)

            # Węzły równoodległe
            x_uni = np.linspace(a, b, n)
            y_uni = f(x_uni)
            interpolator = BarycentricInterpolator(x_uni, y_uni)
            y_interp = interpolator(x_rand)
            errors_uni.append(np.max(np.abs(y_interp - y_true)))

            # Funkcja sklejana
            spline = CubicSpline(x_uni, y_uni)
            y_spl = spline(x_rand)
            errors_spl.append(np.max(np.abs(y_spl - y_true)))

            # Węzły Czebyszewa
            x_cheb = chebyshev_nodes(n, a, b)
            y_cheb = f(x_cheb)
            interpolator = BarycentricInterpolator(x_cheb, y_cheb)
            y_interp = interpolator(x_rand)
            errors_cheb.append(np.max(np.abs(y_interp - y_true)))

        # Wykres błędów
        plt.figure(figsize=(10, 5))
        plt.plot(n_values, errors_uni, 'o-', label="Lagrange równoodległe")
        plt.plot(n_values, errors_spl, 's-', label="Cubic Spline")
        plt.plot(n_values, errors_cheb, '^-', label="Lagrange Czebyszewa")
        plt.xlabel("Liczba węzłów interpolacji, n")
        plt.ylabel("Błąd maksymalny")
        plt.yscale("log")
        plt.title(f"Błąd interpolacji dla {name}(x)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{name}_part_b.png", dpi=300)
        plt.close()


# Wykonanie części (a) i (b)
part_a()
part_b()