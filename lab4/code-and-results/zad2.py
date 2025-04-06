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


def interpolate_and_plot(f, a, b, n_values, title, filename_prefix):
    x_dense = np.linspace(a, b, 1000)
    y_true = f(x_dense)

    for n in n_values:
        plt.figure(figsize=(10, 5))
        plt.plot(x_dense, y_true, label="Prawdziwa funkcja", linewidth=2)

        x_uniform = np.linspace(a, b, n)
        y_uniform = f(x_uniform)

        x_chebyshev = chebyshev_nodes(n, a, b)
        y_chebyshev = f(x_chebyshev)

        # Lagrange (uniform)
        lagrange_uniform = BarycentricInterpolator(x_uniform, y_uniform)
        y_lagrange_uniform = lagrange_uniform(x_dense)
        plt.plot(x_dense, y_lagrange_uniform, linestyle="--", label="Lagrange uniform")

        # Cubic spline
        spline = CubicSpline(x_uniform, y_uniform)
        y_spline = (x_dense)
        plt.plot(x_dense, y_spline, linestyle="-.", label="Cubic Spline")

        # Lagrange (Czebyszew)
        lagrange_chebyshev = BarycentricInterpolator(x_chebyshev, y_chebyshev)
        y_lagrange_chebyshev = lagrange_chebyshev(x_dense)
        plt.plot(x_dense, y_lagrange_chebyshev, linestyle=":", label="Lagrange Chebyshev")

        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(f"Interpolacja {title} dla n={n}")
        plt.legend()

        filename = f"{filename_prefix}_n{n}.png"
        plt.savefig(filename, dpi=300)
        plt.close()


n_values = range(4, 51, 2)
interpolate_and_plot(f1, -1, 1, n_values, "$f_1(x)$", "f1_interpolacja")
interpolate_and_plot(f2, 0, 2 * np.pi, n_values, "$f_2(x)$", "f2_interpolacja")
