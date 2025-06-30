import numpy as np
import matplotlib.pyplot as plt

N_POINTS = 20
N_OBSTACLES = 50
LAMBDA_1 = 1.0
LAMBDA_2 = 1.0
EPSILON = 1e-13
ITERATIONS = 400
RANDOM_INITIALIZATIONS = 5

x_start = np.array([0.0, 0.0])
x_end = np.array([20.0, 20.0])

np.random.seed(42)
obstacles = np.random.uniform(0, 20, size=(N_OBSTACLES, 2))



def cost_function(path, obstacles, lambda1, lambda2, epsilon):
    cost_obs = np.sum(1.0 / (epsilon + np.sum((path[:, np.newaxis, :] - obstacles) ** 2, axis=2)))
    cost_len = np.sum(np.sum((path[1:, :] - path[:-1, :]) ** 2, axis=1))
    return lambda1 * cost_obs + lambda2 * cost_len


def calculate_gradient(path, obstacles, lambda1, lambda2, epsilon):
    grad = np.zeros_like(path)
    for i in range(1, path.shape[0] - 1):
        diffs_obs = path[i, :] - obstacles
        denom = (epsilon + np.sum(diffs_obs ** 2, axis=1)) ** 2
        grad_obs = -2 * lambda1 * np.sum(diffs_obs / denom[:, np.newaxis], axis=0)
        grad_len = 2 * lambda2 * (2 * path[i, :] - path[i - 1, :] - path[i + 1, :])
        grad[i, :] = grad_obs + grad_len
    return grad


def golden_section_search(f, a, b, tol=1e-5):
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2


print("--- Zadanie 2: Planowanie Ścieżki Robota ---")

final_paths = []
cost_histories = []

for i in range(RANDOM_INITIALIZATIONS):
    print(f"--- Inicjalizacja {i + 1}/{RANDOM_INITIALIZATIONS} ---")

    path = np.linspace(x_start, x_end, N_POINTS + 1)
    path[1:-1, :] += np.random.randn(N_POINTS - 1, 2) * 2.5
    history = []

    for j in range(ITERATIONS):
        grad = calculate_gradient(path, obstacles, LAMBDA_1, LAMBDA_2, EPSILON)
        line_search_func = lambda alpha: cost_function(path - alpha * grad, obstacles, LAMBDA_1, LAMBDA_2, EPSILON)
        alpha = golden_section_search(line_search_func, 0, 0.1)
        path = path - alpha * grad
        history.append(cost_function(path, obstacles, LAMBDA_1, LAMBDA_2, EPSILON))

    final_paths.append(path)
    cost_histories.append(history)
    print(f"Zakończono z kosztem: {history[-1]:.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

best_run_idx = np.argmin([h[-1] for h in cost_histories])
path_to_plot = final_paths[best_run_idx]
ax1.plot(path_to_plot[:, 0], path_to_plot[:, 1], 'g-o', markersize=4,
         label=f'Najlepsza ścieżka (run {best_run_idx + 1})')
ax1.scatter(obstacles[:, 0], obstacles[:, 1], c='red', marker='x', label='Przeszkody')
ax1.scatter(x_start[0], x_start[1], c='blue', s=100, label='Start', zorder=5)
ax1.scatter(x_end[0], x_end[1], c='purple', s=100, label='Koniec', zorder=5)
ax1.set_title('Optymalna znaleziona ścieżka robota')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')

for i, history in enumerate(cost_histories):
    ax2.plot(history, label=f'Inicjalizacja {i + 1}')
ax2.set_title('Wartość funkcji kosztu F w zależności od iteracji')
ax2.set_xlabel('Iteracja')
ax2.set_ylabel('Koszt F(X)')
ax2.grid(True)
ax2.set_yscale('log')
ax2.legend()

output_filename = 'wyniki_optymalizacji_sciezki.png'
plt.tight_layout()
plt.savefig(output_filename)
print(f"\nWykresy zostały zapisane do pliku: {output_filename}")

plt.show()