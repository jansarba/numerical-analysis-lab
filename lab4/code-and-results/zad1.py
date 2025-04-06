import numpy as np
import matplotlib.pyplot as plt

# Funkcja do obliczania średniej geometrycznej odległości do pozostałych punktów
def geometric_mean_distance(points):
    distances = []
    for i, x in enumerate(points):
        dists = np.abs(points[np.arange(len(points)) != i] - x)
        geom_mean = np.exp(np.mean(np.log(dists)))
        distances.append(geom_mean)
    return distances

# Generowanie punktow Czebyszewa
def chebyshev_points(n):
    return np.cos((2 * np.arange(n) + 1) / (2 * n) * np.pi)

# Generowanie punktow Legendre'a
def legendre_points(n):
    return np.polynomial.legendre.legroots([0] * n + [1])

# Generowanie punktow rownomiernych
def uniform_points(n):
    return np.linspace(-1, 1, n)

# Funkcja rysujaca wykres
def plot_geometric_mean(points, label):
    y = geometric_mean_distance(points)
    plt.scatter(points, y, label=label)

# Tworzenie wykresow dla n = 10, 20, 50
plt.figure(figsize=(12, 8))

for n in [10, 20, 50]:
    plot_geometric_mean(chebyshev_points(n), f'Czebyszew n={n}')
    plot_geometric_mean(legendre_points(n), f'Legendre n={n}')
    plot_geometric_mean(uniform_points(n), f'Rownomierne n={n}')

plt.xlabel('Punkty')
plt.ylabel('Srednia geometryczna odleglosci')
plt.legend()
plt.title('Srednia geometryczna odleglosci do pozostalych punktow')
plt.grid(True)

# Zapis wykresu
plt.savefig("geometric_mean_distance.png", dpi=300)
plt.close()
