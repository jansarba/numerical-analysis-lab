import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import tan

# Zadanie 1

x = 1 # stała z polecenia

def pochodna_numeryczna(x, h):
    return (tan(x+h) - tan(x))/h

def pochodna_numeryczna_centralna(x, h):
    return (tan(x+h) - tan(x-h))/(2*h)

def pochodna_analityczna(x):
    return 1 + tan(x)**2 # wzor z polecenia

def pochodna_analityczna_2(x):
    return 2 * tan(x) * (1 + tan(x)**2) # druga pochodna

def pochodna_analityczna_3(x):
    return 2 * (3 * tan(x)**4 + 4 * tan(x)**2 + 1) # trzecia pochodna

k = np.arange(17)
h = 10.0**(-k)

poch_num = np.array([pochodna_numeryczna(x, hi) for hi in h])
poch_an = pochodna_analityczna(x)
roznice = np.abs(poch_num - poch_an)

# Nowe obliczenia dla roznice_2
poch_num_2 = np.array([pochodna_numeryczna_centralna(x, hi) for hi in h])
roznice_2 = np.abs(poch_num_2 - poch_an)

sns.set(style="whitegrid")
plt.plot(h, roznice, 'o')

# Pierwszy wykres
for i, txt in enumerate(k):  # cyferki nad punktami
    if i == 0:
        plt.annotate(f'h = {txt}', (h[i], roznice[i]), textcoords="offset points", xytext=(5, 10), ha='center')
    else:
        plt.annotate(txt, (h[i], roznice[i]), textcoords="offset points", xytext=(5, 10), ha='center')

plt.yscale("log")
plt.xscale("log")
plt.xlabel("k")
plt.ylabel("Różnica")
plt.title("Różnica między pochodną analityczną a numeryczną")

eps = np.finfo(float).eps
M_1 = pochodna_analityczna_2(x)
h_min_1 = abs(2 * np.sqrt(eps / M_1))

plt.axvline(x=h_min_1, color='r', linestyle='--')
plt.axhline(y=pochodna_numeryczna(x, h_min_1) - poch_an, color='r', linestyle='--')

plt.savefig("pochodne1.png")
plt.show()

# Drugi wykres
for i, txt in enumerate(k):  # cyferki nad punktami na drugim wykresie
    if i == 0:
        plt.annotate(f'h = {txt}', (h[i], roznice_2[i]), textcoords="offset points", xytext=(5, 10), ha='center')
    else:
        plt.annotate(txt, (h[i], roznice_2[i]), textcoords="offset points", xytext=(5, 10), ha='center')

plt.plot(h, roznice_2, 'o')
plt.yscale("log")
plt.xscale("log")
plt.xlabel("k")
plt.ylabel("Różnica")
plt.title("Różnica między pochodną analityczną a numeryczną 2")

eps = np.finfo(float).eps
M_2 = pochodna_analityczna_3(x)
h_min_2 = abs(np.cbrt(3 * eps / M_2))

plt.axvline(x=h_min_2, color='r', linestyle='--')
plt.axhline(y=pochodna_numeryczna_centralna(x, h_min_2) - poch_an, color='r', linestyle='--')

print(f"h_min_1 = {h_min_1}")
print(min(roznice))
print(pochodna_numeryczna(x, h_min_1) - poch_an)
print(f"h_min_2 = {h_min_2}")
print(min(roznice_2))
print(pochodna_numeryczna_centralna(x, h_min_2) - poch_an)

# plt.savefig("pochodne2.png")
plt.show()

# Wniosek - metoda centralna jest dokladniejsza zarowno teoretycznie jak i praktycznie