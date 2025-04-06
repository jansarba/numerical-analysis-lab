import math

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid")


def suma_double(x):
    suma = np.float64(0)
    for i in range(len(x)):
        suma += x[i]
    return suma

def suma_float(x):
    suma = np.float32(0)
    for i in range(len(x)):
        suma += x[i]
    return suma

def suma_kahan(x):
    suma = np.float32(0)
    err = np.float32(0)
    for i in range(len(x)):
        y = x[i] - err
        temp = suma + y
        err = (temp - suma) - y
        suma = temp
    return suma

def suma_rosnaco(x):
    suma = np.float32(0)
    for i in sorted(x):
        suma += i
    return suma

def suna_malejaco(x):
    suma = np.float32(0)
    for i in sorted(x, reverse=True):
        suma += i
    return suma

def fx(k):
    return np.random.uniform(0, 1, 10**k)

x = [fx(k) for k in range(4, 9)]
sumy = [suma_double, suma_float, suma_kahan, suma_rosnaco, suna_malejaco]

for s in sumy:
    print(s.__name__)
    wyniki = [np.abs(s(xi)-np.sum(xi)) for xi in x]
    print(wyniki) # momentami b. male roznice, wyprintujmy. ciekawy blad 0.0
    plt.plot([4, 5, 6, 7, 8], wyniki, 'o-', label=s.__name__)
    ax=plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))



plt.yscale("log")
plt.xlabel("Liczba elementów (10^k)")
plt.ylabel("Błąd bezwzględny sumowania")
plt.legend()
plt.savefig("sumy.png")
plt.show()