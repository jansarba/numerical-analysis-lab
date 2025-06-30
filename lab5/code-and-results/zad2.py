import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev

x = np.linspace(0, 2, 1000)
y = np.sqrt(x)
cheb_poly = chebyshev.Chebyshev.fit(x, y, 2, domain=[0, 2])
y_approx = cheb_poly(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='f(x) = √x', linewidth=3)
plt.plot(x, y_approx, '--', label='Aproksymacja Czebyszewa (m=2)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Aproksymacja funkcji √x wielomianem Czebyszewa stopnia 2')
plt.legend()
plt.grid(True)
plt.savefig('zadanie2_chebyshev.png', dpi=300, bbox_inches='tight')
plt.close()

print("Współczynniki wielomianu Czebyszewa:", cheb_poly.coef.round(3))