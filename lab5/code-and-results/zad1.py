import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8')

years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980])
population = np.array([76212168, 92228496, 106021537, 123202624, 132164569,
                       151325798, 179323175, 203302031, 226542199])
target_year = 1990
true_value = 248709873
n = len(years)

# 1(a)
plt.figure(figsize=(12, 6))
plt.scatter(years, population / 1e6, color='red', label='Dane historyczne', zorder=5)

errors, predictions = [], []
x_plot = np.linspace(1900, 1990, 100)

for m in range(7):
    coeffs = np.polyfit(years, population, m)
    poly = np.poly1d(coeffs)
    pred = poly(target_year)
    predictions.append(pred)
    errors.append(abs(pred - true_value))

    y_plot = poly(x_plot)
    plt.plot(x_plot, y_plot / 1e6, label=f'm={m}', alpha=0.7)

plt.scatter([target_year], [true_value / 1e6], color='green', marker='*', s=200,
            label='Rzeczywista wartość 1990', zorder=5)
plt.xlabel('Rok')
plt.ylabel('Populacja (mln)')
plt.title('Aproksymacja populacji USA różnymi stopniami wielomianu')
plt.legend()
plt.grid(True)
plt.savefig('zadanie1a_aprox.png', dpi=300, bbox_inches='tight')
plt.close()

for m, error in enumerate(errors):
    print(f"Stopień m = {m}: Błąd bezwzględny = {error:,.0f}")

# plt.figure(figsize=(8, 5))
# plt.bar(range(7), errors, color='skyblue')
# plt.axhline(y=errors[min_error_idx], color='red', linestyle='--',
#             label=f'Najmniejszy błąd (m={min_error_idx})')
# plt.xlabel('Stopień wielomianu (m)')
# plt.ylabel('Błąd względny (%)')
# plt.title('Błąd względny ekstrapolacji dla roku 1990')
# plt.legend()
# plt.grid(True)
# plt.savefig('zadanie1a_error.png', dpi=300, bbox_inches='tight')
# plt.close()
#
# # 1(b) - AICc
# aicc_list = []
# for m in range(7):
#     k = m + 1
#     coeffs = np.polyfit(years, population, m)
#     poly = np.poly1d(coeffs)
#     y_pred = poly(years)
#     sse = np.sum((population - y_pred) ** 2)
#     aic = 2 * k + n * np.log(sse / n)
#     aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if (n - k - 1) > 0 else aic
#     aicc_list.append(aicc)
#
# optimal_m = np.argmin(aicc_list)
# plt.figure(figsize=(8, 5))
# plt.plot(range(7), aicc_list, 'o-', color='purple')
# plt.axvline(x=optimal_m, color='red', linestyle='--',
#             label=f'Optymalne m={optimal_m}')
# plt.xlabel('Stopień wielomianu (m)')
# plt.ylabel('Wartość AICc')
# plt.title('Kryterium AICc dla różnych stopni wielomianu')
# plt.legend()
# plt.grid(True)
# plt.savefig('zadanie1b_aicc.png', dpi=300, bbox_inches='tight')
# plt.close()
#
# plt.figure(figsize=(12, 6))
#
# plt.xlim(1970, 1995)
# plt.ylim(200, 280)
#
# for m in range(7):
#     coeffs = np.polyfit(years, population, m)
#     poly = np.poly1d(coeffs)
#     y_plot = poly(x_plot)
#
#     if m != min_error_idx:
#         plt.plot(x_plot, y_plot / 1e6, label=f'm={m}', alpha=0.7, linewidth=2, linestyle='--')
#     else:
#         plt.plot(x_plot, y_plot / 1e6, label=f'm={m}', alpha=0.7, linewidth=2)
#
#
# mask = years >= 1970
# plt.scatter(years[mask], population[mask]/1e6, color='red', s=80,
#            label='Dane historyczne od 1960', zorder=5)
# plt.scatter([target_year], [true_value/1e6], color='green', marker='*', s=300,
#            label='Rzeczywista wartość 1990', zorder=6)
#
# plt.xlabel('Rok', fontsize=12)
# plt.ylabel('Populacja (mln)', fontsize=12)
# plt.title('Ekstrapolacje populacji USA po 1970 roku', fontsize=14, pad=20)
# plt.legend(loc='upper left', fontsize=10)
# plt.grid(True, alpha=0.4)
#
# plt.annotate(f'Najlepsza ekstrapolacja (m={min_error_idx})',
#             xy=(1990, predictions[min_error_idx]/1e6),
#             xytext=(1990, predictions[min_error_idx]/1e6 - 15),
#             arrowprops=dict(arrowstyle="->", color='black'),
#             fontsize=10)
#
# plt.savefig('zadanie1c_zoom.png', dpi=300, bbox_inches='tight')
# plt.close()