import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize

# Parametry początkowe (oryginalne, używane w częściach a, b, c)
alpha1_orig = 1.0  # współczynnik przyrostu ofiar
beta1_orig = 0.1   # współczynnik intensywności kontaktów (wpływ drapieżników na ofiary)
alpha2_orig = 0.5  # współczynnik ubywania drapieżców
beta2_orig = 0.02  # współczynnik wpływu ofiar na przyrost drapieżców

# Warunki początkowe
x0_sim = 20.0  # początkowa gęstość ofiar
y0_sim = 20.0  # początkowa gęstość drapieżców
Y0_sim = np.array([x0_sim, y0_sim])

t_start_sim = 0
t_end_sim = 80
h_sim = 0.02

# Nazwy plików dla wykresów
plot_filename_populations_prey = "lotka_volterra_populacje_ofiary.png"
plot_filename_populations_predator = "lotka_volterra_populacje_drapiezniki.png"
plot_filename_phase_portrait = "lotka_volterra_portret_fazowy.png"
plot_filename_invariant = "lotka_volterra_niezmiennik_H.png"
plot_filename_fitting_hares = "lotka_volterra_dopasowanie_zajace_poprawione.png"
plot_filename_fitting_lynx = "lotka_volterra_dopasowanie_rysie_poprawione.png"
plot_filename_fitting_errors = "lotka_volterra_dopasowanie_bledy_poprawione.png"


# --- Funkcja prawych stron układu ODE ---
def lotka_volterra_rhs(Y, t, alpha1, beta1, alpha2, beta2):
    x, y = Y
    dx_dt = x * (alpha1 - beta1 * y)
    dy_dt = y * (-alpha2 + beta2 * x)
    return np.array([dx_dt, dy_dt])


# --- Metody Numeryczne ---
def explicit_euler_solver(rhs_func, Y0, t_span, h, params):
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    Y_values = np.zeros((len(t_values), len(Y0)))
    Y_values[0, :] = Y0
    for k in range(len(t_values) - 1):
        Y_k = Y_values[k, :]
        t_k = t_values[k]
        Y_values[k + 1, :] = Y_k + h * rhs_func(Y_k, t_k, *params)
    return t_values, Y_values

def implicit_euler_solver(rhs_func, Y0, t_span, h, params):
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    Y_values = np.zeros((len(t_values), len(Y0)))
    Y_values[0, :] = Y0
    for k in range(len(t_values) - 1):
        Y_k = Y_values[k, :]
        t_kp1 = t_values[k + 1]
        def implicit_eq_solver(Y_kp1_guess):
            return Y_kp1_guess - Y_k - h * rhs_func(Y_kp1_guess, t_kp1, *params)
        Y_kp1_solution, _, ier, _ = fsolve(implicit_eq_solver, Y_k, full_output=True)
        if ier != 1:
            Y_values[k+1, :] = Y_k
        else:
            Y_values[k + 1, :] = Y_kp1_solution
    return t_values, Y_values

def semi_implicit_euler_1_solver(Y0, t_span, h, params):
    alpha1, beta1, alpha2, beta2 = params
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    Y_values = np.zeros((len(t_values), len(Y0)))
    Y_values[0, :] = Y0
    x_n, y_n = Y0[0], Y0[1]
    for k in range(len(t_values) - 1):
        denominator_y = 1 + h * alpha2 - h * beta2 * x_n
        y_np1 = y_n / denominator_y if abs(denominator_y) > 1e-9 else y_n
        x_np1 = x_n + h * x_n * (alpha1 - beta1 * y_np1)
        Y_values[k + 1, :] = [x_np1, y_np1]
        x_n, y_n = x_np1, y_np1
    return t_values, Y_values

def semi_implicit_euler_2_solver(Y0, t_span, h, params):
    alpha1, beta1, alpha2, beta2 = params
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    Y_values = np.zeros((len(t_values), len(Y0)))
    Y_values[0, :] = Y0
    x_n, y_n = Y0[0], Y0[1]
    for k in range(len(t_values) - 1):
        denominator_x = 1 - h * alpha1 + h * beta1 * y_n
        x_np1 = x_n / denominator_x if abs(denominator_x) > 1e-9 else x_n
        y_np1 = y_n + h * y_n * (-alpha2 + beta2 * x_np1)
        Y_values[k + 1, :] = [x_np1, y_np1]
        x_n, y_n = x_np1, y_np1
    return t_values, Y_values

def rk4_solver(rhs_func, Y0, t_span, h, params):
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    Y_values = np.zeros((len(t_values), len(Y0)))
    Y_values[0, :] = Y0
    for k in range(len(t_values) - 1):
        Y_k = Y_values[k, :]
        t_k = t_values[k]
        k1 = rhs_func(Y_k, t_k, *params)
        k2 = rhs_func(Y_k + 0.5 * h * k1, t_k + 0.5 * h, *params)
        k3 = rhs_func(Y_k + 0.5 * h * k2, t_k + 0.5 * h, *params)
        k4 = rhs_func(Y_k + h * k3, t_k + h, *params)
        Y_values[k + 1, :] = Y_k + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t_values, Y_values


print("--- Część (a): Symulacje i Wykresy ---")
params_sim = (alpha1_orig, beta1_orig, alpha2_orig, beta2_orig)
t_span_sim_tuple = (t_start_sim, t_end_sim)

t_exp_eul, Y_exp_eul = explicit_euler_solver(lotka_volterra_rhs, Y0_sim, t_span_sim_tuple, h_sim, params_sim)
t_imp_eul, Y_imp_eul = implicit_euler_solver(lotka_volterra_rhs, Y0_sim, t_span_sim_tuple, h_sim, params_sim)
t_semi1_eul, Y_semi1_eul = semi_implicit_euler_1_solver(Y0_sim, t_span_sim_tuple, h_sim, params_sim)
t_semi2_eul, Y_semi2_eul = semi_implicit_euler_2_solver(Y0_sim, t_span_sim_tuple, h_sim, params_sim)
t_rk4, Y_rk4 = rk4_solver(lotka_volterra_rhs, Y0_sim, t_span_sim_tuple, h_sim, params_sim)

plt.figure(figsize=(12, 6))
plt.plot(t_exp_eul, Y_exp_eul[:, 0], label='Jawny Euler', linestyle='-')
plt.plot(t_imp_eul, Y_imp_eul[:, 0], label='Niejawny Euler', linestyle='-', alpha=0.7)
plt.plot(t_semi1_eul, Y_semi1_eul[:, 0], label='Półjawny 1', linestyle='-', alpha=0.6)
plt.plot(t_semi2_eul, Y_semi2_eul[:, 0], label='Półjawny 2', linestyle='-', alpha=0.5)
plt.plot(t_rk4, Y_rk4[:, 0], label='RK4', color='black', linewidth=1.5, linestyle='-')
plt.xlabel('Czas'); plt.ylabel('Gęstość populacji ofiar (zajęcy)')
plt.title(f'Model Lotki-Volterry: Dynamika populacji ofiar (h={h_sim})')
plt.legend(fontsize='small', loc='upper right'); plt.grid(True); plt.ylim(bottom=0)
plt.tight_layout(); plt.savefig(plot_filename_populations_prey); plt.show()
print(f"Wykres dynamiki populacji ofiar zapisany jako: {plot_filename_populations_prey}")

plt.figure(figsize=(12, 6))
plt.plot(t_exp_eul, Y_exp_eul[:, 1], label='Jawny Euler', linestyle='--')
plt.plot(t_imp_eul, Y_imp_eul[:, 1], label='Niejawny Euler', linestyle='--', alpha=0.7)
plt.plot(t_semi1_eul, Y_semi1_eul[:, 1], label='Półjawny 1', linestyle='--', alpha=0.6)
plt.plot(t_semi2_eul, Y_semi2_eul[:, 1], label='Półjawny 2', linestyle='--', alpha=0.5)
plt.plot(t_rk4, Y_rk4[:, 1], label='RK4', color='dimgray', linewidth=1.5, linestyle='--')
plt.xlabel('Czas'); plt.ylabel('Gęstość populacji drapieżników (rysi)')
plt.title(f'Model Lotki-Volterry: Dynamika populacji drapieżników (h={h_sim})')
plt.legend(fontsize='small', loc='upper right'); plt.grid(True); plt.ylim(bottom=0)
plt.tight_layout(); plt.savefig(plot_filename_populations_predator); plt.show()
print(f"Wykres dynamiki populacji drapieżników zapisany jako: {plot_filename_populations_predator}")



plt.figure(figsize=(8, 7))
plt.plot(Y_exp_eul[:, 0], Y_exp_eul[:, 1], label='Jawny Euler')
plt.plot(Y_imp_eul[:, 0], Y_imp_eul[:, 1], label='Niejawny Euler', alpha=0.7)
plt.plot(Y_semi1_eul[:, 0], Y_semi1_eul[:, 1], label='Półjawny 1', alpha=0.6)
plt.plot(Y_semi2_eul[:, 0], Y_semi2_eul[:, 1], label='Półjawny 2', alpha=0.5)
plt.plot(Y_rk4[:, 0], Y_rk4[:, 1], label='RK4', color='black', linewidth=1.5)
plt.plot(Y0_sim[0], Y0_sim[1], 'ro', label='Punkt startowy')
plt.xlabel('Populacja zajęcy (x)'); plt.ylabel('Populacja rysi (y)')
plt.title(f'Portret fazowy układu Lotki-Volterry (h={h_sim})')
plt.legend(fontsize='small'); plt.grid(True); plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5); plt.axis('equal'); plt.tight_layout()
plt.savefig(plot_filename_phase_portrait); plt.show()


print("\n--- Część (b): Punkty Stacjonarne ---")
sp1_x, sp1_y = 0.0, 0.0
print(f"Trywialny punkt stacjonarny (wymarcie): (x, y) = ({sp1_x}, {sp1_y})")
if beta1_orig != 0 and beta2_orig != 0:
    sp2_x = alpha2_orig / beta2_orig; sp2_y = alpha1_orig / beta1_orig
    print(f"Nietrywialny punkt stacjonarny (współistnienie): (x, y) = ({sp2_x:.2f}, {sp2_y:.2f})")
else:
    print("Nie można obliczyć nietrywialnego punktu stacjonarnego.")

print("\n--- Część (c): Niezmiennik H(x,y) ---")
def invariant_H(Y_pair, alpha1, beta1, alpha2, beta2):
    x, y = Y_pair; epsilon = 1e-12
    safe_x = np.maximum(x, epsilon); safe_y = np.maximum(y, epsilon)
    return beta2 * safe_x + beta1 * safe_y - alpha2 * np.log(safe_x) - alpha1 * np.log(safe_y)

H_exp_eul = np.array([invariant_H(Y_k, *params_sim) for Y_k in Y_exp_eul])
H_imp_eul = np.array([invariant_H(Y_k, *params_sim) for Y_k in Y_imp_eul])
H_semi1_eul = np.array([invariant_H(Y_k, *params_sim) for Y_k in Y_semi1_eul])
H_semi2_eul = np.array([invariant_H(Y_k, *params_sim) for Y_k in Y_semi2_eul])
H_rk4 = np.array([invariant_H(Y_k, *params_sim) for Y_k in Y_rk4])

plt.figure(figsize=(10, 6))
plt.plot(t_exp_eul, H_exp_eul, label='H(x,y) - Jawny Euler')
plt.plot(t_imp_eul, H_imp_eul, label='H(x,y) - Niejawny Euler', alpha=0.7)
plt.plot(t_semi1_eul, H_semi1_eul, label='H(x,y) - Półjawny 1', alpha=0.6)
plt.plot(t_semi2_eul, H_semi2_eul, label='H(x,y) - Półjawny 2', alpha=0.5)
plt.plot(t_rk4, H_rk4, label='H(x,y) - RK4', color='black', linewidth=1.5)
plt.xlabel('Czas'); plt.ylabel('Wartość niezmiennika H(x,y)')
plt.title(f'Zachowanie niezmiennika H dla różnych metod (h={h_sim})')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(plot_filename_invariant); plt.show()


print("\n--- Część (d): Estymacja Parametrów na podstawie danych LynxHare.txt (WERSJA POPRAWIONA) ---")

try:
    data_lh = np.loadtxt('LynxHare.txt')
    time_data_lh = data_lh[:, 0]
    hares_data_lh = data_lh[:, 1]
    lynx_data_lh = data_lh[:, 2]
    print(f"Załadowano dane z LynxHare.txt: {len(hares_data_lh)} punktów danych (lat).")
except FileNotFoundError:
    print("BŁĄD KRYTYCZNY: Plik LynxHare.txt nie został znaleziony. Część (d) zostanie pominięta.")
    exit()
except Exception as e:
    print(f"BŁĄD KRYTYCZNY: Wystąpił problem podczas ładowania pliku LynxHare.txt: {e}. Część (d) pominięta.")
    exit()

h_data_step = 1.0 # Krok danych to 1 rok
num_data_points_lh = len(hares_data_lh)
x0_data_lh = hares_data_lh[0]
y0_data_lh = lynx_data_lh[0]
Y0_data_lh = np.array([x0_data_lh, y0_data_lh])
solver_for_fitting = rk4_solver
h_solver_internal_default = 0.02

def simulate_lv_for_fitting(params_theta_fit, Y0_fit, num_data_points_to_match,
                            h_data_interval, solver_func, h_solver_internal_step):
    alpha1_fit, alpha2_fit, beta1_fit, beta2_fit = params_theta_fit
    internal_params_fit = (alpha1_fit, beta1_fit, alpha2_fit, beta2_fit)
    if any(p < 0 for p in params_theta_fit):
        return np.ones((num_data_points_to_match, 2)) * 1e12
    t_total_simulation_time = (num_data_points_to_match - 1) * h_data_interval
    t_span_solver = (0, t_total_simulation_time)
    try:
        t_sim_fine, Y_sim_fine = solver_func(lotka_volterra_rhs, Y0_fit, t_span_solver,
                                             h_solver_internal_step, internal_params_fit)
        if np.any(np.isnan(Y_sim_fine)) or np.any(np.isinf(Y_sim_fine)):
            return np.ones((num_data_points_to_match, 2)) * 1e12
        Y_sim_fine = np.maximum(Y_sim_fine, 1e-9)
        indices_to_sample = np.round(np.linspace(0, len(t_sim_fine) - 1, num_data_points_to_match)).astype(int)
        Y_simulated_at_data_points = Y_sim_fine[indices_to_sample, :]
        if Y_simulated_at_data_points.shape[0] != num_data_points_to_match:
            return np.ones((num_data_points_to_match, 2)) * 1e11
    except (OverflowError, ValueError):
        return np.ones((num_data_points_to_match, 2)) * 1e12
    return Y_simulated_at_data_points

def cost_function_rss(params_theta_cost, Y0_cost, true_hares_data, true_lynx_data,
                      h_data_interval_cost, solver_func_cost, h_solver_internal_step_cost):
    num_points_cost = len(true_hares_data)
    Y_simulated_cost = simulate_lv_for_fitting(params_theta_cost, Y0_cost, num_points_cost,
                                               h_data_interval_cost, solver_func_cost,
                                               h_solver_internal_step_cost)
    sim_hares = Y_simulated_cost[:, 0]; sim_lynx = Y_simulated_cost[:, 1]
    cost = np.sum((true_lynx_data - sim_lynx)**2) + np.sum((true_hares_data - sim_hares)**2)
    if any(p < 0 for p in params_theta_cost): cost += 1e12 * sum(abs(p) for p in params_theta_cost if p < 0)
    return cost

def cost_function_poisson_like(params_theta_cost, Y0_cost, true_hares_data, true_lynx_data,
                               h_data_interval_cost, solver_func_cost, h_solver_internal_step_cost):
    num_points_cost = len(true_hares_data)
    Y_simulated_cost = simulate_lv_for_fitting(params_theta_cost, Y0_cost, num_points_cost,
                                               h_data_interval_cost, solver_func_cost,
                                               h_solver_internal_step_cost)
    sim_hares_hat = Y_simulated_cost[:, 0]; sim_lynx_hat = Y_simulated_cost[:, 1]
    term1 = np.sum(true_lynx_data * np.log(sim_lynx_hat))
    term2 = np.sum(true_hares_data * np.log(sim_hares_hat))
    term3 = np.sum(sim_lynx_hat); term4 = np.sum(sim_hares_hat)
    cost = -term1 - term2 + term3 + term4
    if np.isnan(cost) or np.isinf(cost): cost = 1e18
    if any(p < 0 for p in params_theta_cost): cost += 1e12 * sum(abs(p) for p in params_theta_cost if p < 0)
    return cost

initial_guess_theta_revised = np.array([alpha1_orig, alpha2_orig, beta1_orig, beta2_orig])
optimization_options_nm_revised = {'maxiter': 20000, 'maxfev': 40000, 'adaptive': True,
                                   'fatol': 1e-9, 'xatol': 1e-9}
args_for_cost_func = (Y0_data_lh, hares_data_lh, lynx_data_lh, h_data_step,
                      solver_for_fitting, h_solver_internal_default)

print(f"\nOptymalizacja z RSS (Nelder-Mead) - POPRAWIONA...")
print(f"Początkowe theta: {initial_guess_theta_revised}, h_solver_internal: {h_solver_internal_default}")
result_rss = minimize(cost_function_rss, initial_guess_theta_revised, args=args_for_cost_func,
                      method='Nelder-Mead', options=optimization_options_nm_revised)
if result_rss.success:
    optimal_params_rss = result_rss.x
    print(f"Sukces RSS. Theta: {optimal_params_rss}, Koszt: {result_rss.fun:.4e}")
else:
    print(f"RSS nie powiodło się: {result_rss.message}"); optimal_params_rss = initial_guess_theta_revised
Y_sim_rss = simulate_lv_for_fitting(optimal_params_rss, Y0_data_lh, num_data_points_lh,
                                    h_data_step, solver_for_fitting, h_solver_internal_default)

print(f"\nOptymalizacja z Poisson-like (Nelder-Mead) - POPRAWIONA...")
print(f"Początkowe theta: {initial_guess_theta_revised}, h_solver_internal: {h_solver_internal_default}")
result_poisson = minimize(cost_function_poisson_like, initial_guess_theta_revised, args=args_for_cost_func,
                          method='Nelder-Mead', options=optimization_options_nm_revised)
if result_poisson.success:
    optimal_params_poisson = result_poisson.x
    print(f"Sukces Poisson. Theta: {optimal_params_poisson}, Koszt: {result_poisson.fun:.4e}")
else:
    print(f"Poisson nie powiodło się: {result_poisson.message}"); optimal_params_poisson = initial_guess_theta_revised
Y_sim_poisson = simulate_lv_for_fitting(optimal_params_poisson, Y0_data_lh, num_data_points_lh,
                                        h_data_step, solver_for_fitting, h_solver_internal_default)

# Wykresy dopasowania
plot_time_axis = time_data_lh if 'time_data_lh' in locals() else np.arange(num_data_points_lh)
plt.figure(figsize=(12, 6))
plt.plot(plot_time_axis, hares_data_lh, 'o-', label='Dane (Zając)', color='blue', markersize=4)
plt.plot(plot_time_axis, Y_sim_rss[:, 0], '--', label='Sym. (Zając) - RSS', color='deepskyblue', lw=1.5)
plt.plot(plot_time_axis, Y_sim_poisson[:, 0], ':', label='Sym. (Zając) - Poisson', color='mediumspringgreen', lw=1.5)
plt.xlabel('Rok'); plt.ylabel('Populacja zajęcy (x)')
plt.legend(); plt.title(f'Populacja zajęcy: Dane vs. Dopasowany model (h_int={h_solver_internal_default})')
plt.grid(True); plt.tight_layout(); plt.savefig(plot_filename_fitting_hares); plt.show()

plt.figure(figsize=(12, 6))
plt.plot(plot_time_axis, lynx_data_lh, 's-', label='Dane (Ryś)', color='red', markersize=4)
plt.plot(plot_time_axis, Y_sim_rss[:, 1], '--', label='Sym. (Ryś) - RSS', color='tomato', lw=1.5)
plt.plot(plot_time_axis, Y_sim_poisson[:, 1], ':', label='Sym. (Ryś) - Poisson', color='orchid', lw=1.5)
plt.xlabel('Rok'); plt.ylabel('Populacja rysi (y)')
plt.legend(); plt.title(f'Populacja rysi: Dane vs. Dopasowany model (h_int={h_solver_internal_default})')
plt.grid(True); plt.tight_layout(); plt.savefig(plot_filename_fitting_lynx); plt.show()

# NOWOŚĆ: Wykres błędów (reszt) dla lepszego dopasowania (np. Poisson)
errors_hares_poisson = hares_data_lh - Y_sim_poisson[:, 0]
errors_lynx_poisson = lynx_data_lh - Y_sim_poisson[:, 1]

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(plot_time_axis, errors_hares_poisson, 'o-', label='Błąd (Zając) - Poisson-like', color='green', markersize=3)
plt.axhline(0, color='black', linestyle='--', lw=1)
plt.xlabel('Rok'); plt.ylabel('Błąd (Dane - Symulacja)')
plt.title(f'Błędy dopasowania modelu (metoda Poisson-like, h_int={h_solver_internal_default})')
plt.legend(); plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(plot_time_axis, errors_lynx_poisson, 's-', label='Błąd (Ryś) - Poisson-like', color='purple', markersize=3)
plt.axhline(0, color='black', linestyle='--', lw=1)
plt.xlabel('Rok'); plt.ylabel('Błąd (Dane - Symulacja)')
plt.legend(); plt.grid(True)

plt.tight_layout(); plt.savefig(plot_filename_fitting_errors); plt.show()
print(f"Wykres błędów dopasowania zapisany jako: {plot_filename_fitting_errors}")

print("\n--- Koniec Skryptu (Wersja Poprawiona z Wykresem Błędów) ---")