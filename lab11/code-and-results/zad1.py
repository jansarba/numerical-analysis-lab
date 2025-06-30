import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score, confusion_matrix

column_names = [
    "patient ID", "Malignant/Benign", "radius (mean)", "texture (mean)",
    "perimeter (mean)", "area (mean)", "smoothness (mean)", "compactness (mean)",
    "concavity (mean)", "concave points (mean)", "symmetry (mean)",
    "fractal dimension (mean)", "radius (stderr)", "texture (stderr)",
    "perimeter (stderr)", "area (stderr)", "smoothness (stderr)",
    "compactness (stderr)", "concavity (stderr)", "concave points (stderr)",
    "symmetry (stderr)", "fractal dimension (stderr)", "radius (worst)",
    "texture (worst)", "perimeter (worst)", "area (worst)", "smoothness (worst)",
    "compactness (worst)", "concavity (worst)", "concave points (worst)",
    "symmetry (worst)", "fractal dimension (worst)"
]

try:
    train_df = pd.read_csv("breast-cancer-train.dat", header=None, names=column_names)
    validate_df = pd.read_csv("breast-cancer-validate.dat", header=None, names=column_names)
except FileNotFoundError:
    print("Blad: Pliki 'breast-cancer-train.dat' i 'breast-cancer-validate.dat' musza byc w tym samym folderze.")
    exit()


def prepare_data(df):
    b = np.where(df["Malignant/Benign"] == "M", 1, -1)

    feature_cols = [col for col in column_names if col not in ["patient ID", "Malignant/Benign"]]
    A_lin = df[feature_cols].values

    quad_base_cols = ["radius (mean)", "perimeter (mean)", "area (mean)", "symmetry (mean)"]
    A_quad_base = df[quad_base_cols].values

    A_quad_sq = A_quad_base ** 2
    interactions = []
    for i in range(4):
        for j in range(i + 1, 4):
            interactions.append(A_quad_base[:, i] * A_quad_base[:, j])
    A_quad_inter = np.array(interactions).T
    A_quad = np.hstack([A_quad_base, A_quad_sq, A_quad_inter])

    return A_lin, A_quad, b


A_lin_train, A_quad_train, b_train = prepare_data(train_df)
A_lin_test, A_quad_test, b_test = prepare_data(validate_df)

print(f"Zbiory danych przygotowane.")
print(f"Model liniowy: {A_lin_train.shape[1]} cech")
print(f"Model kwadratowy: {A_quad_train.shape[1]} cech")



def solve_least_squares(A, b):
    AtA = A.T @ A
    Atb = A.T @ b
    w = np.linalg.solve(AtA, Atb)
    return w


def solve_gradient_descent(A, b, learning_rate, iterations=10000, tol=1e-6):
    w = np.zeros(A.shape[1])
    for i in range(iterations):
        gradient = 2 * A.T @ (A @ w - b)
        w_new = w - learning_rate * gradient
        if np.linalg.norm(w_new - w) < tol:
            break
        w = w_new
    return w


def evaluate_model(w, A_test, b_test):
    p = A_test @ w
    predictions = np.where(p > 0, 1, -1)
    accuracy = accuracy_score(b_test, predictions)
    cm = confusion_matrix(b_test, predictions, labels=[1, -1])
    return accuracy, cm



print("\n*** PRZYPOMNIENIE: WYNIKI Z LAB 2 (METODA NAJMNIEJSZYCH KWADRATOW) ***")

w_ls_lin = solve_least_squares(A_lin_train, b_train)
acc_ls_lin, cm_ls_lin = evaluate_model(w_ls_lin, A_lin_test, b_test)
print(f"\nModel Liniowy (LS):")
print(f"  Dokladnosc: {acc_ls_lin:.4f}")

w_ls_quad = solve_least_squares(A_quad_train, b_train)
acc_ls_quad, cm_ls_quad = evaluate_model(w_ls_quad, A_quad_test, b_test)
print(f"\nModel Kwadratowy (LS):")
print(f"  Dokladnosc: {acc_ls_quad:.4f}")

print("\n\n*** NOWE WYNIKI: METODA SPADKU WZGLUZ GRADIENTU ***")

print("\n-- Model Liniowy (Gradient Descent) --")
start_time = time.time()
AtA_lin = A_lin_train.T @ A_lin_train
eigenvalues_lin = np.linalg.eigvalsh(AtA_lin)
alpha_opt_lin = 1 / (np.max(eigenvalues_lin) + np.min(eigenvalues_lin))
print(f"  Wybrana stala uczaca (alpha): {alpha_opt_lin:.4e}")

w_gd_lin = solve_gradient_descent(A_lin_train, b_train, learning_rate=alpha_opt_lin)
gd_time_lin = time.time() - start_time
acc_gd_lin, cm_gd_lin = evaluate_model(w_gd_lin, A_lin_test, b_test)
print(f"  Czas obliczen: {gd_time_lin:.4f} s")
print(f"  Dokladnosc: {acc_gd_lin:.4f}")
print(f"  Macierz pomylek:\n{cm_gd_lin}")

print("\n-- Model Kwadratowy (Gradient Descent) --")
start_time = time.time()
AtA_quad = A_quad_train.T @ A_quad_train
eigenvalues_quad = np.linalg.eigvalsh(AtA_quad)
alpha_opt_quad = 1 / (np.max(eigenvalues_quad) + np.min(eigenvalues_quad))
print(f"  Wybrana stala uczaca (alpha): {alpha_opt_quad:.4e}")

w_gd_quad = solve_gradient_descent(A_quad_train, b_train, learning_rate=alpha_opt_quad)
gd_time_quad = time.time() - start_time
acc_gd_quad, cm_gd_quad = evaluate_model(w_gd_quad, A_quad_test, b_test)
print(f"  Czas obliczen: {gd_time_quad:.4f} s")
print(f"  Dokladnosc: {acc_gd_quad:.4f}")
print(f"  Macierz pomylek:\n{cm_gd_quad}")