import pandas as pd
import numpy as np
from itertools import combinations
from scipy.linalg import lstsq
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import time # <-- DODANO

# --- 1. Wczytanie i przygotowanie danych ---
# Wczytanie danych
try:
    train_data = pd.read_csv("dataset/breast-cancer-train.dat", delimiter=",")
    validate_data = pd.read_csv("dataset/breast-cancer-validate.dat", delimiter=",")
    # Wczytanie nazw kolumn
    with open("dataset/breast-cancer.labels") as f:
        column_names = f.read().splitlines()
    train_data.columns = column_names
    validate_data.columns = column_names
except FileNotFoundError:
    print("Błąd: Pliki danych ('breast-cancer-train.dat', 'breast-cancer-validate.dat', 'breast-cancer.labels') nie zostały znalezione.")
    print("Upewnij się, że znajdują się w tym samym folderze co skrypt.")
    exit()

# Konwersja etykiet na wartości numeryczne
train_data["Malignant/Benign"] = np.where(train_data["Malignant/Benign"] == "M", 1, -1)
validate_data["Malignant/Benign"] = np.where(validate_data["Malignant/Benign"] == "M", 1, -1)

# --- 2. Wizualizacja (opcjonalnie, można zakomentować) ---
feature = "radius (mean)"
plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(train_data[feature][train_data["Malignant/Benign"] == 1], alpha=0.7, label="Malignant", bins=25)
plt.hist(train_data[feature][train_data["Malignant/Benign"] == -1], alpha=0.7, label="Benign", bins=25)
plt.xlabel(feature)
plt.ylabel("Count")
plt.title("Histogram of " + feature)
plt.legend()

# Wykres posortowanych wartości
plt.subplot(1, 2, 2)
sorted_feature_df = train_data.sort_values(by=feature)
plt.plot(sorted_feature_df[feature].values, label=feature)
plt.xlabel("Index")
plt.ylabel(feature)
plt.title("Sorted Feature Values")
plt.legend()
plt.tight_layout()
plt.savefig("data_visualization.png")
plt.show()

# --- 3. Przygotowanie macierzy A i wektora b ---
# Definicja cech
all_features = [col for col in column_names if col not in ["patient ID", "Malignant/Benign"]]
quad_features = ["radius (mean)", "perimeter (mean)", "area (mean)", "symmetry (mean)"]

# Przygotowanie danych dla modelu LINIOWEGO
A_lin_train = train_data[all_features].values
A_lin_validate = validate_data[all_features].values

# Przygotowanie danych dla modelu KWADRATOWEGO
def create_quadratic_features(df_slice):
    A_quad_base = df_slice.copy()
    # Dodaj kwadraty cech
    A_quad_sq = A_quad_base ** 2
    # Dodaj interakcje (iloczyny par)
    interactions = []
    for i, j in combinations(range(A_quad_base.shape[1]), 2):
        interactions.append(A_quad_base[:, i] * A_quad_base[:, j])
    A_quad_inter = np.array(interactions).T
    return np.hstack([A_quad_base, A_quad_sq, A_quad_inter])

A_quad_train = create_quadratic_features(train_data[quad_features].values)
A_quad_validate = create_quadratic_features(validate_data[quad_features].values)

# Wektory etykiet
b_train = train_data["Malignant/Benign"].values
b_validate = validate_data["Malignant/Benign"].values


# --- 4. Obliczenia, pomiar czasu i ewaluacja ---

# Funkcja do ewaluacji modelu
def evaluate_model(A_val, w, b_true):
    p = A_val @ w
    pred = np.where(p > 0, 1, -1)
    cm = confusion_matrix(b_true, pred, labels=[1, -1])
    acc = accuracy_score(b_true, pred)
    return cm, acc

print("\n" + "="*40)
print("Analiza modeli - Metody analityczne i SVD")
print("="*40 + "\n")

# Model Liniowy - Równania normalne
print("--- Model Liniowy (Równania Normalne) ---")
start_time = time.time()
AtA_lin = A_lin_train.T @ A_lin_train
Atb_lin = A_lin_train.T @ b_train
w_lin = np.linalg.solve(AtA_lin, Atb_lin)
time_lin = time.time() - start_time
cm_lin, acc_lin = evaluate_model(A_lin_validate, w_lin, b_validate)
print(f"Czas obliczeń: {time_lin:.6f} s")
print(f"Dokładność: {acc_lin:.4f}")
print(f"Macierz pomyłek:\n{cm_lin}")
print(f"Współczynnik uwarunkowania (A^T A): {np.linalg.cond(AtA_lin):.2e}\n")


# Model Liniowy - SVD (lstsq)
print("--- Model Liniowy (SVD - scipy.linalg.lstsq) ---")
start_time = time.time()
w_lin_svd, _, _, _ = lstsq(A_lin_train, b_train)
time_svd = time.time() - start_time
cm_svd, acc_svd = evaluate_model(A_lin_validate, w_lin_svd, b_validate)
print(f"Czas obliczeń: {time_svd:.6f} s")
print(f"Dokładność: {acc_svd:.4f}")
print(f"Macierz pomyłek:\n{cm_svd}\n")


# Model Liniowy - Regularyzacja Ridge
print("--- Model Liniowy (Regularyzacja Ridge) ---")
lambda_reg = 0.01
I = np.eye(A_lin_train.shape[1])
AtA_ridge = AtA_lin + lambda_reg * I
start_time = time.time()
w_ridge = np.linalg.solve(AtA_ridge, Atb_lin)
time_ridge = time.time() - start_time
cm_ridge, acc_ridge = evaluate_model(A_lin_validate, w_ridge, b_validate)
print(f"Czas obliczeń: {time_ridge:.6f} s")
print(f"Dokładność: {acc_ridge:.4f}")
print(f"Macierz pomyłek:\n{cm_ridge}")
print(f"Współczynnik uwarunkowania (A^T A + lambda*I): {np.linalg.cond(AtA_ridge):.2e}\n")


# Model Kwadratowy - Równania normalne
print("--- Model Kwadratowy (Równania Normalne) ---")
start_time = time.time()
AtA_quad = A_quad_train.T @ A_quad_train
Atb_quad = A_quad_train.T @ b_train
w_quad = np.linalg.solve(AtA_quad, Atb_quad)
time_quad = time.time() - start_time
cm_quad, acc_quad = evaluate_model(A_quad_validate, w_quad, b_validate)
print(f"Czas obliczeń: {time_quad:.6f} s")
print(f"Dokładność: {acc_quad:.4f}")
print(f"Macierz pomyłek:\n{cm_quad}")
print(f"Współczynnik uwarunkowania (A^T A): {np.linalg.cond(AtA_quad):.2e}\n")