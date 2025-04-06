import pandas as pd
import numpy as np
from itertools import combinations
from scipy.linalg import lstsq
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Wczytanie danych
train_data = pd.read_csv("dataset/breast-cancer-train.dat", delimiter=",")
validate_data = pd.read_csv("dataset/breast-cancer-validate.dat", delimiter=",")

# Wczytanie nazw kolumn
with open("dataset/breast-cancer.labels") as f:
    column_names = f.read().splitlines()

train_data.columns = column_names
validate_data.columns = column_names

# Konwersja etykiet na wartości numeryczne
train_data["Malignant/Benign"] = np.where(train_data["Malignant/Benign"] == "M", 1, -1)
validate_data["Malignant/Benign"] = np.where(validate_data["Malignant/Benign"] == "M", 1, -1)

# Wizualizacja danych
feature = "radius (mean)"

# Histogram
plt.hist(train_data[feature][train_data["Malignant/Benign"] == 1], alpha=0.5, label="Malignant", bins=30)
plt.hist(train_data[feature][train_data["Malignant/Benign"] == -1], alpha=0.5, label="Benign", bins=30)
plt.xlabel(feature)
plt.ylabel("Count")
plt.title("Histogram of " + feature)
plt.legend()
plt.savefig("histogram.png")
plt.show()

# Wykres posortowanych wartości
sorted_feature = train_data.sort_values(by=feature)
plt.plot(sorted_feature[feature].values, label=feature)
plt.xlabel("Index")
plt.ylabel(feature)
plt.title("Sorted Feature Values")
plt.legend()
plt.savefig("sorted.png")
plt.show()

# Definicja cech
all_features = [col for col in column_names if col not in ["patient ID", "Malignant/Benign"]]  # Wszystkie 30 cech
quad_features = ["radius (mean)", "perimeter (mean)", "area (mean)", "symmetry (mean)"]  # 4 wybrane cechy

# Przygotowanie danych dla modelu LINIOWEGO (wszystkie 30 cech)
A_lin_train = train_data[all_features].values
A_lin_validate = validate_data[all_features].values

# Przygotowanie danych dla modelu KWADRATOWEGO (4 cechy + kwadraty + interakcje)
A_quad_train = train_data[quad_features].values
A_quad_validate = validate_data[quad_features].values

# Rozszerzenie macierzy kwadratowej
A_quad_train = np.hstack([A_quad_train, A_quad_train ** 2])
A_quad_validate = np.hstack([A_quad_validate, A_quad_validate ** 2])

for i, j in combinations(range(len(quad_features)), 2):
    A_quad_train = np.column_stack((A_quad_train, A_quad_train[:, i] * A_quad_train[:, j]))
    A_quad_validate = np.column_stack((A_quad_validate, A_quad_validate[:, i] * A_quad_validate[:, j]))

# Wektory etykiet
b_train = train_data["Malignant/Benign"].values
b_validate = validate_data["Malignant/Benign"].values

# Rozwiązanie równań normalnych
w_lin = np.linalg.solve(A_lin_train.T @ A_lin_train, A_lin_train.T @ b_train)
w_quad = np.linalg.solve(A_quad_train.T @ A_quad_train, A_quad_train.T @ b_train)

# Rozwiązanie z użyciem SVD (tylko dla modelu liniowego)
w_lin_svd, _, _, _ = lstsq(A_lin_train, b_train)

# Regularyzacja Ridge (tylko dla modelu liniowego)
lambda_reg = 0.01
I = np.eye(A_lin_train.shape[1])
w_ridge = np.linalg.solve(A_lin_train.T @ A_lin_train + lambda_reg * I, A_lin_train.T @ b_train)

# Obliczenie współczynników uwarunkowania
cond_lin = np.linalg.cond(A_lin_train.T @ A_lin_train)
cond_quad = np.linalg.cond(A_quad_train.T @ A_quad_train)
cond_ridge = np.linalg.cond(A_lin_train.T @ A_lin_train + lambda_reg * I)

print(f"Condition number (linear): {cond_lin}")
print(f"Condition number (quadratic): {cond_quad}")
print(f"Condition number (linear - Ridge Regression): {cond_ridge}")

# Funkcja do ewaluacji modelu
def evaluate_model(A, w, b_true, model_name):
    p = A @ w
    pred = np.where(p > 0, 1, -1)
    cm = confusion_matrix(b_true, pred)
    acc = accuracy_score(b_true, pred)
    print(f"Confusion Matrix ({model_name}):\n{cm}")
    print(f"Accuracy ({model_name}): {acc:.4f}")
    return cm, acc

# Ewaluacja wszystkich modeli
print("\n=== Linear Models ===")
cm_lin, acc_lin = evaluate_model(A_lin_validate, w_lin, b_validate, "Linear - Normal Equations")
cm_svd, acc_svd = evaluate_model(A_lin_validate, w_lin_svd, b_validate, "Linear - SVD")
cm_ridge, acc_ridge = evaluate_model(A_lin_validate, w_ridge, b_validate, "Linear - Ridge Regression")

print("\n=== Quadratic Model ===")
cm_quad, acc_quad = evaluate_model(A_quad_validate, w_quad, b_validate, "Quadratic - Normal Equations")