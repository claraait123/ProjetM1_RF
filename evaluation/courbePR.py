import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# ajout du dossier code au path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(CURRENT_DIR, "..", "code")
sys.path.append(CODE_DIR)

from loadBDshape import data
from kppv import k_plus_proches_voisins

METHODES = ['E34', 'GFD', 'SA', 'F0', 'F2']
K = 10

for methode in METHODES:
    print("\n=== Courbe PR pour la méthode", methode, "===")

    X = data[methode]
    y = data['labels']     # labels ∈ {1..9}

    # Découpage 60 / 20 / 20
    n = len(X)
    indices = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(indices)

    n_train = int(0.6 * n)
    n_valid = int(0.2 * n)

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx  = indices[n_train + n_valid:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # === Prédiction KPPV avec K = 10 ===
    y_pred = k_plus_proches_voisins(X_train, y_train, X_test, k=K)

    # === PR + AUC par classe (one-vs-all) ===
    classes = np.unique(y)
    aucs = []

    plt.figure(figsize=(7,5))

    for c in classes:
        y_true_bin = (y_test == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)

        precision, recall, _ = precision_recall_curve(y_true_bin, y_pred_bin)
        A = auc(recall, precision)
        aucs.append(A)

        plt.plot(recall, precision, label=f"Classe {c} (AUC={A:.2f})")

    # Macro AUC
    macro_auc = np.mean(aucs)

    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.title(f"Courbe PR — {methode} (K={K}) — AUC macro = {macro_auc:.3f}")
    plt.legend()
    plt.grid(True)

    plt.show()

    print(f"AUC macro pour {methode} = {macro_auc:.3f}")