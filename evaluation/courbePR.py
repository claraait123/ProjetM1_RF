import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# === chemin vers code/ ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(CURRENT_DIR, "..", "code")
sys.path.append(CODE_DIR)

from loadBDshape import data
from kppv import k_plus_proches_voisins
from kmeans_v2 import kmeans_clustering, predict_kmeans
# On récupère directement les meilleurs k déjà calculés dans kmeans_v2.py
from kmeans_v2 import best_results

METHODES = ['E34', 'GFD', 'SA', 'F0', 'F2']
K_KPPV = 10
SEED_SPLIT = 0          # même seed que dans la partie kPPV originale
SEED_KMEANS = 0         # cohérence avec la recherche du meilleur k

print("=== COURBES PR - kPPV (K=10) et K-means (meilleur k trouvé) ===\n")

# ===================================================================
# 1. Courbes PR pour kPPV (K=10) → comme demandé dans l'énoncé
# ===================================================================
for methode in METHODES:
    print(f"\n=== Courbe PR - {methode} - kPPV (K={K_KPPV}) ===")

    X = data[methode]
    y = data['labels']

    n = len(X)
    indices = np.arange(n)
    np.random.seed(SEED_SPLIT)
    np.random.shuffle(indices)

    n_train = int(0.6 * n)
    train_idx = indices[:n_train]
    test_idx  = indices[n_train + int(0.2*n):]   # 20% test

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # Prédiction kPPV
    y_pred = k_plus_proches_voisins(X_train, y_train, X_test, k=K_KPPV)

    classes = np.unique(y)
    aucs = []
    plt.figure(figsize=(7, 5))

    for c in classes:
        y_true_bin = (y_test == c).astype(int)
        y_score_bin = (y_pred == c).astype(int)   # score binaire (pas de probas)

        precision, recall, _ = precision_recall_curve(y_true_bin, y_score_bin)
        auc_pr = auc(recall, precision)
        aucs.append(auc_pr)
        plt.plot(recall, precision, label=f"Classe {c} (AUC={auc_pr:.3f})")

    macro_auc = np.mean(aucs)
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.title(f"{methode} - kPPV (K={K_KPPV})\nAUC macro = {macro_auc:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.show()

    print(f"→ AUC macro kPPV = {macro_auc:.3f}")

# ===================================================================
# 2. Courbes PR pour K-means avec le MEILLEUR k trouvé dans kmeans_v2.py
# ===================================================================
print("\n" + "="*80)
print("COURBES PR - K-means avec le meilleur k optimisé (seed_split=42, seed_cluster=0)")
print("="*80)

for methode in METHODES:
    meilleur_k, acc_ref = best_results[methode]
    print(f"\n=== Courbe PR - {methode} - K-means (k={meilleur_k}) ===")

    X = data[methode]
    y = data['labels']

    n = len(X)
    indices = np.arange(n)
    np.random.seed(SEED_SPLIT)
    np.random.shuffle(indices)

    n_train = int(0.6 * n)
    train_idx = indices[:n_train]
    test_idx  = indices[n_train + int(0.2*n):]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # 1. Clustering sur le train
    centroids, _ = kmeans_clustering(X_train, k=meilleur_k, max_it=200, random_state=SEED_KMEANS)

    # 2. Mapping cluster → classe (vote majoritaire sur train)
    mapping = {}
    for cid in range(meilleur_k):
        mask = (_ == cid)
        if mask.sum() > 0:
            mapping[cid] = np.bincount(y_train[mask]).argmax()
        else:
            mapping[cid] = -1

    # 3. Prédiction sur test
    test_clusters = predict_kmeans(X_test, centroids)
    y_pred = np.array([mapping.get(c, -1) for c in test_clusters])

    # 4. Courbe PR
    classes = np.unique(y)
    aucs = []
    plt.figure(figsize=(7, 5))

    for c in classes:
        y_true_bin = (y_test == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)

        precision, recall, _ = precision_recall_curve(y_true_bin, y_pred_bin)
        auc_pr = auc(recall, precision)
        aucs.append(auc_pr)
        plt.plot(recall, precision, label=f"Classe {c} (AUC={auc_pr:.3f})")

    macro_auc = np.mean(aucs)
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.title(f"{methode} - K-means (k={meilleur_k})\nAUC macro = {macro_auc:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.show()

    print(f"→ Meilleur k = {meilleur_k} | AUC macro K-means = {macro_auc:.3f} (acc référence = {acc_ref:.3f})")