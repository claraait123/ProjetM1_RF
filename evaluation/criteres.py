import os
import sys
import numpy as np

# === chemin vers code/ ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(CURRENT_DIR, "..", "code")
sys.path.append(CODE_DIR)

from loadBDshape import data
from kppv import k_plus_proches_voisins
from kmeans_v2 import kmeans_clustering, predict_kmeans
# On importe directement les résultats déjà calculés dans kmeans_v2.py
from kmeans_v2 import best_results  # C'EST ÇA LA MAGIE

METHODES = ['E34', 'GFD', 'SA', 'F0', 'F2']

# === Fonctions utilitaires (confusion, F1, etc.) restent identiques ===
def confusion_matrix(y_true, y_pred, n_classes=9):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    M = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            M[t, p] += 1
    return M

def evaluation_from_confusion(M):
    n_classes = M.shape[0]
    precisions = np.zeros(n_classes)
    rappels = np.zeros(n_classes)
    f1s = np.zeros(n_classes)

    for c in range(n_classes):
        tp = M[c, c]
        fp = M[:, c].sum() - tp
        fn = M[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions[c] = precision
        rappels[c] = recall
        f1s[c] = f1

    f1_macro = f1s.mean()
    return precisions, rappels, f1s, f1_macro

def accuracy_score(y_true, y_pred):
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))

# === kPPV (inchangé) ===
def evaluation_kppv(methode, k=1, seed=0):
    X = data[methode]
    y = data['labels']
    n = len(X)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)

    n_train = int(0.6 * n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train + int(0.2*n):]  # on garde 60% train, 20% ignoré, 20% test

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"\n=== Évaluation kPPV sur {methode} (k={k}) ===")
    y_pred = k_plus_proches_voisins(X_train, y_train, X_test, k=k)

    acc = accuracy_score(y_test, y_pred)
    M = confusion_matrix(y_test, y_pred)
    _, _, f1s, f1_macro = evaluation_from_confusion(M)

    print(f"Accuracy : {acc:.3f}")
    print("Matrice de confusion :\n", M)
    print("F1 par classe :", np.round(f1s, 3))
    print(f"F1-macro : {f1_macro:.3f}")

    return acc, f1_macro

# === K-MEANS : on utilise le meilleur k déjà trouvé dans kmeans_v2.py ===
def evaluation_kmeans_avec_meilleur_k(methode, random_state_split=42, random_state_cluster=0):
    best_k, best_acc_ref = best_results[methode]  # On récupère le meilleur k déjà calculé

    print(f"\n=== Évaluation K-means sur {methode} avec le MEILLEUR k = {best_k} "
          f"(trouvé avec seed_split=42, seed_cluster=0) ===")

    X = data[methode]
    y_true = data['labels']

    n = len(X)
    np.random.seed(random_state_split)
    indices = np.random.permutation(n)
    n_test = int(0.2 * n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train, y_train = X[train_idx], y_true[train_idx]
    X_test, y_test = X[test_idx], y_true[test_idx]

    # Clustering
    centroids, train_labels = kmeans_clustering(X_train, k=best_k, max_it=200, random_state=random_state_cluster)

    # Mapping cluster → classe
    cluster_to_class = {}
    for cid in range(best_k):
        mask = (train_labels == cid)
        if mask.sum() > 0:
            cluster_to_class[cid] = np.bincount(y_train[mask]).argmax()
        else:
            cluster_to_class[cid] = -1

    # Prédiction test
    test_labels = predict_kmeans(X_test, centroids)
    y_pred = np.array([cluster_to_class.get(c, -1) for c in test_labels])

    # Métriques
    acc = accuracy_score(y_test, y_pred)
    M = confusion_matrix(y_test, y_pred)
    _, _, f1s, f1_macro = evaluation_from_confusion(M)

    print(f"Accuracy (confirmée) : {acc:.3f} (référence dans kmeans_v2: {best_acc_ref:.3f})")
    print("Matrice de confusion :\n", M)
    print("F1 par classe :", np.round(f1s, 3))
    print(f"F1-macro : {f1_macro:.3f}")
    print(f"Mapping clusters → classes : {cluster_to_class}")

    return acc, f1_macro, M, f1s


# =======================
#   Lancement final
# =======================

print("=== ÉVALUATION FINALE POUR LE RAPPORT ===\n")

for meth in METHODES:
    # 1. kPPV avec k=1 (comme demandé souvent dans le sujet)
    evaluation_kppv(meth, k=1, seed=0)

    # 2. K-means avec le meilleur k trouvé dans kmeans_v2.py
    evaluation_kmeans_avec_meilleur_k(meth)
    print("\n" + "-"*80)