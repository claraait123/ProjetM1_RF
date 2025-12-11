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

METHODES = ['E34', 'GFD', 'SA', 'F0', 'F2']


def confusion_matrix(y_true, y_pred, n_classes=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1

    M = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(len(y_true)):
        t = y_true[i]
        p = y_pred[i]
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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


# =====================
#   KPPV
# =====================

def evaluation_kppv(methode, k=1, seed=0):
    X = data[methode]
    y = data['labels']

    n = len(X)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)

    n_train = int(0.6 * n)
    n_valid = int(0.2 * n)

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    print(f"\n=== Évaluation kPPV sur {methode} (k={k}) ===")

    y_test_pred = k_plus_proches_voisins(X_train, y_train, X_test, k=k)

    acc = accuracy_score(y_test, y_test_pred)
    M = confusion_matrix(y_test, y_test_pred, n_classes=9)
    precisions, rappels, f1s, f1_macro = evaluation_from_confusion(M)

    print(f"Taux de reconnaissance (accuracy) = {acc:.3f}")
    print("Matrice de confusion :\n", M)
    print("F1 par classe :", np.round(f1s, 3))
    print(f"F1 macro = {f1_macro:.3f}")

    return acc, M, f1s, f1_macro


# =====================
#   K-MEANS 
# =====================

def evaluation_kmeans(methode, k_clusters=9, test_ratio=0.2, random_state=42):
    X = data[methode]
    y_true = data['labels']

    n = len(X)
    np.random.seed(random_state)
    indices = np.random.permutation(n)

    n_test = int(test_ratio * n)
    test_idx  = indices[:n_test]
    train_idx = indices[n_test:]

    X_train, y_train = X[train_idx], y_true[train_idx]
    X_test,  y_test  = X[test_idx],  y_true[test_idx]

    print(f"\n=== Évaluation K-means sur {methode} (k={k_clusters}) ===")

    # 1. Clustering non supervisé sur le train
    centroids, train_cluster_labels = kmeans_clustering(
        X_train, k=k_clusters, max_it=200, random_state=random_state
    )

    # 2. Mapping cluster → classe par vote majoritaire (train)
    cluster_to_class = {}
    for cluster_id in range(k_clusters):
        mask = (train_cluster_labels == cluster_id)
        if mask.sum() > 0:
            unique_labels, counts = np.unique(y_train[mask], return_counts=True)
            most_common = unique_labels[np.argmax(counts)]
            cluster_to_class[cluster_id] = most_common
        else:
            cluster_to_class[cluster_id] = -1  # cluster vide

    # 3. Prédiction sur le test
    test_cluster_labels = predict_kmeans(X_test, centroids)

    # 4. Conversion cluster → classe
    y_pred = np.array([cluster_to_class.get(c, -1) for c in test_cluster_labels])

    # 5. Metrics
    acc = accuracy_score(y_test, y_pred)
    M = confusion_matrix(y_test, y_pred, n_classes=9)
    precisions, rappels, f1s, f1_macro = evaluation_from_confusion(M)

    print(f"Taux de reconnaissance (accuracy) = {acc:.3f}")
    print("Matrice de confusion :\n", M)
    print("F1 par classe :", np.round(f1s, 3))
    print(f"F1 macro = {f1_macro:.3f}")
    print("Mapping clusters -> classes :", cluster_to_class)

    return acc, M, f1s, f1_macro, cluster_to_class


MEILLEUR_KS = {
    'E34': 5,
    'GFD': 3,
    'SA': 3,
    'F0': 3,
    'F2': 5,
}

for meth in METHODES:
    k_opt = MEILLEUR_KS[meth]
    _ = evaluation_kppv(meth, k=k_opt, seed=0)
    _ = evaluation_kmeans(meth, k_clusters=9, test_ratio=0.2, random_state=42)
