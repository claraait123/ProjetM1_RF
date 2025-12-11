import os
import sys
import numpy as np

# ======================================================================
# Configuration des chemins et import des modules du projet
# ======================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(CURRENT_DIR, "..", "code")
sys.path.append(CODE_DIR)

from loadBDshape import data
from kppv import k_plus_proches_voisins
from kmeans import kmeans_clustering, predict_kmeans


METHODES = ['E34', 'GFD', 'SA', 'F0', 'F2']


# ======================================================================
# Fonctions de métriques
# ======================================================================

def confusion_matrix(y_true, y_pred, n_classes=None):
    """
    Calcule la matrice de confusion pour un problème de classification.

    Paramètres :
        y_true (array-like)  : Étiquettes réelles.
        y_pred (array-like)  : Étiquettes prédites.
        n_classes (int|None) : Nombre de classes (si None, déduit de y_true/y_pred).

    Retour :
        ndarray (n_classes, n_classes) :
            M[i, j] = nombre d'échantillons de la classe i prédits en j.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1

    M = np.zeros((n_classes, n_classes), dtype=int)

    # Incrément pour chaque paire (vrai, prédit)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            M[t, p] += 1

    return M


def evaluation_from_confusion(M):
    """
    Calcule précision, rappel et F1 par classe à partir d'une matrice de confusion.

    Paramètres :
        M (ndarray) : Matrice de confusion (n_classes, n_classes).

    Retour :
        precisions (ndarray) : Précisions par classe.
        rappels (ndarray)    : Rappels par classe.
        f1s (ndarray)        : Scores F1 par classe.
        f1_macro (float)     : Moyenne macro des F1.
    """
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
    """
    Calcule l'accuracy (taux global de bonnes prédictions).

    Paramètres :
        y_true (array-like) : Étiquettes réelles.
        y_pred (array-like) : Étiquettes prédites.

    Retour :
        float : Accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


# ======================================================================
# Évaluation kPPV
# ======================================================================

def evaluation_kppv(methode, k=1, seed=0):
    """
    Évalue un k-plus-proches-voisins sur une méthode de descripteur donnée.

    Paramètres :
        methode (str) : Nom de la méthode ('E34', 'GFD', 'SA', 'F0', 'F2').
        k (int)       : Nombre de voisins pour kPPV.
        seed (int)    : Graine aléatoire pour le découpage des données.

    Retour :
        acc (float)        : Accuracy sur la base de test.
        M (ndarray)        : Matrice de confusion.
        f1s (ndarray)      : F1 par classe.
        f1_macro (float)   : F1 macro.
    """
    X = data[methode]
    y = data['labels']

    # Découpage 60 % train, 20 % validation, 20 % test (ici on n'utilise que train + test)
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

    # Prédiction sur le test
    y_test_pred = k_plus_proches_voisins(X_train, y_train, X_test, k=k)

    # Calcul des métriques
    acc = accuracy_score(y_test, y_test_pred)
    M = confusion_matrix(y_test, y_test_pred, n_classes=9)
    precisions, rappels, f1s, f1_macro = evaluation_from_confusion(M)

    print(f"Taux de reconnaissance (accuracy) = {acc:.3f}")
    print("Matrice de confusion :\n", M)
    print("F1 par classe :", np.round(f1s, 3))
    print(f"F1 macro = {f1_macro:.3f}")

    return acc, M, f1s, f1_macro


# ======================================================================
# Évaluation K-means
# ======================================================================

def evaluation_kmeans(methode, k_clusters=9, test_ratio=0.2, random_state=42):
    """
    Évalue un K-means supervisé a posteriori (mapping cluster → classe) sur une méthode.

    Paramètres :
        methode (str)    : Nom de la méthode ('E34', 'GFD', 'SA', 'F0', 'F2').
        k_clusters (int) : Nombre de clusters K-means.
        test_ratio (float): Proportion de données en test (le reste en train).
        random_state (int): Graine aléatoire pour split et K-means.

    Retour :
        acc (float)                    : Accuracy sur le test.
        M (ndarray)                    : Matrice de confusion.
        f1s (ndarray)                  : F1 par classe.
        f1_macro (float)               : F1 macro.
        cluster_to_class (dict[int,int]): Mapping cluster → classe.
    """
    X = data[methode]
    y_true = data['labels']

    # Split train / test (sans validation ici)
    n = len(X)
    np.random.seed(random_state)
    indices = np.random.permutation(n)

    n_test = int(test_ratio * n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train, y_train = X[train_idx], y_true[train_idx]
    X_test,  y_test  = X[test_idx],  y_true[test_idx]

    print(f"\n=== Évaluation K-means sur {methode} (k={k_clusters}) ===")

    # 1. Clustering non supervisé sur le train
    centroids, train_cluster_labels = kmeans_clustering(
        X_train, k=k_clusters, max_it=200, random_state=random_state
    )

    # 2. Mapping cluster → classe par vote majoritaire sur y_train
    cluster_to_class = {}
    for cluster_id in range(k_clusters):
        mask = (train_cluster_labels == cluster_id)
        if mask.sum() > 0:
            unique_labels, counts = np.unique(y_train[mask], return_counts=True)
            most_common = unique_labels[np.argmax(counts)]
            cluster_to_class[cluster_id] = most_common
        else:
            cluster_to_class[cluster_id] = -1  # cluster vide

    # 3. Prédiction des clusters sur le test
    test_cluster_labels = predict_kmeans(X_test, centroids)

    # 4. Conversion clusters → classes prédictes
    y_pred = np.array([cluster_to_class.get(c, -1) for c in test_cluster_labels])

    # 5. Calcul des métriques
    acc = accuracy_score(y_test, y_pred)
    M = confusion_matrix(y_test, y_pred, n_classes=9)
    precisions, rappels, f1s, f1_macro = evaluation_from_confusion(M)

    print(f"Taux de reconnaissance (accuracy) = {acc:.3f}")
    print("Matrice de confusion :\n", M)
    print("F1 par classe :", np.round(f1s, 3))
    print(f"F1 macro = {f1_macro:.3f}")
    print("Mapping clusters -> classes :", cluster_to_class)

    return acc, M, f1s, f1_macro, cluster_to_class


# ======================================================================
# Lancement des évaluations
# ======================================================================

MEILLEUR_KS = {
    'E34': 5,
    'GFD': 3,
    'SA':  3,
    'F0':  3,
    'F2':  5,
}

if __name__ == "__main__":
    for meth in METHODES:
        k_opt = MEILLEUR_KS[meth]
        # Évaluation kPPV avec le meilleur k trouvé
        _ = evaluation_kppv(meth, k=k_opt, seed=0)
        # Évaluation K-means avec k_clusters fixé à 9
        _ = evaluation_kmeans(meth, k_clusters=9, test_ratio=0.2, random_state=42)
