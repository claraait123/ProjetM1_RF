import os
import sys
import numpy as np

# ======================================================================
# Configuration des chemins et import des modules du projet
# ======================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(CURRENT_DIR, "..", "classification")
sys.path.append(CODE_DIR)

from loadBDshape import data
from kppv import k_plus_proches_voisins
from kmeans import kmeans_clustering, predict_kmeans, best_results


# Méthodes de description utilisées
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
        n_classes (int|None) : Nombre de classes.
                               Si None, déduit de y_true et y_pred.

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
    Calcule le taux de reconnaissance (accuracy).

    Paramètres :
        y_true (array-like) : Étiquettes réelles.
        y_pred (array-like) : Étiquettes prédites.

    Retour :
        float : Taux de reconnaissance.
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
        acc (float)      : Taux de reconnaissance sur la base de test.
        M (ndarray)      : Matrice de confusion.
        f1s (ndarray)    : F1 par classe.
        f1_macro (float) : F1 macro.
    """
    X = data[methode]
    y = data['labels']

    # Découpage 60 % train, 20 % validation, 20 % test (validation non utilisée ici)
    n = len(X)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)

    n_train = int(0.6 * n)
    n_valid = int(0.2 * n)

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]  # gardé pour cohérence du split
    test_idx = indices[n_train + n_valid:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    print(f"\n=== Évaluation kPPV sur {methode} (k={k}) ===")

    # Prédiction sur le test
    y_test_pred = k_plus_proches_voisins(X_train, y_train, X_test, k=k)

    # Calcul des métriques
    acc = accuracy_score(y_test, y_test_pred)
    M = confusion_matrix(y_test, y_test_pred, n_classes=9)
    _, _, f1s, f1_macro = evaluation_from_confusion(M)

    print(f"Taux de reconnaissance = {acc:.3f}")
    print("Matrice de confusion :\n", M)
    print("F1 par classe :", np.round(f1s, 3))
    print(f"F1 macro = {f1_macro:.3f}")

    return acc, M, f1s, f1_macro


# ======================================================================
# Évaluation K-means
# ======================================================================

def evaluation_kmeans(methode, random_state_split=42, random_state_cluster=0):
    """
    Évalue un K-means supervisé a posteriori (mapping cluster → classe)
    en utilisant le meilleur k déjà trouvé dans kmeans.py.

    Paramètres :
        methode (str)         : Nom de la méthode ('E34', 'GFD', 'SA', 'F0', 'F2').
        random_state_split    : Graine pour le split train/test.
        random_state_cluster  : Graine pour l'initialisation de K-means.

    Retour :
        acc (float)      : Accuracy sur le test.
        f1_macro (float) : F1 macro.
        M (ndarray)      : Matrice de confusion.
        f1s (ndarray)    : F1 par classe.
    """
    # On récupère le meilleur k et son accuracy de référence
    best_k, best_acc_ref = best_results[methode]

    print(
        f"\n=== Évaluation K-means sur {methode} avec le MEILLEUR k = {best_k} "
        f"(trouvé avec seed_split=42, seed_cluster=0) ==="
    )

    X = data[methode]
    y_true = data['labels']

    # Split train/test : 80 % train / 20 % test
    n = len(X)
    np.random.seed(random_state_split)
    indices = np.random.permutation(n)
    n_test = int(0.2 * n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train, y_train = X[train_idx], y_true[train_idx]
    X_test,  y_test  = X[test_idx],  y_true[test_idx]

    # 1. Clustering non supervisé sur le train
    centroids, train_labels = kmeans_clustering(
        X_train, k=best_k, max_it=200, random_state=random_state_cluster
    )

    # 2. Mapping cluster → classe par vote majoritaire
    cluster_to_class = {}
    for cid in range(best_k):
        mask = (train_labels == cid)
        if mask.sum() > 0:
            cluster_to_class[cid] = np.bincount(y_train[mask]).argmax()
        else:
            cluster_to_class[cid] = -1  # cluster vide

    # 3. Prédiction des clusters sur le test
    test_labels = predict_kmeans(X_test, centroids)

    # 4. Conversion clusters → classes
    y_pred = np.array([cluster_to_class.get(c, -1) for c in test_labels])

    # 5. Métriques
    acc = accuracy_score(y_test, y_pred)
    M = confusion_matrix(y_test, y_pred, n_classes=9)
    _, _, f1s, f1_macro = evaluation_from_confusion(M)

    print(f"Accuracy (confirmée) : {acc:.3f} (référence dans kmeans_v2 : {best_acc_ref:.3f})")
    print("Matrice de confusion :\n", M)
    print("F1 par classe :", np.round(f1s, 3))
    print(f"F1-macro : {f1_macro:.3f}")
    print(f"Mapping clusters → classes : {cluster_to_class}")

    return acc, f1_macro, M, f1s


# ======================================================================
# Lancement des évaluations (appelées depuis un autre script si besoin)
# ======================================================================

MEILLEUR_KS = {
    'E34': 5,
    'GFD': 3,
    'SA':  3,
    'F0':  3,
    'F2':  5,
}
for meth in METHODES:
    k_opt = MEILLEUR_KS[meth]
    # Évaluation kPPV avec le meilleur k trouvé
    _ = evaluation_kppv(meth, k=k_opt, seed=0)
    # Évaluation K-means avec k_clusters fixé à 9
    _ = evaluation_kmeans(meth)
