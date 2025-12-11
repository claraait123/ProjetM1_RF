import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# ======================================================================
# Configuration des chemins et import des modules du projet
# ======================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(CURRENT_DIR, "..", "code")
sys.path.append(CODE_DIR)

from loadBDshape import data
from kppv import k_plus_proches_voisins
from kmeans import kmeans_clustering, predict_kmeans, best_results


# ======================================================================
# Constantes globales
# ======================================================================

METHODES = ['E34', 'GFD', 'SA', 'F0', 'F2']
K_KPPV = 10          # nombre de voisins pour kPPV
SEED_SPLIT = 0       # seed pour le découpage train/valid/test
SEED_KMEANS = 0      # seed pour l'initialisation de K-means


def split_train_test(X, y, seed_split=0, train_ratio=0.6, valid_ratio=0.2):
    """
    Découpe X, y en apprentissage / validation / test selon les ratios fournis.

    Paramètres :
        X (ndarray)        : Données complètes.
        y (ndarray)        : Étiquettes complètes.
        seed_split (int)   : Graine aléatoire pour le mélange des indices.
        train_ratio (float): Proportion pour l'apprentissage.
        valid_ratio (float): Proportion pour la validation (le reste en test).

    Retour :
        (X_train, y_train, X_valid, y_valid, X_test, y_test)
    """
    n = len(X)
    indices = np.arange(n)
    np.random.seed(seed_split)
    np.random.shuffle(indices)

    n_train = int(train_ratio * n)
    n_valid = int(valid_ratio * n)

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def plot_pr_curves_per_class(y_test, y_pred, classes, title_prefix):
    """
    Calcule et trace les courbes précision–rappel par classe (binarisation un-versus-rest),
    puis renvoie l'AUC macro.

    Paramètres :
        y_test (ndarray)  : Étiquettes réelles du test.
        y_pred (ndarray)  : Étiquettes prédites (kPPV ou K-means).
        classes (ndarray) : Liste des classes présentes.
        title_prefix (str): Préfixe pour le titre de la figure.

    Retour :
        float : AUC macro (moyenne des AUC PR par classe).
    """
    aucs = []
    plt.figure(figsize=(7, 5))

    for c in classes:
        # Binarisation : classe c vs le reste
        y_true_bin = (y_test == c).astype(int)
        y_pred_bin = (y_pred == c).astype(int)

        precision, recall, _ = precision_recall_curve(y_true_bin, y_pred_bin)
        auc_pr = auc(recall, precision)
        aucs.append(auc_pr)
        plt.plot(recall, precision, label=f"Classe {c} (AUC={auc_pr:.3f})")

    macro_auc = np.mean(aucs)
    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.title(f"{title_prefix}\nAUC macro = {macro_auc:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.show()

    return macro_auc


def evaluate_pr_kppv(methodes, k, seed_split):
    """
    Trace les courbes PR pour kPPV (K fixé) sur toutes les méthodes.

    Paramètres :
        methodes (list[str]) : Liste des méthodes de descripteurs.
        k (int)              : Nombre de voisins pour kPPV.
        seed_split (int)     : Graine pour le split.
    """
    print("=== COURBES PR - kPPV (K={}) ===\n".format(k))

    for methode in methodes:
        print(f"\n=== Courbe PR - {methode} - kPPV (K={k}) ===")

        X = data[methode]
        y = data['labels']

        # Découpage train/valid/test (on n'utilise ici que train + test)
        X_train, y_train, _, _, X_test, y_test = split_train_test(
            X, y, seed_split=seed_split, train_ratio=0.6, valid_ratio=0.2
        )

        # Prédictions kPPV sur le test
        y_pred = k_plus_proches_voisins(X_train, y_train, X_test, k=k)

        classes = np.unique(y)
        title = f"{methode} - kPPV (K={k})"
        macro_auc = plot_pr_curves_per_class(y_test, y_pred, classes, title)

        print(f"→ AUC macro kPPV = {macro_auc:.3f}")


def build_cluster_to_class_mapping(X_train, y_train, k, seed_kmeans):
    """
    Apprend un K-means sur X_train, puis construit un mapping cluster → classe
    par vote majoritaire sur les labels de train.

    Paramètres :
        X_train (ndarray) : Données d'apprentissage.
        y_train (ndarray) : Étiquettes d'apprentissage.
        k (int)           : Nombre de clusters K-means.
        seed_kmeans (int) : Graine pour l'initialisation des centroïdes.

    Retour :
        centroids (ndarray)        : Centroïdes appris.
        mapping (dict[int, int])   : cluster_id → classe_majoritaire ou -1 si vide.
    """
    centroids, train_cluster_labels = kmeans_clustering(
        X_train, k=k, max_it=200, random_state=seed_kmeans
    )

    mapping = {}
    for cid in range(k):
        mask = (train_cluster_labels == cid)
        if mask.sum() > 0:
            # Vote majoritaire sur les étiquettes de train pour ce cluster
            mapping[cid] = np.bincount(y_train[mask]).argmax()
        else:
            mapping[cid] = -1  # cluster vide

    return centroids, mapping


def evaluate_pr_kmeans(methodes, seed_split, seed_kmeans):
    """
    Trace les courbes PR pour K-means avec le meilleur k déjà trouvé
    (stocké dans best_results) pour chaque méthode.

    Paramètres :
        methodes (list[str]) : Liste des méthodes de descripteurs.
        seed_split (int)     : Graine pour le split train/valid/test.
        seed_kmeans (int)    : Graine pour l'initialisation de K-means.
    """
    print("\n" + "=" * 80)
    print("COURBES PR - K-means avec le meilleur k optimisé")
    print("=" * 80)

    for methode in methodes:
        meilleur_k, acc_ref = best_results[methode]
        print(f"\n=== Courbe PR - {methode} - K-means (k={meilleur_k}) ===")

        X = data[methode]
        y = data['labels']

        X_train, y_train, _, _, X_test, y_test = split_train_test(
            X, y, seed_split=seed_split, train_ratio=0.6, valid_ratio=0.2
        )

        # 1. Clustering K-means + mapping cluster → classe
        centroids, mapping = build_cluster_to_class_mapping(
            X_train, y_train, k=meilleur_k, seed_kmeans=seed_kmeans
        )

        # 2. Prédiction des clusters sur le test
        test_clusters = predict_kmeans(X_test, centroids)
        # 3. Conversion clusters → classes
        y_pred = np.array([mapping.get(c, -1) for c in test_clusters])

        # 4. Courbes PR par classe
        classes = np.unique(y)
        title = f"{methode} - K-means (k={meilleur_k})"
        macro_auc = plot_pr_curves_per_class(y_test, y_pred, classes, title)

        print(
            f"→ Meilleur k = {meilleur_k} | "
            f"AUC macro K-means = {macro_auc:.3f} "
            f"(acc référence = {acc_ref:.3f})"
        )


if __name__ == "__main__":
    print("=== COURBES PR - kPPV (K=10) et K-means (meilleur k trouvé) ===\n")

    # 1) Courbes PR pour kPPV
    evaluate_pr_kppv(METHODES, k=K_KPPV, seed_split=SEED_SPLIT)

    # 2) Courbes PR pour K-means avec meilleur k
    evaluate_pr_kmeans(METHODES, seed_split=SEED_SPLIT, seed_kmeans=SEED_KMEANS)
