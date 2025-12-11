import numpy as np
import matplotlib.pyplot as plt

from loadBDshape import data
from kppv import distance_minkowski


def kmeans_clustering(X, k=9, max_it=200, random_state=None):
    """
    Implémentation simple de K-means avec distance de Minkowski (p=2, euclidienne).

    Paramètres :
        X (ndarray)        : Données (n_samples, n_features).
        k (int)            : Nombre de clusters.
        max_it (int)       : Nombre maximal d'itérations.
        random_state (int) : Graine aléatoire pour l'initialisation des centroïdes.

    Retour :
        centroids (ndarray) : Centroïdes finaux (k, n_features).
        labels (ndarray)    : Indices de clusters pour chaque point de X (n_samples,).
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Initialisation aléatoire des centroïdes dans le bounding box des données
    centroids = np.random.uniform(
        np.amin(X, axis=0),
        np.amax(X, axis=0),
        size=(k, X.shape[1])
    )

    for _ in range(max_it):

        # 1. Attribution : on assigne chaque point au centroïde le plus proche
        distances = distance_minkowski(X, centroids, 2)    # p=2 → euclidienne
        labels = np.argmin(distances, axis=1)

        # 2. Mise à jour des centroïdes
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = (labels == i)
            if np.any(mask):
                new_centroids[i] = X[mask].mean(axis=0)
            else:
                # Cluster vide → on garde l'ancien centroïde
                new_centroids[i] = centroids[i]

        # 3. Test de convergence
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break

        centroids = new_centroids

    return centroids, labels


def predict_kmeans(X_test, centroids):
    """
    Assigne chaque point de X_test au cluster le plus proche (K-means déjà entraîné).

    Paramètres :
        X_test (ndarray)    : Données à clusteriser (n_samples, n_features).
        centroids (ndarray) : Centroïdes appris (k, n_features).

    Retour :
        ndarray : Labels de clusters (n_samples,).
    """
    distances = distance_minkowski(X_test, centroids, 2)
    return np.argmin(distances, axis=1)


def simple_pca(X, n_components=2):
    """
    Réduction de dimension via PCA (SVD) pour visualisation.

    Paramètres :
        X (ndarray)       : Données initiales (n_samples, n_features).
        n_components (int): Nombre de composantes principales.

    Retour :
        X_reduced (ndarray) : Données projetées (n_samples, n_components).
    """
    # Centrage des données
    X_centered = X - np.mean(X, axis=0)

    # SVD sur les données centrées
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Les composantes principales sont les premières lignes de Vt
    components = Vt[:n_components]

    # Projection sur les composantes
    X_reduced = np.dot(X_centered, components.T)
    return X_reduced


def evaluate_kmeans_on_method(
    method='E34',
    k_clusters=9,
    test_ratio=0.2,
    random_state=42,
    aff=False,
    random_state_cluster=None
):
    """
    Évalue K-means (non supervisé) sur une méthode de description donnée,
    puis mappe chaque cluster à une classe par vote majoritaire.

    Paramètres :
        method (str)            : Nom de la méthode dans `data` ('E34', 'GFD', etc.).
        k_clusters (int)        : Nombre de clusters K-means.
        test_ratio (float)      : Pourcentage de données mises en test.
        random_state (int)      : Graine pour le split train/test.
        aff (bool)              : Si True, affiche les infos et la visualisation.
        random_state_cluster    : Graine pour l'initialisation des centroïdes K-means.

    Retour :
        accuracy (float)        : Taux de bonne classification sur le test.
        cluster_to_class (dict) : Dictionnaire {id_cluster: classe_majoritaire}.
    """
    X = data[method]
    y_true = data['labels']  # étiquettes vraies (ici de 1 à 9)

    # Split train / test
    n = len(X)
    np.random.seed(random_state)
    indices = np.random.permutation(n)

    n_test = int(test_ratio * n)
    train_idx = indices[n_test:]
    test_idx = indices[:n_test]

    X_train, y_train = X[train_idx], y_true[train_idx]
    X_test, y_test = X[test_idx], y_true[test_idx]

    # 1. Clustering non supervisé sur le train
    centroids, train_cluster_labels = kmeans_clustering(
        X_train,
        k=k_clusters,
        random_state=random_state_cluster
    )

    # 2. Mapping cluster → classe réelle par vote majoritaire (sur le train)
    cluster_to_class = {}
    for cluster_id in range(k_clusters):
        mask = (train_cluster_labels == cluster_id)
        if mask.sum() > 0:
            unique_labels, counts = np.unique(y_train[mask], return_counts=True)
            if len(unique_labels) > 0:
                most_common = unique_labels[np.argmax(counts)]
            else:
                most_common = -1
            cluster_to_class[cluster_id] = most_common
        else:
            # Cluster vide
            cluster_to_class[cluster_id] = -1

    # 3. Prédiction des clusters sur le test
    test_cluster_labels = predict_kmeans(X_test, centroids)

    # 4. Conversion clusters → classes prédites
    y_pred = np.array([cluster_to_class.get(c, -1) for c in test_cluster_labels])

    # 5. Accuracy globale
    accuracy = np.mean(y_pred == y_test)

    if aff:
        print(f"{method} + K-means (k={k_clusters}) → Accuracy test = {accuracy:.3f}")
        print(f"   Mapping clusters → classes : {cluster_to_class}")

        # Visualisation des clusters en 2D via PCA (sur le train)
        if X_train.shape[1] > 2:
            X_train_2d = simple_pca(X_train, n_components=2)
        else:
            X_train_2d = X_train

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_train_2d[:, 0],
            X_train_2d[:, 1],
            c=train_cluster_labels,
            cmap='viridis',
            alpha=0.7
        )
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f'Visualisation des clusters K-means (k={k_clusters}) - {method} (PCA 2D)')
        plt.xlabel('Composante Principale 1')
        plt.ylabel('Composante Principale 2')
        plt.grid(True, alpha=0.3)
        plt.show()

    return accuracy, cluster_to_class


# ===================================================================
# Lancement sur toutes les méthodes : recherche du meilleur k
# ===================================================================

methodes = ['E34', 'GFD', 'SA', 'F0', 'F2']
print("\n=== Évaluation K-means (non supervisé) avec vote majoritaire ===\n")

print("Évaluation avec seed_split = 42 et seed_kmeans = 0 :\n")
best_results = {}

for meth in methodes:
    best_acc = -1.0
    best_k = None

    # Recherche du meilleur k entre 2 et 25
    for k in range(2, 26):
        acc, _ = evaluate_kmeans_on_method(
            method=meth,
            k_clusters=k,
            test_ratio=0.2,
            random_state=42,
            random_state_cluster=0
        )
        if acc > best_acc:
            best_acc = acc
            best_k = k

    best_results[meth] = (best_k, best_acc)

    # Affichage du meilleur résultat
    print(f"Meilleur résultat pour {meth} : k={best_k}, Accuracy={best_acc:.3f}")

    # Appel avec aff=True pour afficher les détails + plot
    evaluate_kmeans_on_method(
        method=meth,
        k_clusters=best_k,
        test_ratio=0.2,
        random_state=42,
        aff=True,
        random_state_cluster=0
    )
    print("-" * 70)


# ===================================================================
# Tests de robustesse : plusieurs seeds
# ===================================================================

print("\n=== Évaluation multi-seeds pour robustesse ===\n")
seeds = [(0, 15), (0, 42), (3, 100), (32, 128), (6, 556)]  # (seed_kmeans, seed_split)

results_per_seed = {}

for seed_kmeans, seed_split in seeds:
    print(f"\n→ Split seed = {seed_split} | K-means seed = {seed_kmeans}")
    current_seed_results = {}

    for meth in methodes:
        best_acc = -1.0
        best_k = None

        for k in range(2, 26):
            acc, _ = evaluate_kmeans_on_method(
                method=meth,
                k_clusters=k,
                test_ratio=0.2,
                random_state=seed_split,          # split train/test
                random_state_cluster=seed_kmeans, # init des centroïdes
                aff=False
            )
            if acc > best_acc:
                best_acc = acc
                best_k = k

        current_seed_results[meth] = (best_k, best_acc)
        print(f"   {meth}: k={best_k}, acc={best_acc:.3f}")

    results_per_seed[(seed_split, seed_kmeans)] = current_seed_results


# ===================================================================
# Résumé statistique final
# ===================================================================

print("\n=== Résumé statistique par méthode ===")
for meth in methodes:
    ks = [results_per_seed[sd][meth][0] for sd in results_per_seed]
    accs = [results_per_seed[sd][meth][1] for sd in results_per_seed]

    print(f"\n{meth} : Seeds (seed_split, seed_kmeans) = {seeds}")
    print(f"   Meilleurs k  : {ks}")
    print(f"   Accuracies   : {['{:.3f}'.format(a) for a in accs]}")
    # Si besoin, décommenter pour moyennes et écarts-types :
    # print(f"   Moyenne k    : {np.mean(ks):.1f} ± {np.std(ks):.1f}")
    # print(f"   Moyenne acc
