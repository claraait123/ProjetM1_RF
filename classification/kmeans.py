import numpy as np
import matplotlib.pyplot as plt

from loadBDshape import data

#L'utilisateur choisit la distance qu'il souhaite utiliser pour k-means.
p_select = int(input("\nChoisissez l'ordre p de la distance de Minkowski : "))

print(f"Distance de Minkowski sélectionnée : p = {p_select}")
if p_select == 1:
    print("  → Distance de Manhattan")
elif p_select == 2:
    print("  → Distance euclidienne (classique)")
else : 
    print(f"  → Distance de Minkowski d'ordre {p_select}")

P_MINKOWSKI = p_select

def minkowski_distance(X, centroids, p=P_MINKOWSKI):
    """
    Calcule la distance de Minkowski d'ordre p entre chaque point de X 
    et chaque centroïde. Ici, on a pris p=2 par défaut pour la distance euclidienne.

    Paramètres :
        X         : (n_samples, n_features) – les données
        centroids : (k, n_features)         – les centroïdes
        p         : ordre de la norme Minkowski (par défaut 5 dans le code,
                    mais utilisé avec p=2 pour la distance euclidienne)

    Retour :
        distances : (n_samples, k) – matrice des distances
    """
    return (np.abs(X[:, np.newaxis] - centroids) ** p).sum(axis=2) ** (1 / p)

def kmeans_clustering(X, k=9, max_it=200, random_state=None):
    """
    Implémentation de k-means en utilisant la distance de Minkowski (avec p=2 par défaut : distance euclidienne).
    Le nb de clusters k est fixé à 9 par défaut (car 9 classes).

    Paramètres :
        X (ndarray)        : Données (n_samples, n_features).
        k (int)            : Nombre de clusters.
        max_it (int)       : Nombre maximal d'itérations.
        random_state (int) : Seed aléatoire pour l'initialisation des centroïdes.

    Retour :
        centroids (ndarray) : Centroïdes finaux (k, n_features).
        labels (ndarray)    : Indices de clusters pour chaque point de X (n_samples,).
    """

    if random_state is not None:
        np.random.seed(random_state) #on fixe la seed pour mieux retroyver les résultats

    #initialisation aléatoire des centroïdes. Garde les centroids dans les limites des données.
    centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(k, X.shape[1]))

    #va créer les clusters.
    for _ in range(max_it):

        #Va assigner chaque point au centroïde le plus proche
        distances = minkowski_distance(X, centroids) #p=P_MINKOWSKI entré par l'utilisateur précédemment.
        labels = np.argmin(distances, axis=1) #prend l'index de la plus petite valeur.
        #quel centroid a la plus petite distance avec le point

        #mets à jour les centroïdes et les repositionne.
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = (labels == i)
            if np.any(mask):
                new_centroids[i] = X[mask].mean(axis=0) #prend la moyenne
            else:
                #si le cluster est vide, on garde l'ancien centroïde.
                new_centroids[i] = centroids[i]

        #on teste la convergence
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break

        centroids = new_centroids

    return centroids, labels



def predict_kmeans(X_test, centroids):
    """
    Assigne chaque point de X_test au cluster le plus proche (pour k-means déjà entraîné).
    Sert au mapping des clusters.

    Paramètres :
        X_test (ndarray)    : Données à clusteriser (n_samples, n_features).
        centroids (ndarray) : Centroïdes appris (k, n_features).

    Retour :
        ndarray : Labels de clusters (n_samples,).
    """

    distances = minkowski_distance(X_test, centroids)
    return np.argmin(distances, axis=1)



def simple_pca(X, n_components=2):
    """
    Réduction de dimension via PCA (SVD) pour visualisation.
    Cette méthode a été fournie par Grok.

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


def evaluate_kmeans_on_method(method='E34', k_clusters=9, test_ratio=0.2, random_state=42, aff=False, random_state_cluster=None):
    """
    Évalue k-means sur une méthode de description donnée, puis va effectuer du mapping de cluster.
    Cette méthode va mapper (donc assigner) chaque cluster à une classe par vote majoritaire.

    Paramètres :
        method (str)            : Nom de la méthode dans `data` ('E34', 'GFD', etc.).
        k_clusters (int)        : Nombre de clusters K-means.
        test_ratio (float)      : Pourcentage de données pour la bdd de tests (validation). Sert à découper la bdd
        random_state (int)      : Graine pour le split train/test.
        aff (bool)              : Si True, affiche les infos et la visualisation. Évite l'affichage inutile de certains résultats pour les tests.
        random_state_cluster    : Seed pour l'initialisation des centroïdes dans kmeans_clustering.

    Retour :
        Taux de reconnaissance (float) : Taux de bonne classification sur le test.
        cluster_to_class (dict)        : Dictionnaire {id_cluster: classe_majoritaire}.
    """

    X = data[method]
    y_true = data['labels']  # étiquettes vraies (ici de 1 à 9)

    #Split la bdd en train / test (validation)
    #Split en 80-20
    n = len(X)
    np.random.seed(random_state)
    indices = np.random.permutation(n)

    n_test = int(test_ratio * n)
    train_idx = indices[n_test:]
    test_idx = indices[:n_test]

    X_train, y_train = X[train_idx], y_true[train_idx]
    X_test, y_test = X[test_idx], y_true[test_idx]

    #On applique k-means sur la bdd train
    centroids, train_cluster_labels = kmeans_clustering(
        X_train,
        k=k_clusters,
        random_state=random_state_cluster
    )

    #On effectue le mapping de cluster -> classe réelle par vote majoritaire (sur le train)
    cluster_to_class = {}
    for cluster_id in range(k_clusters):
        mask = (train_cluster_labels == cluster_id)
        if mask.sum() > 0:
            unique_labels, counts = np.unique(y_train[mask], return_counts=True)
            if len(unique_labels) > 0:
                most_common = unique_labels[np.argmax(counts)]
            else:
                most_common = -1
            cluster_to_class[cluster_id] = most_common #donne la classe dominante au cluster
        else:
            #quand le cluster est vide, on met -1
            cluster_to_class[cluster_id] = -1

    #on prédit les clusters sur le test (validation). Pas sur le train !
    test_cluster_labels = predict_kmeans(X_test, centroids)

    # Assigne au clusters les classes prédites.
    y_pred = np.array([cluster_to_class.get(c, -1) for c in test_cluster_labels])

    # Calcul le taux de reconnaissance global.
    accuracy = np.mean(y_pred == y_test)

    if aff:
        
        #### Affichage avec matplotlib des nuages de points généré par Grok.

        print(f"{method} + K-means (k={k_clusters}) → Taux de reconnaissance test = {accuracy:.3f}")
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



####################################################################
# Lancement sur toutes les méthodes : recherche du meilleur k pour chaque méthode
####################################################################

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
    print(f"Meilleur résultat pour {meth} : k={best_k}, Taux de reconnaissance={best_acc:.3f}")

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


####################################################################
# Tests sur plusieurs seeds
####################################################################

print("\n=== Évaluation multi-seeds pour robustesse ===\n")
seeds = [(0, 15), (0, 42), (3, 100), (32, 128), (6, 556)]  # (seed_kmeans, seed_split_bdd)

results_per_seed = {}

#seed_kmeans pour random_state_cluster (initialise les centroïdes)
#seed_split pour random_state (pour le split train/test)
for seed_kmeans, seed_split in seeds:
    print(f"\n Split seed = {seed_split} | K-means seed = {seed_kmeans}")
    current_seed_results = {}

    for meth in methodes:
        best_acc = -1.0
        best_k = None

        for k in range(2, 26):
            acc, _ = evaluate_kmeans_on_method(method=meth, k_clusters=k, test_ratio=0.2, random_state=seed_split, random_state_cluster=seed_kmeans, aff=False)
            if acc > best_acc:
                #meilleur k selon meilleur taux de reconnaissance
                best_acc = acc
                best_k = k

        current_seed_results[meth] = (best_k, best_acc)
        print(f"   {meth}: k={best_k}, acc={best_acc:.3f}")

    results_per_seed[(seed_split, seed_kmeans)] = current_seed_results


# ===================================================================
# Résumé des résultats pour le test sur les seeds.
# ===================================================================

print("\n=== Résumé statistique par méthode ===")
for meth in methodes:
    ks = [results_per_seed[sd][meth][0] for sd in results_per_seed]
    accs = [results_per_seed[sd][meth][1] for sd in results_per_seed]

    print(f"\n{meth} : Seeds (seed_split, seed_kmeans) = {seeds}")
    print(f"   Meilleurs k  : {ks}")
    print(f"   Accuracies   : {['{:.3f}'.format(a) for a in accs]}")
