import numpy as np
from loadBDshape import data
import matplotlib.pyplot as plt

#not between one data_point and one centroid but between one data_point and n centroids.
def euclidean_distance(X, centroids):
    return np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))

def manhattan_distance(X, centroids):
    return (np.abs((X[:, np.newaxis] - centroids))).sum(axis=2)

def minkowski_distance(X, centroids, p=5):
    return (np.abs(X[:, np.newaxis] - centroids) ** p).sum(axis=2) ** (1 / p)

def kmeans_clustering(X, k=9, max_it=200, random_state=None) :

    if random_state is not None :
        np.random.seed(random_state)
    
    #min_values = X.min(axis=0)
    #max_values = X.max(axis=0)
    centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(k, X.shape[1])) #shape of data point is the centroid.

    for it in range(max_it) : 
        
        distances = euclidean_distance(X, centroids)
        labels = np.argmin(distances, axis=1) #cluster numéro


        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            if np.sum(labels == i) > 0:
                new_centroids[i] = X[labels == i].mean(axis=0)
            else:
                new_centroids[i] = centroids[i]  # cluster vide → on garde l'ancien

        # 3. Convergence ?
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break

        centroids = new_centroids

    return centroids, labels


def predict_kmeans(X_test, centroids):
    """Prédit le cluster le plus proche pour de nouvelles données"""
    distances = euclidean_distance(X_test, centroids)
    return np.argmin(distances, axis=1)




def simple_pca(X, n_components=2):
    """Réduction de dimension simple via PCA avec SVD (pour visualisation 2D)"""
    # Centrage des données
    X_centered = X - np.mean(X, axis=0)
    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Projection sur les n premiers composants
    components = Vt[:n_components]
    X_reduced = np.dot(X_centered, components.T)
    return X_reduced



def evaluate_kmeans_on_method(method='E34', k_clusters=9, test_ratio=0.2, random_state=42, aff=False):
    X = data[method]
    y_true = data['labels']          # maintenant 1 à 9

    n = len(X)
    np.random.seed(random_state)
    indices = np.random.permutation(n)

    n_test = int(test_ratio * n)
    train_idx = indices[n_test:]
    test_idx  = indices[:n_test]

    X_train, y_train = X[train_idx], y_true[train_idx]
    X_test,  y_test  = X[test_idx],  y_true[test_idx]

    # 1. Clustering non supervisé sur le train
    centroids, train_cluster_labels = kmeans_clustering(X_train, k=k_clusters, random_state=0)

    # 2. Mapping cluster → classe réelle par vote majoritaire (sur le train)
    cluster_to_class = {}
    for cluster_id in range(k_clusters):
        mask = (train_cluster_labels == cluster_id)
        if mask.sum() > 0:
            #unique, counts = np.unique(y_train[mask], return_counts=True)
            #cluster_to_class[cluster_id] = unique[np.argmax(counts)]
            unique_labels, counts = np.unique(y_train[mask], return_counts=True)
            if len(unique_labels) > 0:
                most_common = unique_labels[np.argmax(counts)]
            else:
                most_common = -1
            cluster_to_class[cluster_id] = most_common
        else:
            cluster_to_class[cluster_id] = -1   # cluster vide

    # 3. Prédiction sur le test
    test_cluster_labels = predict_kmeans(X_test, centroids)

    # 4. Conversion en classes prédites
    y_pred = np.array([cluster_to_class.get(c, -1) for c in test_cluster_labels])

    # 5. Accuracy
    accuracy = np.mean(y_pred == y_test)
    if aff :
        print(f"{method} + K-means (k={k_clusters}) → Accuracy test = {accuracy:.3f}")
        print(f"   Mapping clusters → classes : {cluster_to_class}")




    if aff :
        # --- Visualisation des clusters en 2D via PCA ---
        if X_train.shape[1] > 2:  # Réduction si dim > 2
            X_train_2d = simple_pca(X_train, n_components=2)
        else:
            X_train_2d = X_train  # Si déjà 2D ou moins

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=train_cluster_labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f'Visualisation des clusters K-means (k={k_clusters}) - {method} (PCA 2D)')
        plt.xlabel('Composante Principale 1')
        plt.ylabel('Composante Principale 2')
        plt.grid(True, alpha=0.3)
        plt.show()

    return accuracy, cluster_to_class



# ===================================================================
# Lancement sur toutes les méthodes
# ===================================================================
"""methodes = ['E34', 'GFD', 'SA', 'F0', 'F2']
print("=== Évaluation K-means (non supervisé) avec vote majoritaire ===\n")
for meth in methodes:
    evaluate_kmeans_on_method(method=meth, k_clusters=15, test_ratio=0.2, random_state=42)
    print("-" * 70)
"""

methodes = ['E34', 'GFD', 'SA', 'F0', 'F2']
print("=== Évaluation K-means (non supervisé) avec vote majoritaire ===\n")

best_results = {}
for meth in methodes:
    best_acc = -1
    best_k = None
    for k in range(2, 26):  # de 2 à 25
        acc, _ = evaluate_kmeans_on_method(method=meth, k_clusters=k, test_ratio=0.2, random_state=42)
        if acc > best_acc:
            best_acc = acc
            best_k = k
    best_results[meth] = (best_k, best_acc)
    
    # Affichage du meilleur
    print(f"Meilleur résultat pour {meth}: k={best_k}, Accuracy={best_acc:.3f}")
    # Appel pour afficher le print et le plot du meilleur
    evaluate_kmeans_on_method(method=meth, k_clusters=best_k, test_ratio=0.2, random_state=42, aff=True)
    print("-" * 70)



