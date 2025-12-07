import numpy as np
from loadBDshape import data
import matplotlib.pyplot as plt

#not between one data_point and one centroid but between one data_point and n centroids.
def euclidean_distance(X, centroids):
    return np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))

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
    distances = np.sqrt(((X_test[:, np.newaxis] - centroids) ** 2).sum(axis=2))
    return np.argmin(distances, axis=1)


def evaluate_kmeans_on_method(method='E34', k_clusters=9, test_ratio=0.2, random_state=42):
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
    print(f"{method} + K-means (k={k_clusters}) → Accuracy test = {accuracy:.3f}")
    print(f"   Mapping clusters → classes : {cluster_to_class}")

    return accuracy, cluster_to_class



# ===================================================================
# Lancement sur toutes les méthodes
# ===================================================================
if __name__ == "__main__":
    methodes = ['E34', 'GFD', 'SA', 'F0', 'F2']
    print("=== Évaluation K-means (non supervisé) avec vote majoritaire ===\n")
    for meth in methodes:
        evaluate_kmeans_on_method(method=meth, k_clusters=9, test_ratio=0.2, random_state=42)
        print("-" * 70)
