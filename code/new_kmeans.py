# fichier : kmeans.py  (ou à coller dans ton fichier principal)

import numpy as np
from loadBDshape import data
import matplotlib.pyplot as plt


class KMeans:

    def __init__(self, k=9, max_iters=200, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None  # labels des clusters pour les données d'entraînement

    #not between one data_point and one centroid but between one data_point and n centroids.
    def _euclidean_distance(self, X, centroids):
        return np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        #d'abord initialise aléatoirement les centroids.
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)
        self.centroids = np.random.uniform(min_vals, max_vals, size=(self.k, X.shape[1])) #shape of data point is the centroid.
        #keep the centroids between the boundaries of the data.

        for _ in range(self.max_iters):
            #va créer le cluster.
            distances = self._euclidean_distance(X, self.centroids) #distances entre chaque data_points et les centroids.
            new_labels = np.argmin(distances, axis=1) #index of the smallest value

            # Recalcul des centroids
            new_centroids = np.array([
                X[new_labels == i].mean(axis=0) if np.sum(new_labels == i) > 0 else self.centroids[i]
                for i in range(self.k)
            ])

            # Critère de convergence
            if np.allclose(self.centroids, new_centroids, atol=1e-6):
                break

            self.centroids = new_centroids
            self.labels_ = new_labels

        return self

    def predict(self, X):
        distances = self._euclidean_distance(X, self.centroids)
        return np.argmin(distances, axis=1)


# ===================================================================
# Utilisation dans le contexte du projet BDshape (comme ton KNN)
# ===================================================================

def evaluate_kmeans_on_method(method='E34', k_clusters=9, test_ratio=0.2, random_state=42):
    X = data[method]       # ex: E34 → shape (99, 16)
    y_true = data['labels']  # vraies classes (0 à 8)

    n = len(X)
    np.random.seed(random_state)
    indices = np.random.permutation(n)

    n_test = int(test_ratio * n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train, y_train = X[train_idx], y_true[train_idx]
    X_test,  y_test  = X[test_idx],  y_true[test_idx]

    # 1. Apprentissage non supervisé sur le train
    kmeans = KMeans(k=k_clusters, max_iters=300, random_state=0)
    kmeans.fit(X_train)

    # 2. Prédiction sur le test
    y_pred_clusters = kmeans.predict(X_test)

    # 3. Association cluster → classe réelle par vote majoritaire (sur le train !)
    cluster_to_class = {}
    for cluster_id in range(k_clusters):
        mask = (kmeans.labels_ == cluster_id)
        if mask.sum() > 0:
            #most_common = np.bincount(y_train[mask]).argmax()
            #cluster_to_class[cluster_id] = most_common

            unique_labels, counts = np.unique(y_train[mask], return_counts=True)
            if len(unique_labels) > 0:
                most_common = unique_labels[np.argmax(counts)]
            else:
                most_common = -1
            cluster_to_class[cluster_id] = most_common

        else:
            cluster_to_class[cluster_id] = -1  # cluster vide → on ignorera

    # 4. Conversion des prédictions en classes réelles
    y_pred = np.array([cluster_to_class.get(c, -1) for c in y_pred_clusters])

    # 5. Calcul du taux de bonne classification
    accuracy = np.mean(y_pred == y_test)
    print(f"{method} + K-means (k={k_clusters}) → Accuracy test = {accuracy:.3f}")

    return accuracy, kmeans, cluster_to_class


# ===================================================================
# Exemple d'exécution pour toutes les méthodes
# ===================================================================

if __name__ == "__main__":
    methodes = ['E34', 'GFD', 'SA', 'F0', 'F2']
    
    print("=== Évaluation K-means (non supervisé) avec association par vote majoritaire ===\n")
    for meth in methodes:
        acc, model, mapping = evaluate_kmeans_on_method(method=meth, k_clusters=9, test_ratio=0.2)
        print(f"   Mapping clusters → classes : {mapping}")
        print("-" * 60)
