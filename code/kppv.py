import numpy as np
from loadBDshape import data

def distance_euclidienne(x, y):
    diff = x - y
    return np.sqrt(np.sum(diff ** 2))

def distance_manhattan(x, y):
    diff = x - y
    return np.sum(np.abs(diff))

def distance_minkowski(x, y, p=2):
    diff = np.abs(x - y)
    return np.power(np.sum(diff ** p), 1.0 / p)

def k_plus_proches_voisins_x(x, X_train, Y_train, k=3):
    distances = []
    for x_train_i in X_train:
        distances.append(distance_minkowski(x, x_train_i))
    idx_knn = np.argsort(distances)[:k]
    labels_knn = Y_train[idx_knn]

    # classe majoritaire
    valeurs, comptes = np.unique(labels_knn, return_counts=True)
    idx_max = np.argmax(comptes)
    return valeurs[idx_max]

def k_plus_proches_voisins(X_train, y_train, X, k=3):
    predictions = []
    for i in range(len(X)):
        predictions.append(k_plus_proches_voisins_x(X[i], X_train, y_train, k))
    return np.array(predictions)


METHODES = ['E34', 'GFD', 'SA', 'F0', 'F2']
liste_k = [1, 3, 5, 7, 9]


for methode in METHODES:
    print("\n=== Méthode :", methode, "===")
    X = data[methode]
    y = data['labels']

    # Decoupage de la BDD : 60 / 20 / 20 
    n = len(X)
    indices = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(indices)

    n_train = int(0.6 * n)
    n_valid = int(0.2 * n)

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx  = indices[n_train + n_valid:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    meilleur_k = None
    meilleure_acc = 0.0

    for k in liste_k:
        y_valid_pred = k_plus_proches_voisins(X_train, y_train, X_valid, k=k)
        acc = np.mean(y_valid_pred == y_valid)
        print(f"k = {k}, taux validation ({methode}) = {acc:.3f}")
        if acc > meilleure_acc:
            meilleure_acc = acc
            meilleur_k = k

    print(f"Meilleur k pour {methode} = {meilleur_k}, acc validation = {meilleure_acc:.3f}")

    # Évaluation sur la base de test
    y_test_pred = k_plus_proches_voisins(X_train, y_train, X_test, k=meilleur_k)
    acc_test = np.mean(y_test_pred == y_test)
    print(f"Taux de reconnaissance test ({methode}, k={meilleur_k}) = {acc_test:.3f}")