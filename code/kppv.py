import numpy as np
from loadBDshape import data

# Choix d'une méthode
X = data['E34']
y = data['labels'] 


# Decoupage de la BDD
n = len(X)
indices = np.arange(n)
np.random.shuffle(indices)

# Découpage 60% / 20% / 20%
n_train = int(0.6 * n)
n_valid = int(0.2 * n)
train_idx = indices[:n_train]
valid_idx = indices[n_train:n_train + n_valid]
test_idx  = indices[n_train + n_valid:]

X_train, y_train = X[train_idx], y[train_idx]
X_valid, y_valid = X[valid_idx], y[valid_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

def distance_euclidienne(x, y):
    diff = x - y
    return np.sqrt(np.sum(diff ** 2))

def k_plus_proches_voisins_x(x, X_train, Y_train, k=3):
    distances = np.array([distance_euclidienne(x, x_train_i) for x_train_i in X_train])
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


liste_k = [1, 3, 5, 7, 9]
meilleur_k = None
meilleure_acc = 0.0

for k in liste_k:
    y_valid_pred = k_plus_proches_voisins(X_train, y_train, X_valid, k=k)
    acc = np.mean(y_valid_pred == y_valid)
    print(f"k = {k}, taux de reconnaissance validation (E34) = {acc:.3f}")
    if acc > meilleure_acc:
        meilleure_acc = acc
        meilleur_k = k

print(f"Meilleur k (E34) sur validation = {meilleur_k}, acc = {meilleure_acc:.3f}")