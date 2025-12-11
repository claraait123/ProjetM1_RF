import numpy as np
from loadBDshape import data


def distance_minkowski(x, y, p):
    """
    Calcule la distance de Minkowski entre deux vecteurs x et y.

    Paramètres :
        x (ndarray) : Premier vecteur.
        y (ndarray) : Deuxième vecteur.
        p (float) : Ordre de la distance (p=1 pour Manhattan, p=2 pour Euclidienne, etc.).

    Retour :
        float : Distance de Minkowski entre x et y.
    """
    diff = np.abs(x - y)
    return np.power(np.sum(diff ** p), 1.0 / p)


def k_plus_proches_voisins_x(x, X_train, Y_train, k=3):
    """
    Prédit la classe d’un échantillon x selon le modèle des k plus proches voisins.

    Paramètres :
        x (ndarray) : Échantillon à classer.
        X_train (ndarray) : Données d’apprentissage.
        Y_train (ndarray) : Étiquettes correspondantes.
        k (int) : Nombre de voisins à considérer.

    Retour :
        La classe majoritaire parmi les k plus proches voisins.
    """
    # Calcul des distances entre x et chaque point de l’ensemble d’apprentissage
    distances = [distance_minkowski(x, x_train_i, 1) for x_train_i in X_train]

    # Récupération des indices des k plus proches voisins
    idx_knn = np.argsort(distances)[:k]

    # Étiquettes des k voisins
    labels_knn = Y_train[idx_knn]

    # Sélection de la classe majoritaire parmi ces voisins
    valeurs, comptes = np.unique(labels_knn, return_counts=True)
    idx_max = np.argmax(comptes)
    return valeurs[idx_max]


def k_plus_proches_voisins(X_train, y_train, X, k=3):
    """
    Applique le modèle des k plus proches voisins à plusieurs échantillons.

    Paramètres :
        X_train (ndarray) : Données d’apprentissage.
        y_train (ndarray) : Étiquettes d’apprentissage.
        X (ndarray) : Données sur lesquelles faire la prédiction.
        k (int) : Nombre de voisins à considérer.

    Retour :
        ndarray : Classes prédites pour chaque échantillon de X.
    """
    predictions = [k_plus_proches_voisins_x(X[i], X_train, y_train, k) for i in range(len(X))]
    return np.array(predictions)


# Méthodes de description de la forme
METHODES = ['E34', 'GFD', 'SA', 'F0', 'F2']
liste_k = [1, 3, 5, 7, 9]

# Boucle principale sur chaque méthode
for methode in METHODES:
    print(f"\n=== Méthode : {methode} ===")

    X = data[methode]
    y = data['labels']

    # Découpage de la base de données en 60 % apprentissage, 20 % validation, 20 % test
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

    # Sélection du meilleur k selon la performance sur la validation
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

    # Évaluation sur la base de test avec le meilleur k trouvé
    y_test_pred = k_plus_proches_voisins(X_train, y_train, X_test, k=meilleur_k)
    acc_test = np.mean(y_test_pred == y_test)
    print(f"Taux de reconnaissance test ({methode}, k={meilleur_k}) = {acc_test:.3f}")
