import numpy as np
import sys
import os

# Détermination du chemin du dossier courant (où se trouve ce fichier)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ajout du dossier ../code au PYTHONPATH pour pouvoir importer les modules
CODE_DIR = os.path.join(CURRENT_DIR, "..", "classification")
sys.path.append(CODE_DIR)

from loadBDshape import data
from kppv import k_plus_proches_voisins, METHODES, liste_k


def accuracy_par_classe(y_true, y_pred):
    """
    Calcule l'accuracy (taux de bonnes prédictions) pour chaque classe.

    Paramètres :
        y_true (ndarray) : Étiquettes réelles.
        y_pred (ndarray) : Étiquettes prédites.

    Retour :
        dict : Dictionnaire {classe: accuracy_sur_cette_classe}.
               Si une classe n'apparaît pas dans y_true, sa valeur sera np.nan.
    """
    classes = np.unique(y_true)   # Liste des classes présentes dans les vraies étiquettes
    acc_classes = {}

    for c in classes:
        # Indices des échantillons appartenant à la classe c
        idx_c = (y_true == c)
        n_c = np.sum(idx_c)

        if n_c == 0:
            # Aucun exemple de cette classe dans y_true
            acc_classes[c] = np.nan
        else:
            # Accuracy = (nombre de bonnes prédictions sur cette classe) / (nombre total d'exemples de cette classe)
            acc_classes[c] = np.mean(y_pred[idx_c] == y_true[idx_c])

    return acc_classes


if __name__ == "__main__":

    # Listes globales pour agréger toutes les prédictions de toutes les méthodes
    y_test_global = []
    y_pred_global = []

    # Boucle sur chaque méthode de description
    for methode in METHODES:
        print(f"\n=== Méthode : {methode} ===")

        X = data[methode]
        y = data["labels"]

        # Découpage 60 % apprentissage, 20 % validation, 20 % test
        n = len(X)
        indices = np.arange(n)
        np.random.seed(0)          # Pour rendre le découpage reproductible
        np.random.shuffle(indices)

        n_train = int(0.6 * n)
        n_valid = int(0.2 * n)

        train_idx = indices[:n_train]
        valid_idx = indices[n_train:n_train + n_valid]
        test_idx  = indices[n_train + n_valid:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        # Recherche du meilleur k sur la base de validation
        meilleur_k = None
        meilleure_acc = 0.0

        for k in liste_k:
            y_valid_pred = k_plus_proches_voisins(X_train, y_train, X_valid, k=k)
            acc = np.mean(y_valid_pred == y_valid)

            if acc > meilleure_acc:
                meilleure_acc = acc
                meilleur_k = k

        print(f"Meilleur k pour {methode} = {meilleur_k}, acc validation = {meilleure_acc:.3f}")

        # Prédictions sur la base de test avec ce meilleur k
        y_test_pred = k_plus_proches_voisins(X_train, y_train, X_test, k=meilleur_k)

        # Stockage pour calculer les métriques globales (toutes méthodes confondues)
        y_test_global.append(y_test)
        y_pred_global.append(y_test_pred)

    # Concaténation des résultats de toutes les méthodes
    y_test_global = np.concatenate(y_test_global)
    y_pred_global = np.concatenate(y_pred_global)

    # Accuracy moyenne par classe (toutes méthodes confondues)
    acc_classes = accuracy_par_classe(y_test_global, y_pred_global)

    print("\n=== Accuracy moyenne par classe (global, toutes méthodes) ===")
    for c in sorted(acc_classes.keys()):
        print(f"Classe {c} : {acc_classes[c]:.3f}")
