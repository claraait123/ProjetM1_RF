import os
import sys
import numpy as np


# === chemin vers code/ ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(CURRENT_DIR, "..", "code")
sys.path.append(CODE_DIR)


from loadBDshape import data
from kppv import k_plus_proches_voisins

# Méthodes et k optimaux trouvés expérimentalement
METHODES = ['E34', 'GFD', 'SA', 'F0', 'F2']
BEST_K = {
    'E34': 1,
    'GFD': 1,
    'SA': 1,
    'F0': 1,
    'F2': 3
}

# Optionnel : ordre de priorité pour départager en cas d'égalité
PRIORITE_METHODES = ['GFD', 'F0', 'E34', 'SA', 'F2']


def split_indices(n, seed=0):
    """Découpe indices en 60/20/20 pour train/valid/test."""
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)

    n_train = int(0.6 * n)
    n_valid = int(0.2 * n)

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]

    return train_idx, valid_idx, test_idx


def vote_majoritaire(preds_par_methode):
    """
    preds_par_methode : dict {methode: label_prédit}
    Retourne la classe finale par vote majoritaire
    avec priorité en cas d’égalité.
    """
    # Liste des labels prédits
    votes = list(preds_par_methode.values())
    valeurs, comptes = np.unique(votes, return_counts=True)

    # Classes ayant le max de votes
    max_votes = np.max(comptes)
    classes_gagnantes = valeurs[comptes == max_votes]

    if len(classes_gagnantes) == 1:
        return classes_gagnantes[0]

    # Égalité : on départage via la priorité des méthodes
    for methode in PRIORITE_METHODES:
        c = preds_par_methode[methode]
        if c in classes_gagnantes:
            return c

    # Sécurité : si jamais
    return classes_gagnantes[0]


def main():
    y = data['labels']
    n = len(y)

    train_idx, _, test_idx = split_indices(n, seed=0)

    # Préparation des ensembles train/test pour chaque méthode
    X_train = {}
    X_test = {}
    for methode in METHODES:
        X = data[methode]
        X_train[methode] = X[train_idx]
        X_test[methode] = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    # Prédictions par méthode sur la base de test
    preds_test_par_methode = {m: None for m in METHODES}
    for methode in METHODES:
        k = BEST_K[methode]
        print(f"Prédictions kPPV pour {methode} avec k={k}...")
        preds = k_plus_proches_voisins(X_train[methode], y_train,
                                       X_test[methode], k=k)
        preds_test_par_methode[methode] = preds

    # Vote majoritaire pour chaque échantillon de test
    y_pred_vote = []
    for i in range(len(test_idx)):
        preds_i = {m: preds_test_par_methode[m][i] for m in METHODES}
        classe_finale = vote_majoritaire(preds_i)
        y_pred_vote.append(classe_finale)

    y_pred_vote = np.array(y_pred_vote)

    # Taux de reconnaissance global
    acc_vote = np.mean(y_pred_vote == y_test)
    print("\n=== Vote majoritaire sur la base de test ===")
    print(f"Taux de reconnaissance (vote majoritaire) = {acc_vote:.3f}")

    # (Optionnel) afficher quelques exemples
    print("\nExemples (vraie classe / prédiction vote) :")
    for i in range(min(10, len(y_test))):
        print(f"y_true = {y_test[i]}, y_pred = {y_pred_vote[i]}")


if __name__ == "__main__":
    main()
