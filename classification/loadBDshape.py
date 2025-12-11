import numpy as np
from pathlib import Path
from typing import List, Dict

# ================================
# CONFIGURATION
# ================================

# Dossier contenant ce fichier
THIS_DIR = Path(__file__).resolve().parent
# Dossier racine où se trouvent les sous-dossiers de descripteurs
BASE_DIR = THIS_DIR.parent / "BDD"

# Noms des méthodes de description utilisées
METHODES = ['E34', 'GFD', 'SA', 'F0', 'F2']

# Dossiers associés à chaque méthode
METHOD_DIRS = {
    'E34': BASE_DIR / "E34",
    'GFD': BASE_DIR / "GFD",
    'SA':  BASE_DIR / "SA",
    'F0':  BASE_DIR / "F0",
    'F2':  BASE_DIR / "F2",
}

# Dimensions attendues pour les descripteurs de chaque méthode
METHODES_DIM = {
    'E34': 16,
    'GFD': 100,
    'SA':  90,
    'F0':  128,
    'F2':  128,
}


def read_met_file(filepath: Path) -> np.ndarray:
    """
    Lit un fichier .MET et retourne un vecteur NumPy de flottants.

    Paramètres :
        filepath (Path) : Chemin du fichier .MET.

    Retour :
        np.ndarray : Vecteur de descripteurs (float64).
    """
    with filepath.open('r', encoding='utf-8') as f:
        # On enlève les lignes vides et on convertit chaque ligne en float
        lines = [line.strip() for line in f if line.strip()]
        values = [float(x) for x in lines]

    return np.array(values, dtype=np.float64)


def load_bdshape_data() -> List[Dict]:
    """
    Charge tous les descripteurs BDshape depuis les dossiers METHOD_DIRS.

    Pour chaque image, on construit une entrée de la forme :
        {
            'id'    : str  (ex. "S01N001"),
            'class' : int  (1..9),
            'E34'   : np.ndarray,
            'GFD'   : np.ndarray,
            'SA'    : np.ndarray,
            'F0'    : np.ndarray,
            'F2'    : np.ndarray
        }

    Retour :
        List[Dict] : Liste d'échantillons (un dictionnaire par image).
    """
    print("Chargement des descripteurs depuis les dossiers séparés...")

    # Dictionnaire temporaire indexé par l'id d'image
    samples_dict: Dict[str, Dict] = {}

    # Parcours de chaque méthode et de son dossier
    for method, dir_path in METHOD_DIRS.items():
        print(f"  Lecture {method} depuis {dir_path}...")
        # On récupère uniquement les fichiers (évite les sous-dossiers)
        files = sorted([f for f in dir_path.iterdir() if f.is_file()])

        for file_path in files:
            filename = file_path.name

            # Exemple : "S01N001.E34" → img_id = "S01N001"
            img_id, _ = filename.split('.', 1)

            # La classe est encodée dans les caractères 1:3 (ex. "01" → 1)
            class_str = img_id[1:3]
            class_id = int(class_str)  # classes 1 à 9

            # Lecture du fichier .MET en vecteur NumPy
            vector = read_met_file(file_path)

            # Initialisation du dictionnaire pour cette image si besoin
            if img_id not in samples_dict:
                samples_dict[img_id] = {
                    "id": img_id,
                    "class": class_id,
                    "E34": None,
                    "GFD": None,
                    "SA":  None,
                    "F0":  None,
                    "F2":  None,
                }

            # Stockage du descripteur pour la méthode courante
            samples_dict[img_id][method] = vector

    # Conversion en liste triée par identifiant pour avoir un ordre stable
    samples: List[Dict] = []
    for img_id in sorted(samples_dict.keys()):
        samples.append(samples_dict[img_id])

    # Rapport final
    print(f"\nChargement terminé : {len(samples)} échantillons trouvés.")

    # Vérification de la cohérence des dimensions pour chaque méthode
    print("\nVérification des dimensions :")
    for method, dim_attendu in METHODES_DIM.items():
        dims = [len(s[method]) for s in samples]
        min_dim, max_dim = min(dims), max(dims)
        etat = "OK" if (min_dim == max_dim == dim_attendu) else "ERREUR"
        print(f"  {method}: {min_dim}–{max_dim} valeurs → {etat}")

    return samples


# ================================
# Conversion en matrices
# ================================

def samples_to_arrays(samples: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Convertit une liste de samples en dictionnaire de matrices NumPy.

    Paramètres :
        samples (List[Dict]) : Liste d'échantillons produite par load_bdshape_data().

    Retour :
        Dict[str, np.ndarray] :
            - data[method] : matrice (n_samples, dim_method),
            - data['labels'] : vecteur des classes (n_samples,),
            - data['ids']    : vecteur des identifiants (n_samples,).
    """
    data: Dict[str, np.ndarray] = {}

    # Étiquettes de classes et identifiants
    labels = np.array([s['class'] for s in samples])
    ids = np.array([s['id'] for s in samples])

    # Matrices de descripteurs pour chaque méthode
    for method in METHODES_DIM.keys():
        # np.stack empile les vecteurs en une matrice (n_samples, dim)
        X = np.stack([s[method] for s in samples])
        data[method] = X

    data['labels'] = labels
    data['ids'] = ids
    return data


# ================================
# Exemple d'utilisation directe
# ================================

samples = load_bdshape_data()
data = samples_to_arrays(samples)

print(f"\nExemple : id : {samples[0]['id']}")
print(f"  Classe : {samples[0]['class']}")
print(f"  E34 (5 premiers) : {samples[0]['E34'][:5]}")
print(f"  GFD (5 premiers) : {samples[0]['GFD'][:5]}")
print(f"  SA  (5 premiers) : {samples[0]['SA'][:5]}")
print(f"  F0  (5 premiers) : {samples[0]['F0'][:5]}")
print(f"  F2  (5 premiers) : {samples[0]['F2'][:5]}")
