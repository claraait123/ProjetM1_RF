import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# ================================
# CONFIGURATION
# ================================
BASE_DIR = Path("/Users/MariaAydin1/Documents/M1_VMI_25_26/semestre1/Intro_Reconnaissance_Formes/projet")  # Dossier courant, ou remplace par ton chemin

METHODES = ['E34', 'GFD', 'SA', 'F0', 'F2']

METHOD_DIRS = {
    'E34': BASE_DIR / "E34",
    'GFD': BASE_DIR / "GFD",
    'SA':  BASE_DIR / "SA",
    'F0':  BASE_DIR / "F0",
    'F2':  BASE_DIR / "F2"
}


METHODES_DIM = {
    'E34': 16,
    'GFD': 36,
    'SA':  90,
    'F0':  128,
    'F2':  128
}


#def get_all_met_files():
#    """Renvoie la liste de tous les fichiers .MET présents dans les sous-dossiers """
#    return [f for method in METHODES for f in (BDD_DIR / method).glob(f"S??N???.{method}")]



def read_met_file(filepath):
    """Lit un fichier .MET et retourne un vecteur numpy de float."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        values = [float(x) for x in lines]
    
    return np.array(values, dtype=np.float64)


def load_bdshape_data() :

    print("Chargement des descripteurs depuis les dossiers séparés...")

    samples_dict = {}
    
    for method, dir_path in METHOD_DIRS.items() :
        
        print(f"  Lecture {method} depuis {dir_path}...")
        files = sorted([f for f in dir_path.iterdir() if f.is_file()])

        for file_path in files:

            filename = file_path.name
            
            img_id, ext = filename.split('.', 1)

            class_str = img_id[1:3]
            class_id = int(class_str) - 1  # 0 à 8

            vector = read_met_file(file_path)

            if img_id not in samples_dict :
                samples_dict[img_id] = {"id" : img_id, "class" : class_id, "E34" : None, "GFD" : None, "SA" : None, "F0" : None, "F2" : None }
            
            samples_dict[img_id][method] = vector
    

    # Convertir en liste triée
    samples = []
    for img_id in sorted(samples_dict.keys()):
        sample = samples_dict[img_id]
        samples.append(sample)
    
    # === Rapport final ===
    print(f"\nChargement terminé : {len(samples)} échantillons trouvés.")


    # Vérification finale des dimensions
    print("\nVérification des dimensions :")
    for method in METHODES_DIM.keys():
        dims = [len(s[method]) for s in samples]
        print(f"  {method}: {min(dims)}–{max(dims)} valeurs →", 
              "OK" if min(dims) == max(dims) == METHODES_DIM[method] else "ERREUR")

    return samples



# ================================
# 3. Conversion en matrices (pour scikit-learn, etc.)
# ================================
def samples_to_arrays(samples: List[Dict]) -> Dict[str, np.ndarray]:
    """Convertit samples → data (matrices NumPy par méthode)"""
    data = {}
    labels = np.array([s['class'] for s in samples])
    ids = [s['id'] for s in samples]
    
    for method in METHODES_DIM.keys():
        X = np.stack([s[method] for s in samples])  # (99, dim)
        data[method] = X
    
    data['labels'] = labels
    data['ids'] = np.array(ids)
    return data


samples = load_bdshape_data()
data = samples_to_arrays(samples)

print(f"\nExemple : {samples[0]['id']}")
print(f"  Classe : {samples[0]['class']}")
print(f"  E34 (premiers 5) : {samples[0]['E34'][:5]}")

print(f"\nMatrice E34 : {data['E34'].shape} → prêt pour k-NN !")


