import numpy as np
from pathlib import Path

BDD_DIR = Path("BDD")

METHODES = ['E34', 'GFD', 'SA', 'F0', 'F2']

METHODES_DIM = {
    'E34': 16,
    'GFD': 36,
    'SA':  90,
    'F0':  128,
    'F2':  128
}


def get_all_met_files():
    """Renvoie la liste de tous les fichiers .MET présents dans les sous-dossiers """
    return [f for method in METHODES for f in (BDD_DIR / method).glob(f"S??N???.{method}")]

def read_met_file(filepath):
    """Lit un fichier .MET et retourne un vecteur numpy de float."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        values = [float(x) for x in lines]
    
    return np.array(values, dtype=np.float64)
    

