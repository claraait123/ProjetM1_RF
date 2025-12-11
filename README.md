# ProjetM1_RF
Projet - Reconnaissance des Formes  - M1 Master Informatique – année 2025 / 2026

Ce projet implémente un système de reconnaissance de formes sur la base BDshape (99 formes, 9 classes) décrite par cinq types de descripteurs (E34, GFD, SA, F0, F2). Les données sont chargées en matrices NumPy, puis évaluées avec un classifieur k plus proches voisins (k-PPV) supervisé et une approche k‑means non supervisée. Les métriques (accuracy, matrice de confusion, F1) y sont inclus.

# Dépendances

- Python 3.10
- Bibliothèques :
    - numpy
    - matplotlib
    - scikit-learn

# Arborescence

ProjetRF2025/
│
├── code/
│   ├── loadBDshape.py
│   ├── kppv.py
│   └── kmeans.py
│
├── evaluation/
│   ├── accuracy.py
│   ├── courbePR.py
│   └── critere.py
│
└── BDD/
    ├── E34/
    ├── GFD/
    ├── SA/
    ├── F0/
    └── F2/


## Utilisation

## Classification

- **loadBDshape.py** charge automatiquement les descripteurs en mémoire dans data
- **kppv.py** lance la méthode de classification k-PPV :
    - sépare la base de données 60/20/20
    - teste plusieurs valeurs de k
    - choisi le meilleur k pour chaque méthode
    - mesure le taux de reconnaissance
- **kmeans.py** lance la méthode de classification k-means :
    - demande à l'utilisateur le p voulu pour la distance de Minkowski
    - sépare la base de données 80/20
    - choisi le meilleur k selon le taux de reconnaissance
    - évalue la robustesse via plusieurs seed
    - permet de visualiser les clusters en 2D

## Evaluation

- **accuracy.py** affiche le taux de reconnaissance moyen pour chaque classe
- **courbePR.py** trace la courbe précision-rappel et affiche les AUC pour chaque classe et les AUC moyens
- **criteres.py** :
    - évalue k-PPV pour chaque méthode de descripteur
    - évalue k-means
    - affiche pour chaque méthode la matrice de confusion, le taux de reconnaissance et les scores F1


# Execution

A executer dans le dossier où se trouve le fichier : 

python3 script.py

