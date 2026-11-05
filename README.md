# ⚙️ MachinePulse — Prédiction des pannes industrielles par Machine Learning

**MachinePulse** est un projet de Machine Learning consacré à la maintenance prédictive des équipements industriels.

L'objectif est de développer un modèle capable d'estimer le risque de défaillance d'une machine à partir de ses caractéristiques de fonctionnement, puis de rendre ce modèle accessible à travers une application web interactive développée avec Streamlit.

🌐 **[Tester l'application MachinePulse](https://machinepulse-b5rqh7rmfculrvwuxakv3q.streamlit.app/)**

## 1. Contexte et problématique

Dans l'industrie, les pannes imprévues peuvent entraîner des interruptions de production, des coûts de maintenance importants et une diminution de la disponibilité des équipements.

La maintenance prédictive consiste à exploiter les données de fonctionnement des machines pour identifier les situations présentant un risque de défaillance.

**Problématique :** Peut-on prédire une panne industrielle à partir de mesures telles que la température, la vitesse de rotation, le couple mécanique et l'usure des outils ?

## 2. Données utilisées

Le projet utilise le jeu de données **AI4I 2020 Predictive Maintenance Dataset**, disponible sur l'UCI Machine Learning Repository.

Il contient **10 000 observations** décrivant le fonctionnement d'équipements industriels.

Les six variables explicatives utilisées sont :

| Variable | Description |
|---|---|
| Type | Catégorie du produit (L, M ou H) |
| Air temperature | Température de l'air (K) |
| Process temperature | Température du processus (K) |
| Rotational speed | Vitesse de rotation (tr/min) |
| Torque | Couple mécanique (Nm) |
| Tool wear | Usure de l'outil (min) |

**Variable cible : `Machine failure`**

- `0` : fonctionnement normal
- `1` : défaillance de la machine

### Répartition des classes

| Classe | Observations | Proportion |
|---|---:|---:|
| Fonctionnement normal | 9 661 | 96,61 % |
| Défaillance | 339 | 3,39 % |

Le jeu de données est fortement déséquilibré. L'évaluation ne peut donc pas reposer uniquement sur l'accuracy.

## 3. Méthodologie

Le projet suit les étapes suivantes :

**Préparation des données**

- Séparation des variables explicatives et de la variable cible.
- Division en ensembles d'entraînement (80 %) et de test (20 %), avec stratification.
- Standardisation des variables numériques avec `StandardScaler`.
- Encodage de la variable catégorielle avec `OneHotEncoder`.
- Intégration des transformations et du modèle dans un `Pipeline` scikit-learn.

**Entraînement et comparaison**

Trois algorithmes de classification ont été étudiés :

- Régression logistique
- Random Forest
- XGBoost

Les modèles ont été comparés avec une validation croisée stratifiée à cinq plis (`StratifiedKFold`), en utilisant le **F1-score** comme métrique principale.

**Optimisation**

Le modèle retenu a ensuite été optimisé avec `GridSearchCV` afin de sélectionner ses hyperparamètres.

## 4. Résultats

### Comparaison des modèles

| Modèle | F1-score moyen en validation croisée |
|---|---:|
| Random Forest | **0,589** |
| XGBoost | 0,559 |
| Régression logistique | 0,235 |

Le **Random Forest** obtient le meilleur F1-score moyen en validation croisée parmi les trois modèles évalués.

### Performances sur les données de test

Après optimisation, le modèle Random Forest a été évalué sur 2 000 observations non utilisées pour son entraînement.

| Métrique | Résultat |
|---|---:|
| Accuracy | 98 % |
| Précision — classe panne | 0,89 |
| Rappel — classe panne | 0,49 |
| F1-score — classe panne | 0,63 |

**Matrice de confusion :**

```text
                    Prédiction
                   Normal  Panne
Réel Normal          1928      4
Réel Panne             35     33
```

Sur 68 pannes réelles dans l'ensemble de test, le modèle en détecte correctement 33 et en manque 35.

Ces résultats illustrent une limite importante : malgré une accuracy élevée, le rappel de la classe minoritaire reste modéré. Dans un contexte industriel, le coût des pannes non détectées doit être pris en compte avant toute utilisation opérationnelle.

## 5. Application web MachinePulse

Le modèle sélectionné est sauvegardé avec `joblib` puis intégré dans une application développée avec **Streamlit**.

L'application permet de :

- Renseigner les six caractéristiques de fonctionnement d'une machine.
- Exécuter une prédiction à partir du modèle entraîné.
- Afficher le résultat de classification : fonctionnement normal ou risque de panne.
- Consulter les probabilités estimées par le modèle.

L'interface constitue un démonstrateur interactif de maintenance prédictive. Elle ne remplace pas un système industriel de surveillance validé en conditions réelles.

**Application en ligne :**  
https://machinepulse-b5rqh7rmfculrvwuxakv3q.streamlit.app/

## 6. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python | Développement du projet |
| Pandas | Manipulation des données |
| scikit-learn | Prétraitement, pipelines, modèles et évaluation |
| XGBoost | Modèle de classification comparatif |
| Joblib | Sérialisation du modèle |
| Streamlit | Interface web interactive |
| Git et GitHub | Versionnement et hébergement du code |
| Streamlit Community Cloud | Déploiement de l'application |

## 7. Structure du projet

Les fichiers de l'application sont actuellement organisés dans le dossier `M2/ML_Panne/streamlit_app/` du dépôt GitHub.

```text
machinepulse/
└── M2/
    └── ML_Panne/
        └── streamlit_app/
            ├── app.py
            ├── modele_maintenance.joblib
            ├── requirements.txt
            └── README.md
```

- `app.py` : interface Streamlit et prédictions.
- `modele_maintenance.joblib` : pipeline de Machine Learning entraîné.
- `requirements.txt` : dépendances Python.
- `README.md` : documentation du projet.

## 8. Exécution en local

Cloner le dépôt :

```bash
git clone https://github.com/kami-koba/machinepulse.git
cd machinepulse/M2/ML_Panne/streamlit_app
```

Créer un environnement virtuel avec Python 3.11 :

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer l'application :

```bash
streamlit run app.py
```

L'application sera accessible depuis le navigateur à l'adresse indiquée par Streamlit, généralement `http://localhost:8501`.

## 9. Limites et perspectives

Ce projet constitue une première approche de la maintenance prédictive à partir de données tabulaires.

Plusieurs améliorations sont envisageables :

- Optimiser le seuil de classification selon le coût des faux négatifs et des faux positifs.
- Étudier des méthodes adaptées au déséquilibre des classes.
- Analyser l'importance des variables explicatives.
- Ajouter des outils d'interprétabilité tels que SHAP.
- Évaluer la robustesse du modèle sur de nouvelles données industrielles.
- Développer une chaîne automatisée d'entraînement, de suivi et de déploiement du modèle.

## 10. Ressources

- **Dataset :** [UCI — AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Code source :** [GitHub — MachinePulse](https://github.com/kami-koba/machinepulse)
- **Démonstration :** [MachinePulse sur Streamlit](https://machinepulse-b5rqh7rmfculrvwuxakv3q.streamlit.app/)

---

**MachinePulse — Projet personnel de Machine Learning et de déploiement d'une application de maintenance prédictive.**
