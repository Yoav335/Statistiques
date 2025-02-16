# Présentation du Projet

Ce projet est un projet étudiant répondant aux exigences du cours de Statistique. Il vise à analyser les facteurs influençant le taux d'insertion des étudiants diplômés du supérieur grace à diverses methodes, modèles explicatifs et tests statistiques.

L'application explore l'impact de variables socio-économiques, telles que le taux de chômage et la qualité des emplois disponibles, sur l'employabilité des diplômés 18 mois après l'obtention de leur diplôme.  

L'analyse utilise des techniques de modélisation statistique, notamment : 
- la **régression linéaire**
- **XGBoost**
- Des **méthodes d'interprétabilité comme SHAP**, pour identifier les facteurs clés et leurs contributions à l'insertion professionnelle. 
- Une **analyse de séries temporelles avec ARIMA** est effectuée pour prévoir les taux d'insertion futurs et comparer les résultats avec la régression linéaire.


## Etapes d'installation et lancement de l'application


### Téléchargement des requirements

1. Lancer cette commande pour installer les package necessaires

```
pip install -r requirements.txt
```

### Traitement des données

2. Exécutez le script data_process.py pour nettoyer et préparer les données brutes:

```
python data_process.py
```

Cela créera un nouveau fichier CSV cleaned_data.csv contenant les données nettoyées.

### Exécution de l'application Streamlit

3. Lancez l'application Streamlit pour interagir avec les résultats et les visualisations :

```
streamlit run Streamlit.py
```


## Structure des Fichiers

```
├── data_process.py     # Programme python pour l'exploration et le nettoyage initiaux des données
├── cleaned_data.csv    # Data apres traitement utilise pour le Streamlit
├── data.csv            # Data brute
├── Streamlit.py          # Script principal de l'application Streamlit
└── requirements.txt      # Liste des paquets Python requis
```

## Contributeurs

- Yoav Cohen
- Salma Lahbati
- Cyrena Ramdani


