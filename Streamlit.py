import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import shap
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.formula.api import ols
from sklearn.model_selection import cross_val_score
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from xgboost import XGBRegressor
from scipy.stats import norm



# Titre de l'application
st.title("Analyse de régression multiple : Taux d'insertion")
st.write("""Ce travail a été réalisé par :
- **Salma Lahbati**
- **Cyrena Ramdani**
- **Yoav Cohen**
""")


# Texte introductif
st.markdown("""

L'objectif principal de cette analyse est d'étudier les facteurs influençant le **taux d'insertion**, c'est-à-dire le pourcentage de diplômés qui trouvent un emploi stable et rémunéré dans les **18 mois suivant leur diplôme**. Pour répondre à cet objectif, nous structurerons notre travail en plusieurs étapes.

### **Plan de l'étude**

#### 1. Présentation des données et des variables

#### 2. Construction du modèle de régression multiple

#### 3. Validation et diagnostic du modèle
##### 3.1 Analyse de la multicolinéarité
##### 3.2 Test de Breusch-Pagan (homoscédasticité)
##### 3.3: Correction de l'hétéroscédasticité
##### 3.4 Test de Breusch-Pagan pour XGBoost"

#### 4. Interprétation des résultats et analyse 
##### 4.1: Fondements théoriques de SHAP
##### 4.2: Test de Durbin-Watson (Autocorrélation)

#### 5: Prévisions (ARIMA)
##### 5.1: Modèle ARIMA : Explication théorique
##### 5.2: Comparaison des résidus : Régression Linéaire vs ARIMA
##### 5.3: Comparaison des prévisions futures : ARIMA vs Régression Linéaire
##### 5.4: Interprétation  et conclusion des prévisions


#### 6. Discussion et perspectives

---""") 
st.write("### 1: Présentation des données et des variables")

st.markdown("""
Cette étude repose sur un échantillon de diplômés hommes ayant suivi un cursus de Master dans la filière Sciences Technologiques et Sociales (STS) et Droit Économie Gestion (DEG). Les données incluent des observations issues de différentes disciplines, notamment : 
            - les Sciences fondamentales, 
            - Ensemble sciences, technologies et santé, 
            - Sciences de la vie et de la terre, 
            - Sciences de l'ingénieur, 
            - Informatique. 
""")  

st.markdown("""
Nous avons également plusieurs composantes au sein de :
"Code de la discipline" :
disc12 : Ensemble sciences, technologies et santé
disc13 : Sciences de la vie et de la terre
disc14 : Sciences fondamentales
disc15 : Sciences de l'ingénieur
disc16 : Informatique
"Code du secteur disciplinaire" :
Disc14_01 : Chimie
""")

st.markdown("""
Dans cette analyse, nous faisons appel à un **modèle de régression multiple** pour étudier les facteurs influençant le taux d'insertion des diplômés. Ce taux est expliqué par un ensemble de **variables explicatives**, qui incluent :
- Les caractéristiques spécifiques du **domaine d'études** des diplômés.
- La **situation des diplômés** 18 mois après l'obtention de leur diplôme.
- Des **indicateurs économiques et sociaux** clés tels que le **taux de chômage national**, les **niveaux de salaires** et la **répartition des emplois** en fonction de la stabilité et du type de contrat.

##### Première équation du modèle 
""")


st.latex(r'''
Taux\ d'insertion_i = \beta_0 + \beta_1 \cdot X_1 + \beta_2 \cdot X_2 + \dots + \beta_n \cdot X_n + \epsilon_i
''')

### Explications des termes dans l'équation
st.write("Où :")
st.latex(r'''
X_1, X_2, \dots, X_n
''')
st.write(": représentent les variables explicatives, comme le taux de chômage, les salaires, etc.")

st.latex(r'''
\beta_0, \beta_1, \dots, \beta_n
''')
st.write(": sont les coefficients à estimer.")

st.latex(r'''
\epsilon_i
''')
st.write(": désigne l'erreur aléatoire ou le résidu pour l'observation \(i\).")


# Chemin absolu du fichier
file_path = "/Users/yoavcohen/Desktop/DU Data/Statistiques/Projet Statistiques/Projet Baptiste/cleaned_data.csv"

# Fonction pour charger les données
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Vérification si le fichier existe
if not os.path.exists(file_path):
    st.error(f"Fichier introuvable : {file_path}")
    st.stop()
else:
    st.success(f"Fichier trouvé : {file_path}")

# Chargement des données
data = load_data(file_path)

# Prétraitement des données
data = data.dropna(subset=["Taux d’insertion"])  # Supprimer les lignes où "Taux d'insertion" est NaN
data["Taux d’insertion"] = pd.to_numeric(data["Taux d’insertion"], errors="coerce")
data["situation"] = data["situation"].apply(lambda x: 1 if x == "18 mois après le diplôme" else 0)

if 'Année' in data.columns:
    try:
        # Convertir en datetime et extraire uniquement l'année
        data['Année'] = pd.to_datetime(data['Année'], errors='coerce').dt.year

    except Exception as e:
        st.error(f"Erreur lors de la conversion de la colonne 'Année' : {e}")
        st.stop()
else:
    st.error("La colonne 'Année' n'existe pas dans le jeu de données.")


# Transformation des colonnes numériques
numeric_columns = [
    "Taux de chômage national", "Part des emplois de niveau cadre", 
    "Part des emplois de niveau cadre ou profession intermédiaire", 
    "Part des emplois à temps plein", "Salaire brut annuel estimé", 
    "Part des diplômés boursiers dans la discipline", 
    "% emplois extérieurs à la région de l’université", 
    "Part des emplois stables", "Année"
]
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna()

# Variables disponibles pour X
all_features = [
    "situation", "Taux de chômage national",
    "Part des emplois de niveau cadre", "Part des emplois de niveau cadre ou profession intermédiaire",
    "Part des emplois à temps plein", "Salaire brut annuel estimé",
    "Part des diplômés boursiers dans la discipline",
    "% emplois extérieurs à la région de l’université", "Part des emplois stables",
    "Code de la discipline", "Code du secteur disciplinaire", "Genre", "Année"
]
# Sidebar pour transformer "Code de la discipline"
st.sidebar.header("Transformer les valeurs de 'Code de la discipline'")
unique_disciplines = sorted(data["Code de la discipline"].unique())
default_disciplines = ["disc14", "disc12", "disc13", "disc15", "disc16"]
valid_default_disciplines = [x for x in default_disciplines if x in unique_disciplines]

selected_disciplines = st.sidebar.multiselect(
    "Choisissez les codes à transformer en 1 :",
    options=unique_disciplines,
    default=valid_default_disciplines,
    key="disciplines_multiselect"
)
data["Code de la discipline"] = data["Code de la discipline"].apply(
    lambda x: 1 if x in selected_disciplines else 0
)


# Sidebar pour transformer "Code du domaine"
st.sidebar.header("Transformer les valeurs de 'Code du domaine'")
unique_domaine = sorted(data["Code du domaine"].unique())
default_domaine = ["STS", "DEG"]
valid_default_domaine = [x for x in default_domaine if x in unique_domaine]

selected_domaine = st.sidebar.multiselect(
    "Choisissez les codes à transformer en 1 :",
    options=unique_domaine,
    default=valid_default_domaine,
    key="domaine_multiselect"
)
data["Code du domaine"] = data["Code du domaine"].apply(
    lambda x: 1 if x in selected_domaine else 0
)

# Sidebar pour transformer "Code du secteur disciplinaire"
st.sidebar.header("Transformer les valeurs de 'Code du secteur disciplinaire'")
unique_sectors = sorted(data["Code du secteur disciplinaire"].unique())
selected_sectors = st.sidebar.multiselect(
    "Choisissez les codes à transformer en 1 :",
    options=unique_sectors,
    default=[],
    key="sectors_multiselect"
)
data["Code du secteur disciplinaire"] = data["Code du secteur disciplinaire"].apply(
    lambda x: 1 if x in selected_sectors else 0
)

# Sidebar pour transformer "Genre"
st.sidebar.header("Transformer les valeurs de 'Genre'")
unique_genre = sorted(data["Genre"].unique())
default_genre = ["hommes"]
valid_default_genre = [x for x in default_genre if x in unique_genre]

selected_genre = st.sidebar.multiselect(
    "Choisissez les codes à transformer en 1 :",
    options=unique_genre,
    default=valid_default_genre,
    key="genre_multiselect"
)
data["Genre"] = data["Genre"].apply(
    lambda x: 1 if x in selected_genre else 0
)

# Sidebar pour sélectionner les variables explicatives
st.sidebar.header("Filtrer les variables explicatives")
valid_default_features = [x for x in all_features if x in all_features]  # Validation
selected_features = st.sidebar.multiselect(
    "Choisissez les variables explicatives à inclure dans le modèle :",
    options=all_features,
    default=valid_default_features,
    key="features_multiselect"
)

# Définition des variables explicatives et de la cible
X = data[selected_features]  # Inclure uniquement les variables sélectionnées
y = data["Taux d’insertion"]

# Vérification des colonnes sélectionnées
st.write("Variables explicatives sélectionnées :")
st.write(selected_features)

# Calcul du modèle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_before = LinearRegression()
model_before.fit(X_train, y_train)
y_pred_before = model_before.predict(X_test)


# Résultats avec statsmodels
X_sm_before = sm.add_constant(X)
model_sm_before = sm.OLS(y, X_sm_before).fit()
r2_before_statsmodels = model_sm_before.rsquared
condition_number_before = np.linalg.cond(X_sm_before)


st.markdown("""### 2. **Construction du modèle de régression multiple** """)
st.write(model_sm_before.summary())

st.write(f"**R²  :** {r2_before_statsmodels:.3f}")
st.write(f"**Condition Number :** {condition_number_before:.2e}")

st.markdown("""
#### Variables clés et interprétations :

1. **Taux de chômage national** """)

st.latex(r'''(\beta = -0.6688, \, p < 0.01)''')

st.markdown("""
   - Une augmentation de **1 %** du taux de chômage national entraîne une diminution moyenne de **0.6688 points** du taux d'insertion des diplômés. Cela souligne l'importance du contexte économique sur l'insertion professionnelle.

2. **Part des emplois stables**""") 
st.latex(r'''(\beta = 0.1965, \, p < 0.01)''')
st.markdown("""
   - Une augmentation de **1 %** de la part des emplois stables est associée à une augmentation moyenne de **0.1965 points** du taux d'insertion. Cela montre que les diplômés bénéficient directement d’un marché du travail avec plus d’emplois sécurisés.

3. **Part des emplois de niveau cadre ou profession intermédiaire**""") 

st.latex(r'''(\beta =  0.1519, \, p < 0.01)''')
st.markdown("""
   - Une augmentation de **1 %** de la part des emplois de niveau cadre ou intermédiaire entraîne une hausse moyenne de **0.1519 points** du taux d'insertion. Les opportunités professionnelles mieux qualifiées jouent un rôle important dans l'employabilité des diplômés.

---

Ces résultats montrent que l’insertion professionnelle des diplômés dépend fortement du contexte économique et de la qualité des emplois disponibles. Une amélioration des conditions du marché du travail, notamment avec plus d'emplois stables et qualifiés, pourrait significativement augmenter leur taux d'insertion.
""")

# Graphique des prédictions
fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred_before, alpha=0.7, label="Prédictions")
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Idéal")
ax1.set_xlabel("Valeurs réelles")
ax1.set_ylabel("Valeurs prédites")
ax1.set_title("Régression ")
ax1.legend()
st.pyplot(fig1)

y_real = y_test  # Les valeurs réelles
y_pred = y_pred_before  # Les valeurs prédites

# Création de deux graphiques côte à côte
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Histogramme des prédictions
ax1.hist(y_pred, bins=30, density=True, alpha=0.6, color='blue', label="Prédictions")
mu_pred, std_pred = norm.fit(y_pred)  # Moyenne et écart type
x_pred = np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], 100)
p_pred = norm.pdf(x_pred, mu_pred, std_pred)
ax1.plot(x_pred, p_pred, 'k', linewidth=2, label=f"Normale (μ={mu_pred:.2f}, σ={std_pred:.2f})")
ax1.set_title("Distribution des Prédictions")
ax1.set_xlabel("Valeurs prédites")
ax1.set_ylabel("Densité")
ax1.legend()

# Histogramme des valeurs réelles
ax2.hist(y_real, bins=30, density=True, alpha=0.6, color='orange', label="Valeurs réelles")
mu_real, std_real = norm.fit(y_real)  # Moyenne et écart type
x_real = np.linspace(ax2.get_xlim()[0], ax2.get_xlim()[1], 100)
p_real = norm.pdf(x_real, mu_real, std_real)
ax2.plot(x_real, p_real, 'r', linewidth=2, linestyle='--', label=f"Normale (μ={mu_real:.2f}, σ={std_real:.2f})")
ax2.set_title("Distribution des Valeurs Réelles")
ax2.set_xlabel("Valeurs réelles")
ax2.legend()

# Ajustement de l'espacement entre les graphiques
plt.tight_layout()

# Afficher les graphiques dans Streamlit
st.pyplot(fig)




# Multicolinéarité (VIF)
st.write("### 3: Validation et diagnostic du modèle")

st.write("#### 3.1: Analyse de la multicolinéarité (VIF)")
st.markdown("""
##### **Définition du VIF :**

Pour chaque variable explicative \(X_j\) d’un modèle de régression, le VIF est calculé comme :
""")
st.latex(r'''
\text{VIF}(X_j) = \frac{1}{1 - R_j^2}
''')
st.markdown("""où :""") 
st.latex(r''' R_j^2 ''') 
st.markdown("""est le coefficient de détermination obtenu en régressant  \(X_j\) sur toutes les autres variables explicatives du modèle.
""")

X_scaled = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled.values, i + 1) for i in range(len(X.columns))]
st.table(vif_data)

# Regroupement des variables liées aux salaires
salary_cols = [
    "Salaire brut annuel estimé"
    ]
X["Salaire moyen regroupé"] = X[salary_cols].mean(axis=1)
X = X.drop(columns=salary_cols)


st.markdown("""
##### Analyse de la Multicolinéarité

Les résultats du **Variance Inflation Factor (VIF)** indiquent une forte multicolinéarité pour les variables **Année** et **Taux de chômage national** (VIF > 10), ce qui risque de biaiser le modèle. Certaines variables, comme **Part des emplois de niveau cadre**, montrent une corrélation modérée.

Pour améliorer la robustesse du modèle, nous allons supprimer la variable **Année** car elle explique la même chose que la variable **Taux de chômage national**.

###### Explication : Pourquoi le taux de chômage pourrait expliquer la même chose que l'année ?

Le taux de chômage peut être lié à l'année en raison de plusieurs facteurs :

- **Tendance temporelle commune** : Le taux de chômage suit souvent une évolution régulière au fil des années, reflétant des tendances économiques globales.

- **Événements économiques ou politiques spécifiques** : Certains événements marquants, comme une récession ou une reprise économique, peuvent fortement associer une année donnée à un niveau particulier de chômage (par exemple, la crise financière de 2008 ou la pandémie de 2020-2022).

- **Manque de variables explicatives dans le modèle** : Si d'autres facteurs influençant le taux de chômage (comme des politiques économiques ou des innovations majeures) ne sont pas inclus dans le modèle, l'année peut agir comme une variable de substitution en capturant ces effets.

- **Structure des données** : Si le taux de chômage évolue de manière systématique d'une année à l'autre dans le jeu de données, une forte corrélation avec l'année est inévitable.

""")

# Analyse des résidus
residuals = model_sm_before.resid
y_pred = model_sm_before.predict(X_sm_before)


# Test de Breusch-Pagan pour l'homoscédasticité
bp_test = het_breuschpagan(residuals, X_sm_before)

st.write("#### 3.2: Test de Breusch-Pagan (Homoscédasticité)")

st.markdown("""
La statistique de Breusch-Pagan ($BP$) repose sur le coefficient de détermination $R^2$ de la régression auxiliaire :
""")

# Équation LaTeX
st.latex(r"BP = n \cdot R^2")

# Explications des termes
st.markdown("""
où :
$n$ est le nombre d'observations,
$R^2$ est le coefficient de détermination de la régression auxiliaire.
""")

# Test de Breusch-Pagan pour l'homoscédasticité
residuals = model_sm_before.resid  # Résidus du modèle
X_sm_before = sm.add_constant(X)  # Ajouter une constante aux variables explicatives

bp_test = het_breuschpagan(residuals, X_sm_before)

# Affichage des résultats du test
st.markdown(f"""
- **Statistique de Breusch-Pagan** : {bp_test[0]:.2f}  
- **p-valeur** : {bp_test[1]:.3e}  
""")

# Interprétation
if bp_test[1] < 0.05:
    st.markdown("""
    - La p-valeur est inférieure à 0.05, ce qui indique une **hétéroscédasticité** (variance des résidus non constante).
    - Cela suggère que le modèle pourrait ne pas bien représenter la variabilité des erreurs, et qu'un ajustement ou un autre modèle pourrait être nécessaire.
    """)
else:
    st.markdown("""
    - La p-valeur est élevée, ce qui confirme l'**homoscédasticité** (variance constante des résidus).
    - Cela indique que les erreurs du modèle sont réparties de manière uniforme, et que les hypothèses de base du modèle sont respectées.
    """)

# Graphique des résidus par rapport aux valeurs prédites pour visualiser l'homoscédasticité
fig_residuals, ax_residuals = plt.subplots()
ax_residuals.scatter(y_pred, residuals, alpha=0.7, color="skyblue")
ax_residuals.axhline(0, linestyle="--", color="red", linewidth=1)
ax_residuals.set_xlabel("Valeurs prédites")
ax_residuals.set_ylabel("Résidus")
ax_residuals.set_title("Graphique des résidus vs Valeurs prédites")
st.pyplot(fig_residuals)

st.markdown("""
##### Interprétation visuelle :
- Si les résidus sont dispersés de manière aléatoire autour de la ligne 0, cela soutient l'hypothèse d'homoscédasticité.
- Si une forme ou un patron particulier apparaît (par exemple, des "éventails" ou des clusters), cela indiquerait de l'hétéroscédasticité et suggérerait que la variance des erreurs n'est pas constante.
""")

st.write("### 3.3: Correction de l'hétéroscédasticité")
st.markdown("""Ici, il faut réduire l'hétéroscédasticité car la Statistique de Breusch-Pagan est de 92.98. """)

st.markdown("""
XGBoost réduit l'hétéroscédasticité en ajustant les erreurs de manière adaptative et en capturant des relations complexes entre les variables explicatives. 
Le modèle minimise une fonction de perte qui inclut à la fois la perte de prédiction et une régularisation pour contrôler la complexité.

La fonction de perte est définie comme suit :

""")

st.latex(r'''
\mathcal{L} = \sum_{i=1}^{n} \text{loss}(y_i, \hat{y}_i) + \lambda \sum_{t} \| \mathbf{w}_t \|^2
''')

st.markdown("""
Où :

- $\\text{loss}(y_i, \\hat{y}_i)$ est la fonction de perte, représentant l'erreur de prédiction entre la valeur réelle $y_i$ et la valeur prédite $\\hat{y}_i$.
- $\\lambda$ est un paramètre de régularisation qui contrôle la complexité du modèle.
- $\\mathbf{w}_t$ est le vecteur de poids des arbres dans le modèle.

Ce processus aide à réduire la variance des résidus, rendant le modèle plus robuste et réduisant ainsi l'hétéroscédasticité.
""")



# Définir le modèle
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)

# Entraîner le modèle
xgb_model.fit(X_train, y_train)

# Prédictions
y_pred_xgb = xgb_model.predict(X_test)


# Résidus et Test de Breusch-Pagan
residuals_xgb = y_test - y_pred_xgb
bp_test_xgb = het_breuschpagan(residuals_xgb, sm.add_constant(X_test))

st.write("### 3.4 Test de Breusch-Pagan pour XGBoost")
st.markdown(f"""
- **Statistique de Breusch-Pagan** : {bp_test_xgb[0]:.2f}  
- **p-valeur** : {bp_test_xgb[1]:.3e}  
""")

# Visualisation des résidus
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Graphique des résidus vs valeurs prédites
axs[0].scatter(y_pred_xgb, residuals_xgb, color='blue', alpha=0.5)
axs[0].axhline(0, color='red', linestyle='--')
axs[0].set_xlabel("Valeurs Prédites")
axs[0].set_ylabel("Résidus")
axs[0].set_title("Résidus vs Valeurs Prédites")

# Histogramme des résidus
sns.histplot(residuals_xgb, kde=True, color='blue', ax=axs[1])
axs[1].set_xlabel("Résidus")
axs[1].set_ylabel("Fréquence")
axs[1].set_title("Distribution des Résidus")

# Affichage des graphiques
plt.tight_layout()
st.pyplot(fig)

st.markdown(""" ### 4. Interprétation des résultats et analyse 

#### 4.1: Fondements théoriques de SHAP

**SHAP** (Shapley Additive Explanations) est basé sur la **valeur de Shapley**, une méthode issue de la théorie des jeux coopératifs. Elle permet d'attribuer à chaque caractéristique de manière transparente l'impact qu'elle a sur la prédiction d'un modèle, en tenant compte des interactions entre ces caractéristiques.

##### La valeur de Shapley :
La valeur de Shapley pour une caractéristique \( i \) est calculée comme suit :
""")

st.latex(r'''
\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} \left[v(S \cup \{i\}) - v(S)\right]
''')

st.markdown("""
Où :
- \(v(S)\) est la valeur d'un sous-ensemble \(S\) de caractéristiques, indiquant la prédiction ou la performance du modèle en fonction de \( S \).
- \( N \) est l'ensemble total des caractéristiques utilisées dans le modèle.
- Cette formule mesure l'impact marginal de chaque caractéristique \(i\) en fonction de sa contribution à tous les sous-ensembles possibles de caractéristiques.

Dans le cadre de SHAP, pour une caractéristique \(x_i\), la contribution à la prédiction est déterminée par la différence entre la prédiction du modèle avec et sans cette caractéristique :

""")

st.latex(r'''
\text{Contribution}(x_i) = \hat{y}_{\text{avec } x_i} - \hat{y}_{\text{sans } x_i}
''')

st.markdown("""
##### Application de SHAP :
SHAP est principalement utilisé pour interpréter les modèles complexes tels que :
- Les **arbres de décision** et **forêts aléatoires**.
- Les **réseaux neuronaux** et autres modèles non linéaires.

Les méthodes d'approximation, comme **Tree SHAP**, sont utilisées pour rendre ces calculs plus efficaces, notamment dans le cas de modèles à grande échelle.
""")


# Expliquez le modèle 
explainer = shap.LinearExplainer(model_before, X_train)  # Modèle et données d'entraînement
shap_values = explainer.shap_values(X_test)  # Obtenez les valeurs SHAP pour les prédictions sur X_test

# Affichez le résumé des valeurs SHAP
st.write("##### Analyse des contributions des variables avec SHAP")

# Créer une figure explicite avec matplotlib
fig = plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)  # `show=False` pour ne pas afficher automatiquement

# Affichez le graphique SHAP avec Streamlit
st.pyplot(fig)

st.markdown("""
- **Interprétation du graphique SHAP** :
    - Chaque barre indique l'importance de la variable pour la prédiction.
    - Les variables avec des barres plus longues ont une plus grande influence sur la prédiction du modèle.
    - Nous pouvons voir que la variable "Part des emplois stables" est la plus importante. En effet, plus le marché du travail est stable, 
    plus les jeunes sortant du master LMD ont de chance de trouver un travail 18 mois apres l'obtention du diplôme.
""")

st.write("#### 4.2: Test de Durbin-Watson (Autocorrélation)")
st.markdown("""

Le test de Durbin-Watson est utilisé pour détecter l'autocorrélation dans les résidus d'un modèle de régression. La statistique Durbin-Watson est calculée selon l'équation suivante :

""")

st.latex(r'''
DW = \frac{\sum_{t=2}^{n} (e_t - e_{t-1})^2}{\sum_{t=1}^{n} e_t^2}
''')

st.markdown("""
Où :
- \(e_t/) est l'erreur (résidu) à la période \(t\),
- \(n\) est le nombre d'observations,
- \(e_(t-1)\) est l'erreur à la période précédente.
""")
            
# Test de Durbin-Watson pour l'autocorrélation
dw_stat = durbin_watson(residuals)

# Affichage des résultats
st.markdown(f"""
- **Statistique Durbin-Watson** : {dw_stat:.2f}
- Valeurs idéales :
  - **Proche de 2** : Pas d’autocorrélation (les résidus sont indépendants).
  - **< 1 ou > 3** : Présence d’autocorrélation significative (les résidus sont corrélés).
""")

# Interprétation
if dw_stat < 1 or dw_stat > 3:
    st.markdown("""
    - La statistique Durbin-Watson est en dehors de la plage idéale (1 à 3), ce qui suggère une **autocorrélation significative** dans les résidus.
    - Cela signifie que les erreurs du modèle sont liées dans le temps (ou selon une autre dimension) et qu'un modèle plus complexe ou une transformation des variables pourrait être nécessaire.
    """)
else:
    st.markdown("""
    - La statistique Durbin-Watson est proche de 2, ce qui suggère qu'il n'y a **pas d'autocorrélation** significative.
    - Cela indique que les erreurs du modèle sont indépendantes, et que l’hypothèse d'indépendance des erreurs est respectée.
    """)

# Graphique de l'autocorrélation des résidus

fig_acf = plt.figure(figsize=(10, 6))
plot_acf(residuals, lags=40, ax=fig_acf.add_subplot(111), color='blue')
plt.title('Fonction d\'autocorrélation des résidus')
st.pyplot(fig_acf)

st.markdown("""
##### Interprétation visuelle :
- Le graphique de la fonction d'autocorrélation (ACF) des résidus montre les corrélations entre les erreurs à différents décalages.
- Si les barres dépassent les limites de confiance (représentées par les lignes horizontales), cela indique la présence d'autocorrélation à ces décalages.
- Si aucune barre ne dépasse ces limites, cela soutient l'indépendance des résidus et la validité du modèle.
""")


st.write("""
### 5: Prévisions (ARIMA)
#### 5.1: Modèle ARIMA : Explication théorique

Le modèle **ARIMA (AutoRegressive Integrated Moving Average)** est un modèle statistique utilisé pour prédire les séries temporelles. Il se compose de trois éléments principaux :

- **AR (AutoRegressive)** : Utilisation des valeurs passées pour prédire les valeurs futures. C'est le terme de régression basé sur les observations passées.
- **I (Integrated)** : Processus de différenciation des données pour les rendre stationnaires (éliminer la tendance).
- **MA (Moving Average)** : Modélisation de la relation entre l'observation actuelle et l'erreur de prédiction des périodes passées.

##### L'équation théorique d'un modèle ARIMA(p, d, q)

$$
(1 - \phi_1 B - \phi_2 B^2 - ... - \phi_p B^p)(1 - B)^d Y_t = (1 + \theta_1 B + \theta_2 B^2 + ... + \theta_q B^q) \epsilon_t
$$

où :
- \(Y_t\) est la valeur de la série temporelle au temps \(t\),
- \(B\) est l'opérateur de retard (\(B^k Y_t = Y_{t-k}\)),
- \(\phi_1, \phi_2, ..., \phi_p\) sont les coefficients de l'auto-régression (AR),
- \(\theta_1, \theta_2, ..., \theta_q\) sont les coefficients de la moyenne mobile (MA),
- \(d\) est le nombre de différenciations pour rendre la série stationnaire,
- \(\epsilon_t\) est le terme d'erreur à l'instant \(t\).

##### Détails du modèle ARIMA(4, 2, 4)

Dans ce modèle, les paramètres sont :
- **p = 4** : Quatre lags (valeurs passées) sont utilisés pour l'auto-régression (AR).
- **d = 2** : La série a été différenciée deux fois pour la rendre stationnaire.
- **q = 4** : Quatre lags de l'erreur sont utilisés pour la moyenne mobile (MA).
""")


if 'Année' in data.columns and 'Taux d’insertion' in data.columns:
    try:
        # Nettoyer la colonne 'Année'
        data['Année'] = data['Année'].astype(str).str.replace(',', '').astype(int)
        data = data.sort_values(by='Année')  # Trier par année

        # Séries temporelles
        time_series = data.set_index('Année')['Taux d’insertion']

        # Division en données train/test pour ARIMA
        train_size = int(len(time_series) * 0.8)
        train_data = time_series.iloc[:train_size]
        test_data = time_series.iloc[train_size:]

        # **Modèle ARIMA**
        p, d, q = 4, 2, 4  # Paramètres ARIMA
        arima_model = ARIMA(train_data, order=(p, d, q))
        arima_result = arima_model.fit()

        # Prévisions ARIMA sur la période de test
        forecast_arima = arima_result.forecast(steps=len(test_data))
        residuals_arima = test_data - forecast_arima  # Résidus ARIMA

        # MSE ARIMA
        mse_arima = mean_squared_error(test_data, forecast_arima)
        st.write(f"**Erreur quadratique moyenne (MSE) ARIMA :** {mse_arima:.2f}")


        # **Modèle de Régression Linéaire**
        # Utilisation de toutes les variables explicatives
        X = data[all_features]
        y = data['Taux d’insertion']

        # Division train/test pour la régression linéaire
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        # Entraînement et prédictions
        model_before = LinearRegression()
        model_before.fit(X_train, y_train)
        y_pred_before = model_before.predict(X_test)
        residuals_regression = y_test - y_pred_before  # Résidus régression linéaire



        # **Comparaison des résidus**
        st.write("#### 5.2: Comparaison des résidus : Régression Linéaire vs ARIMA")

        # Visualisation des résidus
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(residuals_arima, kde=True, label="Résidus ARIMA", color="green", bins=20, ax=ax)
        sns.histplot(residuals_regression, kde=True, label="Résidus Régression Linéaire", color="red", bins=20, ax=ax)
        ax.set_title("Répartition des résidus : ARIMA vs Régression Linéaire")
        ax.set_xlabel("Valeur des résidus")
        ax.set_ylabel("Fréquence")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # **Comparaison des prévisions futures**
        st.write("#### 5.3: Comparaison des prévisions futures : ARIMA vs Régression Linéaire")

        # Prévisions futures pour 5 ans
        future_years = 40
        last_year = data['Année'].max()

        # Prévisions ARIMA
        future_forecast_arima = arima_result.forecast(steps=future_years)
        future_years_arima = pd.RangeIndex(start=last_year + 1, stop=last_year + future_years + 1)
        future_df_arima = pd.DataFrame({'Année': future_years_arima, 'Prévisions ARIMA': future_forecast_arima})

        # Prévisions Régression Linéaire
        future_features = pd.DataFrame({col: [data[col].mean()] * future_years for col in all_features})
        future_features['Année'] = range(last_year + 1, last_year + future_years + 1)
        future_forecast_regression = model_before.predict(future_features)
        future_df_regression = pd.DataFrame({'Année': future_features['Année'], 'Prévisions Régression Linéaire': future_forecast_regression})

        # **Graphique interactif avec Plotly**
        fig = go.Figure()

        # Trace des données historiques
        fig.add_trace(go.Scatter(x=data['Année'], y=time_series, mode='lines', name="Données historiques", line=dict(color="black")))

        # Trace des prévisions ARIMA
        fig.add_trace(go.Scatter(x=future_df_arima['Année'], y=future_df_arima['Prévisions ARIMA'], mode='lines', name="Prévisions ARIMA", line=dict(color="green", dash="dash")))

        # Trace des prévisions Régression Linéaire
        fig.add_trace(go.Scatter(x=future_df_regression['Année'], y=future_df_regression['Prévisions Régression Linéaire'], mode='lines', name="Prévisions Régression Linéaire", line=dict(color="red", dash="dash")))
        st.markdown("##### Prévisions futures : ARIMA vs Régression Linéaire")


        st.plotly_chart(fig)

        # **Affichage des prévisions sous forme de tableau**
        st.write("##### Tableau des prévisions futures")

        # Fusionner les prévisions ARIMA et Régression dans un seul DataFrame
        forecast_df = pd.merge(future_df_arima, future_df_regression, on='Année')
        st.table(forecast_df)

    except Exception as e:
        st.error(f"Erreur dans l'analyse des modèles : {e}")
else:
    st.error("Les colonnes 'Année' et 'Taux d’insertion' doivent être présentes")

st.markdown("""
#### 5.4: Interprétation et conclusion des prévisions

Les prévisions des modèles **ARIMA** et **Régression Linéaire** montrent une légère baisse du **Taux d'insertion** entre 2021 et 2030, oscillant entre 87 et 88 %. Les deux modèles suivent une tendance similaire, avec des différences minimes dans les valeurs prédites.

- **ARIMA** prédit des taux légèrement plus élevés que la régression linéaire.
- La baisse progressive du taux d'insertion suggère une stabilisation de la situation, mais les variations restent faibles.
- Le modèle ARIMA fournit des prévisions qui tiennent compte des tendances temporelles et des saisonnalités potentielles dans les données.
- La régression linéaire, bien qu'utile, ne capture pas les effets temporels complexes et pourrait donner des prévisions moins précises sur des périodes futures.
""")



st.markdown("""
### 6: Discussion et perspectives:

L’analyse menée sur le taux d’insertion des diplômés a permis de mettre en lumière plusieurs facteurs clés influençant leur employabilité. Parmi ces facteurs, le contexte économique, représenté par le taux de chômage national, et les caractéristiques du marché du travail, comme la stabilité et la qualité des emplois disponibles, jouent un rôle déterminant. Une augmentation de la part des emplois stables ou de niveau cadre est directement associée à une amélioration du taux d’insertion, tandis qu’un environnement économique difficile (taux de chômage élevé) réduit significativement les opportunités pour les diplômés.

L’approche méthodologique, combinant une régression multiple et des outils avancés comme XGBoost et SHAP, a également permis de confirmer la robustesse des résultats. Les tests de diagnostic ont révélé quelques limites, notamment en termes d’hétéroscédasticité, mais des corrections ont été appliquées pour affiner les modèles et améliorer leur interprétabilité.

Cette étude souligne l’importance d’investir dans des politiques favorisant la création d’emplois stables et qualifiés, en particulier dans les secteurs où les diplômés sont les plus représentés. De plus, elle invite à réfléchir aux mesures qui pourraient atténuer l’impact des conditions économiques globales sur l’insertion professionnelle, comme le développement de dispositifs d’accompagnement ou la promotion de la mobilité géographique et sectorielle des diplômés.

Enfin, les perspectives offertes par cette analyse appellent à un approfondissement futur, notamment par l’intégration de données longitudinales pour mieux comprendre les trajectoires professionnelles des diplômés sur le long terme. Cela pourrait permettre de construire des modèles encore plus précis et d’anticiper les évolutions du marché du travail, afin de mieux préparer les générations futures à y évoluer.
""")


