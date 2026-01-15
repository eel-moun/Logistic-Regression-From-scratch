## 🪄 Introduction

Construire un **Sorting Hat** version data science : prédire la maison de Poudlard (*Gryffindor / Hufflepuff / Ravenclaw / Slytherin*) à partir de notes de cours, via une **Logistic Regression implémentée from scratch**.

### ✅ Ce qui est demandé
- Explorer le dataset (train/test) et comprendre ce qu’il contient
- Calculer des statistiques descriptives **sans fonctions “toutes faites”**
- Visualiser les données pour extraire des insights et faire de la **feature selection**
- Entraîner une **Logistic Regression multiclasses** (via **One-vs-Rest / One-vs-Many**)
- Produire un fichier de prédictions `houses.csv`

### 🧱 Contraintes (l’essentiel)
Ici, l’objectif n’est pas seulement un score : c’est **la compréhension** et **l’implémentation**.
- ❌ Pas de `describe()`, pas de `mean()`, `std()`, `percentile()` “magiques”
- ✅ Stats, loss, gradient, descente de gradient : **tout est reconstruit**
- 🎯 Cible : **accuracy ≥ 98% minimum**

---

## 🧭 Approche : 3 étapes (le chemin naturel d’un Data Scientist)

Atteindre **98%+** n’est pas une question de “mettre un modèle”.  
C’est une question de méthode.

### 1) Voir la data (intuition + nettoyage)
Avant de modéliser, on rend la data **fiable** :
- valeurs manquantes, outliers, incohérences
- colonnes inutiles ou trop bruitées
- ordre de grandeur / dispersion

👉 Cette étape passe par une **description statistique from scratch** : `count`, `mean`, `std`, `min/max`, quantiles…  
Le but : comprendre *ce que disent les chiffres*, pas juste exécuter un script.

---

### 2) Visualiser → sélectionner (feature selection)
Une fois la data “lisible”, on la rend **visible** :
- histogrammes : distributions + séparation entre maisons
- scatter plots : corrélations, similarités, clusters
- pair plots : repérer rapidement les couples de features utiles

La visualisation sert un objectif concret : **choisir de meilleures features**.

> Ce n’est pas la quantité de data qui fait la différence,  
> c’est la **qualité du signal** que tu arrives à extraire.

---

### 3) Modéliser : Logistic Regression (from scratch)

#### ✅ Sigmoid : du score à la probabilité
On calcule un score linéaire :

\[
z = w^T x + b
\]

Puis on le transforme en probabilité avec la sigmoid :

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

---

#### ✅ Loss function : pénaliser l’erreur “au bon endroit”
On veut punir fort un modèle **confiant mais faux** → **log-loss / cross-entropy** :

\[
J(w)= -\frac{1}{m}\sum_{i=1}^{m}\Big(y_i\log(\hat y_i) + (1-y_i)\log(1-\hat y_i)\Big)
\]

*(en pratique on protège aussi contre `log(0)` pour la stabilité numérique)*

---

#### ✅ Gradient Descent : apprendre en corrigeant
On met à jour les poids pour minimiser la loss :

\[
w \leftarrow w - \alpha \nabla_w J(w)
\]

##### 🔥 Le rôle de la dérivée (la vraie boussole)
La dérivée (le gradient) indique :
- **la direction** dans laquelle la loss augmente
- **l’intensité** de cette augmentation

Donc, pour *descendre*, on va dans la direction opposée.

Sans dérivée → pas de direction → pas d’apprentissage maîtrisé.

---

## 🌈 Multiclass : One-vs-Rest (One-vs-Many)
La Logistic Regression est binaire à la base.  
Pour 4 maisons, on entraîne **4 classifieurs** :

- Gryffindor vs All  
- Hufflepuff vs All  
- Ravenclaw vs All  
- Slytherin vs All  

En prédiction :
- chaque modèle renvoie une probabilité
- on choisit la classe au **maximum**

Simple, robuste, très efficace quand la feature selection est bonne.

---

## ⚡ Optimisation : Batch vs Mini-batch vs SGD

### Batch Gradient Descent
- gradient calculé sur **tout** le dataset
- stable, mais souvent plus lent

### Stochastic Gradient Descent (SGD)
- update **à chaque exemple**
- rapide, mais plus bruité (oscille davantage)

### Mini-batch Gradient Descent (le meilleur compromis)
- update sur un petit lot (ex: 32, 64…)
- plus rapide que batch, plus stable que SGD
- standard en pratique

---

## ✍️ Note finale

Ce projet ne vise pas juste un score.  
Il entraîne le réflexe le plus important : **faire les choses dans le bon ordre**.

**Comprendre → Visualiser → Sélectionner → Modéliser.**  
Parce qu’en Data Science, le modèle n’est pas la magie.

La magie, c’est ta capacité à transformer de la donnée brute  
en un signal clair… puis en une décision fiable.

✨ *Exactly what a good Sorting Hat does.*
