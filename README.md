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

## 🔍 Core concepts (Sigmoid, Loss, Gradient Descent)

### 1) Why Logistic Regression uses the **sigmoid**
Logistic Regression starts with a **linear score**:

~~~math
z = w^\top x + b
~~~

But a raw linear score can be any real number \((-\infty, +\infty)\).  
We need a function that:
- maps any value to a **probability**
- stays in **[0, 1]**
- increases smoothly with the score

That’s exactly what the sigmoid does:

~~~math
\sigma(z)=\frac{1}{1+e^{-z}}
~~~

So the model outputs a probability:

~~~math
\hat{y}=P(y=1\mid x)=\sigma(w^\top x + b)
~~~

**Intuition**
- big positive \(z\) → \(\hat{y}\approx 1\)
- big negative \(z\) → \(\hat{y}\approx 0\)
- \(z=0\) → \(\hat{y}=0.5\)

---

### 2) Loss function — what it is and why we need it
A model needs a **score** that measures how wrong it is, so we can improve it.  
For binary classification we use **log-loss** (a.k.a. cross-entropy), because it:
- strongly penalizes **confident but wrong** predictions
- matches the probabilistic output of sigmoid
- is differentiable → perfect for gradient-based optimization

~~~math
J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}\left[
y^{(i)}\log\!\left(\hat{y}^{(i)}\right) + (1-y^{(i)})\log\!\left(1-\hat{y}^{(i)}\right)
\right]
~~~

**Key intuition**
- If \(y=1\) and \(\hat{y}\) is small → huge penalty  
- If \(y=0\) and \(\hat{y}\) is large → huge penalty  

To avoid numerical issues (like `log(0)`), we clip probabilities:

~~~math
\hat{y} \leftarrow \text{clip}(\hat{y}, \varepsilon, 1-\varepsilon)
\qquad (\varepsilon \approx 10^{-15})
~~~

---

### 3) Gradient Descent — what it is, how we use it, and why
Once we have a loss \(J(w,b)\), we want parameters \((w,b)\) that **minimize** it.

**Gradient Descent** is an iterative algorithm:
1) compute derivatives (the gradient) of the loss  
2) update parameters in the **opposite direction** of the gradient

Why derivatives matter:
- the gradient tells us the direction where the loss increases fastest
- moving opposite reduces the loss (locally) as efficiently as possible

For Logistic Regression, the gradients are:

~~~math
\frac{\partial J}{\partial w} = \frac{1}{m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)x^{(i)}
~~~

~~~math
\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)
~~~

Update rule:

~~~math
w \leftarrow w - \alpha \frac{\partial J}{\partial w}
\qquad
b \leftarrow b - \alpha \frac{\partial J}{\partial b}
~~~

Where \(\alpha\) is the **learning rate**:
- too big → unstable / diverges
- too small → very slow learning

**In short:** sigmoid → probabilities, log-loss → correct error signal, gradient descent → systematic learning via derivatives.
----------

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
