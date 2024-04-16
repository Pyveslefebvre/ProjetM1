from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier  # Import de RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.svm import SVC  # Import du modèle SVM
from tensorflow.keras import Sequential #Import du réseau de Neurone
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier  # Import de DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import randint

# Chargement des données à partir du fichier CSV
df = pd.read_csv('datasets/cancer_cells.csv')

# Extraction de la variable cible 'diagnosis' et conversion des libellés en valeurs numériques.
y = df['diagnosis']
y = y.replace('B', 1)  # 'B' (bénin) est remplacé par 1
y = y.replace('M', 0)  # 'M' (maligne) est remplacé par 0

# Sélection des caractéristiques pertinentes pour la prédiction
X = df[['radius_mean', 'concave_points_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean']]

# Division des données en ensemble d'apprentissage et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

"""Random Forest"""

# Initialisation du modèle de la forêt aléatoire avec 100 arbres et une graine aléatoire fixée pour la reproductibilité.
rf = RandomForestClassifier(n_estimators=100, max_depth = 3, random_state=1)  # Utilisation de RandomForestClassifier

# Entraînement du modèle sur l'ensemble d'apprentissage
rf.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = rf.predict(X_test)

# Calcul de la précision du modèle en comparant les prédictions avec les valeurs réelles.
acc_rf = accuracy_score(y_test, y_pred)

# Sélection d'un seul arbre
estimator = rf.estimators_[0]

# Configuration de la taille de la figure pour la visualisation
plt.figure(figsize=(20,10))

# Visualisation de l'arbre
plot_tree(estimator, filled=True, feature_names=X_train.columns, class_names=['Malignant', 'Benign'], max_depth=3, fontsize=10)
# On limite la profondeur pour une meilleure lisibilité

# Affichage du graphique
plt.title("Arbre de décision d'un classifieur de forêt aléatoire")
plt.xlabel("Caractéristiques")
plt.ylabel("Classes")
plt.show()

"""SVM"""
# 'concave point' et 'radius mean' sont les plus interessants
X_vis = df[['radius_mean', 'concave_points_mean']]  # seulement 2 feature à visualiser

# split en training et en test
X_train_vis, X_test_vis, y_train, y_test = train_test_split(X_vis, y, test_size=0.2, random_state=21)

# entrainement du model svm sur 2 featuure
svm_vis = SVC(kernel='linear', C=1.0, random_state=1)
svm_vis.fit(X_train_vis, y_train)

# Prédiction sur l'ensemble de test avec le modèle SVM
y_pred_svm = svm_vis.predict(X_test_vis)

x_min, x_max = X_train_vis.iloc[:, 0].min() - 1, X_train_vis.iloc[:, 0].max() + 1
y_min, y_max = X_train_vis.iloc[:, 1].min() - 1, X_train_vis.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])  
Z = Z.reshape(xx.shape)

# Calcul de la précision du modèle SVM en comparant les prédictions avec les valeurs réelles
acc_svm = accuracy_score(y_test, y_pred_svm)

plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.coolwarm)
plt.scatter(X_train_vis.iloc[:, 0], X_train_vis.iloc[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.title('SVM Decision Boundaries (2 Features)')
plt.show()

"""Réseau de neurone"""

# Normalisation des caractéristiques pour améliorer la performance du réseau de neurones
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Création du modèle de réseau de neurones
model = Sequential()
model.add(Dense(12, input_dim=X_train_scaled.shape[1], activation='relu'))  # Couche d'entrée
model.add(Dense(8, activation='relu'))  # Couche cachée
model.add(Dense(1, activation='sigmoid'))  # Couche de sortie
# Compilation du modèle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Entraînement du modèle
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=10, verbose=0)
# Évaluation du modèle
_, accuracy = model.evaluate(X_test_scaled, y_test)
# Tracer la courbe de performance
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Modèle Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

"""Arbre de décision"""

# Initialisation du modèle d'arbre de décision de régression
tree_clf = DecisionTreeClassifier()  # Utilisation de DecisionTreeClassifier avec une profondeur maximale de 3
# Entraînement du modèle sur l'ensemble d'apprentissage
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
# Calcul de la précision du modèle SVM en comparant les prédictions avec les valeurs réelles
acc_rt = accuracy_score(y_test, y_pred_tree)

# Configuration de la taille de la figure pour la visualisation de l'arbre de régression
plt.figure(figsize=(20,10))
# Visualisation de l'arbre de régression
plot_tree(tree_clf, filled=True, feature_names=X_train.columns, class_names=['Malignant', 'Benign'], max_depth=3, fontsize=10)

# Initialisation de l'optimisation des hyperparametre
param_grid = {
    "max_depth": [None],
    "max_features": [randint(1, 5), randint(5, 10), randint(10, 15)],
    "criterion": ["gini", "entropy"],
}

tree_clf_RS = RandomizedSearchCV(tree_clf, param_grid, cv=20)

tree_clf_RS.fit(X_train, y_train)
y_pred_tree_RS = tree_clf_RS.predict(X_test)

acc_tree_RS = accuracy_score(y_test, y_pred_tree_RS)

# Affichage du graphique
plt.show()

"""Affichage"""

# Affichage de la précision des modèles
print('Random Forest test accuracy:', format(acc_rf, '.4f'))

print('SVM test accuracy:', format(acc_svm, '.4f'))

print(f'Reseau de neurones Accuracy: {accuracy*100:.2f}%')

print('Arbre de décision test accuracy:', format(acc_rt, '.4f'))

print('Arbre de décision test accuracy with optimised hypersettings: ', format(acc_tree_RS, '.4f'))