from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.svm import SVC
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import randint
import  time

# Compteur
start_time = time.time()

# Chargement des données à partir du fichier CSV
df = pd.read_csv('datasets/cancer_cells.csv')

# Extraction de la variable cible 'diagnosis' et conversion des libellés en valeurs numériques.
y = df['diagnosis']
y = y.replace('B', 1)  # 'B' (bénin) est remplacé par 1
y = y.replace('M', 0)  # 'M' (maligne) est remplacé par 0

# Sélection des caractéristiques pertinentes pour la prédiction
X = df[['radius_mean', 'concave_points_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

"""Random Forest"""
rf = RandomForestClassifier(n_estimators=400, min_samples_split=2, min_samples_leaf=1, max_depth=16, bootstrap=False)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred)
"""
#Obtention des meilleures paramètres
rf_param_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
    'max_depth': [int(x) for x in np.linspace(1, 20, num=20)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_grid, n_iter=100, cv=5, verbose=2,
                               random_state=42, n_jobs=-1)
rf_random.fit(X_train, y_train)
best_params = rf_random.best_params_
rf_best = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                 max_depth=best_params['max_depth'],
                                 min_samples_split=best_params['min_samples_split'],
                                 min_samples_leaf=best_params['min_samples_leaf'],
                                 bootstrap=best_params['bootstrap'],
                                 random_state=42)
rf_best.fit(X_train, y_train)
y_pred_best = rf_best.predict(X_test)
acc_rf_best = accuracy_score(y_test, y_pred_best)
best_params = rf.best_params_
print(best_params)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf, X, y, cv=5)
print("Scores de validation croisée :", scores)
print("Moyenne des scores de validation croisée :", scores.mean())
"""
best_estimator_tree = rf.estimators_[0]
plt.figure(figsize=(20,10))
plot_tree(best_estimator_tree, filled=True, feature_names=X_train.columns, class_names=['Malignant', 'Benign'], max_depth=3, fontsize=10)
plt.title("Arbre de décision du meilleur classifieur de forêt aléatoire")
plt.xlabel("Caractéristiques")
plt.ylabel("Classes")
plt.show()

"""SVM"""
# 'concave point' et 'radius mean' sont les plus interessants
X_vis = df[['radius_mean', 'concave_points_mean']]  # seulement 2 feature à visualiser
# split en training et en test
X_train_vis, X_test_vis, y_train, y_test = train_test_split(X_vis, y, test_size=0.2, random_state=21)

# Entrainement du model svm sur 2 features
svm_vis = SVC(kernel='poly', C=10, random_state=1, gamma='auto',degree=4)
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
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=10, label='Malignant'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Benign')])
plt.xlabel('Radius Mean')
plt.ylabel('Concave Points Mean')

plt.show()
"""
# Définir la grille des hyperparamètres à rechercher
param_grid = {
    'C': [0.1, 1, 10    ],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4, 5]
}

# Initialiser RandomizedSearchCV
svm_random = RandomizedSearchCV(estimator=svm_vis, param_distributions=param_grid, n_iter=100, cv=5, verbose=2,
                                random_state=42, n_jobs=-1)

# Effectuer la recherche aléatoire sur la grille
svm_random.fit(X_train_vis, y_train)

# Obtenir les meilleurs paramètres
best_params_svm = svm_random.best_params_
print("Meilleurs paramètres pour SVM après optimisation :")
print(best_params_svm)

# Utiliser les meilleurs paramètres pour entraîner le modèle SVM
svm_best = SVC(**best_params_svm)
svm_best.fit(X_train_vis, y_train)

# Prédiction sur l'ensemble de test avec le modèle SVM optimisé
y_pred_svm_best = svm_best.predict(X_test_vis)

# Calcul de la précision du modèle SVM optimisé
acc_svm_best = accuracy_score(y_test, y_pred_svm_best)
print("Précision du SVM après optimisation :", acc_svm_best)
"""

"""Réseau de neurone"""

# Normalisation des caractéristiques pour améliorer la performance du réseau de neurones
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""
# Fonction pour optimiser les hyperparamètres du modèle
def optimize_hyperparameters(X_train_scaled, y_train):
    def build_hypermodel(hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units_input', min_value=8, max_value=64, step=8),
                        input_dim=X_train_scaled.shape[1], activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout_input', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(Dense(units=hp.Int('units_hidden', min_value=8, max_value=64, step=8), activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout_hidden', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    tuner = RandomSearch(
        build_hypermodel,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=2,
        directory='hyperparam_opt',
        project_name='breast_cancer_optimization'
    )

    tuner.search(X_train_scaled, y_train, epochs=100, validation_split=0.2, verbose=1)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Meilleurs hyperparamètres trouvés :")
    print(f"Nombre d'unités dans la couche d'entrée : {best_hps.get('units_input')}")
    print(f"Taux de dropout dans la couche d'entrée : {best_hps.get('dropout_input')}")
    print(f"Nombre d'unités dans la couche cachée : {best_hps.get('units_hidden')}")
    print(f"Taux de dropout dans la couche cachée : {best_hps.get('dropout_hidden')}")

# Utilisation de la fonction d'optimisation pour trouver les meilleurs hyperparamètres
optimize_hyperparameters(X_train_scaled, y_train)
"""

# Création du modèle de réseau de neurones avec les hyperparamètres optimisés
model = Sequential()
model.add(Dense(56, input_dim=X_train_scaled.shape[1], activation='relu'))  # Nombre d'unités optimisé
model.add(Dropout(0.1))  # Taux de dropout optimisé
model.add(Dense(64, activation='relu'))  # Nombre d'unités pour la couche cachée optimisé
model.add(Dropout(0.30000000000000004))  # Taux de dropout pour la couche cachée optimisé
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=10, verbose=0)

# Évaluation du modèle sur l'ensemble de test
_, accuracy = model.evaluate(X_test_scaled, y_test)

# Affichage de la courbe de performance
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Précision du Modèle')
plt.ylabel('Précision')
plt.xlabel('Époque')
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

# Listes des noms de modèles
models = ['Random Forest', 'SVM', 'Réseau de Neurone', 'Decision Tree']

# Listes des précisions des modèles
accuracies = [acc_rf, acc_svm, accuracy, acc_rt]

# Création du graphique avec annotations et fonctionnalité de zoom pour afficher entre 0.8 et 1
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red'])

# Ajout des annotations pour afficher les valeurs de précision sur chaque barre
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, '{:.4f}'.format(height), ha='center', va='bottom')

plt.title('Comparaison de la précision des modèles')
plt.xlabel('Modèles')
plt.ylabel('Précision')
plt.ylim(0.90, 1)
plt.gca().set_ylim([0.90, 1])
plt.tight_layout()
plt.show()

end_time = time.time()
temps_total = end_time - start_time
print(f"Temps d'exécution : {temps_total} seconds")
