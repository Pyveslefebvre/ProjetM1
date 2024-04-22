from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor  # Import de RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor  # Import de DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras import Sequential     # Import du réseau de Neurone
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, Normalizer
from random import randint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


"""Ouverture Data"""

df = pd.read_csv("datasets/samsung_dataset.csv", delimiter=";", on_bad_lines="warn")
# Convertir la colonne de dates en un format temporel
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Extraire des caractéristiques significatives de la date
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Supprimer la colonne de dates originale si nécessaire
df = df.drop("Date", axis=1)

# Modifier les virgule en point 
df = df.replace(',', '.', regex=True)

# Enlever les ligne commprenant un zéro 
df[df=='0']=np.nan
df = df.dropna()

X = df.drop("High",axis=1)
y = df["High"]

# Standardisation/Normalisation des valeur
scale = StandardScaler()
norm = Normalizer()  # Marche mieux avec la Normalisation

X = norm.fit_transform(X)

"""Random Forest"""

# Division des données en ensemble d'apprentissage et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Initialisation du modèle de la forêt aléatoire avec 100 arbres et une graine aléatoire fixée pour la reproductibilité.
rf = RandomForestRegressor(n_estimators=100, max_depth = 3, random_state=1)  # Utilisation de RandomForestRegressor

# Entraînement du modèle sur l'ensemble d'apprentissage
rf.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = rf.predict(X_test)

# Calcul de l'erreur quadratique moyenne (MSE) du modèle en comparant les prédictions avec les valeurs réelles
mse_rf = mean_squared_error(y_test, y_pred)

# Affichage de l'erreur quadratique moyenne (MSE) du modèle
print('Erreur quadratique moyenne (MSE) du modèle Random Forest:', mse_rf,"\n")

# Calcul du coefficient de détermination R²
r2_rf = r2_score(y_test, y_pred)

# Affichage du coefficient de détermination R²
print('Coefficient de détermination (R²) du modèle Random Forest:', r2_rf,"\n")

# Sélection d'un seul arbre pour RandomForestRegressor et visualisation
estimator_rf = rf.estimators_[0]
plt.figure(figsize=(20,10))
plot_tree(estimator_rf, filled=True, feature_names=X_train.columns, max_depth=3, fontsize=10)
plt.title('Visualisation d\'un arbre de la forêt aléatoire')  # Ajouter un titre
plt.show()

"""Decision Tree"""

# Initialisation du modèle d'arbre de décision de régression
tree = DecisionTreeRegressor(random_state=1)  # Utilisation de DecisionTreeRegressor avec une profondeur maximale de 3

# Entraînement du modèle sur l'ensemble d'apprentissage
tree.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred_tree = tree.predict(X_test)

# Calcul de l'erreur quadratique moyenne (MSE) et du coefficient de détermination (R²) pour le modèle Gradient Boosting
mse_tree = mean_squared_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mse_tree)
r2_tree = r2_score(y_test, y_pred_tree)

# Visualisation de l'arbre de régression DecisionTreeRegressor
plt.figure(figsize=(20, 10))
plot_tree(tree, filled=True, feature_names=X_train.columns, max_depth=3, fontsize=10)
plt.title('Visualisation de l\'arbre de décision de régression')  # Ajouter un titre différent
plt.show()

# Initialisation du GridSearch pour fixé les meilleurs hyperparametre
param_grid = {
    "max_depth": [None, 3],
    "max_features": [randint(1, 5), randint(5, 10), randint(10, 15)],
    "criterion": ["squared_error", "absolute_error"],

}

tree_RS = RandomizedSearchCV(tree, param_grid, cv=20)

tree_RS.fit(X_train, y_train)
y_pred_tree_RS = tree_RS.predict(X_test)

mse_tree_RS = mean_squared_error(y_test, y_pred_tree_RS)
rmse_tree_RS = np.sqrt(mse_tree_RS)
r2_tree_RS = r2_score(y_test, y_pred_tree_RS)

print('L\'Erreur quadratique with DecisionTree : ', rmse_tree)  # Doit être faible
print('L\'Erreur quadratique with optimized hyperparametre : ', rmse_tree_RS)  # Doit être faible
print('Score r2 with DecisionTree : ', r2_tree)  # Doit tendre vers 0 (0<r2<1)
print('Score r2 with optimized hyperparametre : ', r2_tree_RS)  # Doit tendre vers 0 (0<r2<1)

# Affichage du graphique
plt.show()

"""Gradient Boosting"""

# Initialisation du modèle GradientBoostingRegressor
gbr = GradientBoostingRegressor()

# Entraînement du modèle sur l'ensemble d'apprentissage
gbr.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred_gbr = gbr.predict(X_test)

# Calcul de l'erreur quadratique moyenne (MSE) et du coefficient de détermination (R²) pour le modèle Gradient Boosting
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mse_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)

# Initialisation RandomSearchCV pour optimiser les hyperparametre

param = {
    'max_depth': [7, 8, 9],
    'learning_rate': [0.1, 0.3, 0.6],
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['squared_error', 'absolute_error'],
    'loss': ['squared_error', 'absolute_error'],
}

gbr_RS = RandomizedSearchCV(gbr, param, cv=20)

gbr_RS.fit(X_train, y_train)
y_pred_gbr_RS = gbr_RS.predict(X_test)

mse_gbr_RS = mean_squared_error(y_test, y_pred_gbr_RS)
rmse_gbr_RS = np.sqrt(mse_gbr_RS)
r2_gbr_RS = r2_score(y_test, y_pred_gbr_RS)

# Affichage des résultats
print('Erreur quadratique moyenne (RMSE) du modèle Gradient Boosting:', rmse_gbr, "\n")
print('Erreur quadratique moyenne (RMSE) du modèle Gradient Boosting avec hyperparametre optimisé:', rmse_gbr_RS, "\n")
print('Coefficient de détermination (R²) du modèle Gradient Boosting:', r2_gbr, "\n")
print('Coefficient de détermination (R²) du modèle Gradient Boosting avec hyperparameter optimisé:', r2_gbr_RS, "\n")


"""Réseau de neurone"""


# Normalisation des caractéristiques pour améliorer la performance du réseau de neurones
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création du modèle de réseau de neurones
model = Sequential()
model.add(Dense(32, input_dim=X_train_scaled.shape[1], activation='relu'))  # Couche d'entrée
model.add(Dense(16, activation='relu'))  # Couche cachée
model.add(Dense(1, activation='linear'))  # Couche de sortie

# Compilation du modèle
model.compile(loss='mean_squared_error', optimizer='adagrad')

# Entraînement du modèle
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=10, verbose=0)

# Évaluation du modèle
loss = model.evaluate(X_test_scaled, y_test)


# Tracer la courbe de performance
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Modèle Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
