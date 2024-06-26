from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import randint

"""Data Loading"""

df = pd.read_csv("datasets/samsung_dataset.csv", delimiter=";", on_bad_lines="warn")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Extracting significant features from the date
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Dropping the original date column if necessary
df = df.drop("Date", axis=1)

# Replace commas with periods
df = df.replace(',', '.', regex=True)

# Removing rows containing zeros
df[df=='0'] = np.nan
df = df.dropna()

X = df.drop("High", axis=1)
y = df["High"]

# Normalizing the features
norm = Normalizer()
X = norm.fit_transform(X)

# Storing column names for later use
column_names = df.drop("High", axis=1).columns

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

"""Random Forest"""

# Initializing the Random Forest model with 100 trees and fixed random state for reproducibility
rf = RandomForestRegressor(n_estimators=100, max_depth=30,max_features='sqrt',bootstrap=True,min_samples_leaf=1,
                           min_samples_split=2, random_state=1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred)

# Plot prediction error
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='Perfect Prediction')
plt.title('Prediction Error')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()
plt.show()

"""
# Définir la grille de recherche des hyperparamètres
param_grid = {
    'n_estimators': [50,75,100,125,150,175,200,225,250,275,300],  # Nombre d'arbres dans la forêt
    'max_features': ['auto', 'sqrt'],  # Nombre de caractéristiques à considérer à chaque division
    'max_depth': [None, 10,15,20,25,30],  # Profondeur maximale de l'arbre
    'min_samples_split': [2, 5, 10],  # Nombre minimum d'échantillons requis pour diviser un nœud
    'min_samples_leaf': [1, 2, 4],  # Nombre minimum d'échantillons requis à chaque nœud feuille
    'bootstrap': [True, False]  # Utiliser Bootstrap pour construire les arbres
}

# Initialiser le modèle Random Forest
rf = RandomForestRegressor()

# Initialiser RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Exécuter RandomizedSearchCV pour trouver les meilleurs paramètres
rf_random.fit(X_train, y_train)

# Afficher les meilleurs paramètres
print(rf_random.best_params_)

# Utiliser les meilleurs paramètres pour entraîner un nouveau modèle
best_rf_model = rf_random.best_estimator_
best_rf_model.fit(X_train, y_train)

# Prédire avec le modèle optimisé
y_pred = best_rf_model.predict(X_test)

# Calculer les métriques de performance
mse_rf = mean_squared_error(y_test, y_pred)
r2_rf = r2_score(y_test, y_pred)

# Afficher les MSE et R2
print("MSE:", mse_rf)
print("R-squared:", r2_rf)

# Afficher les prédictions
print("Prédictions:", y_pred)

# Afficher les meilleurs paramètres
print("Meilleurs paramètres:", rf_random.best_params_)
"""

"""Decision Tree"""

# Initialisation du modèle d'arbre de décision de régression
tree = DecisionTreeRegressor(criterion='squared_error', max_depth= 10, max_features= 8,random_state=1)  # Utilisation de DecisionTreeRegressor avec une profondeur maximale de 3

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
plot_tree(tree, filled=True, max_depth=3, fontsize=10)
plt.title('Visualisation de l\'arbre de décision de régression')  # Ajouter un titre différent
plt.show()

"""
# Initialisation du GridSearch pour fixé les meilleurs hyperparametre
param_grid = {
    "max_depth": [None, 3, 6, 10],
    "max_features": [2, 4, 8, 10, 20, 50, 100],
    "criterion": ["squared_error", "absolute_error"],

}

tree_RS = GridSearchCV(tree, param_grid, cv=20)

tree_RS.fit(X_train, y_train)
y_pred_tree_RS = tree_RS.predict(X_test)

hyperparam = tree_RS.best_params_

"""


"""Gradient Boosting"""

# Initialisation du modèle GradientBoostingRegressor
gbr = GradientBoostingRegressor(criterion= 'friedman_mse', learning_rate= 0.1, loss= 'squared_error', max_depth= 9, max_features= 'log2')

# Entraînement du modèle sur l'ensemble d'apprentissage
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)

mse_gbr = mean_squared_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mse_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)

"""
# Initialisation GridSearch pour optimiser les hyperparametre
param = {
    'max_depth': [7, 8, 9],
    'learning_rate': [0.1, 0.3, 0.6],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['friedman_mse', 'squared_error'],
    'loss': ['squared_error', 'absolute_error'],
}

gbr_RS = GridSearchCV(gbr, param, cv=10, verbose= 1)

gbr_RS.fit(X_train, y_train)
"""

"""Neural Network"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(32, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adagrad')

history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=10, verbose=0)
loss = model.evaluate(X_test_scaled, y_test)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

"""Affichage"""
print("Random forest RMSE:", rmse_rf)
print("Random forest R²:", r2_rf)
print('Decision Tree RMSE: ', rmse_tree)
print('Decision Tree R²: ', r2_tree)
print('Gradient Boosting RMSE:', rmse_gbr)
print('Gradient Boosting R²:', r2_gbr)

# Liste des noms de modèles
models = ['Random Forest', 'Decision Tree', 'Gradient Boosting']

# Liste des valeurs RMSE pour chaque modèle
rmse_values = [rmse_rf, rmse_tree, rmse_gbr]

# Liste des valeurs R² pour chaque modèle
r2_values = [r2_rf, r2_tree, r2_gbr]

# Création du graphique
plt.figure(figsize=(10, 6))

# Plot RMSE values
plt.subplot(1, 2, 1)
bars_rmse = plt.bar(models, rmse_values, color='skyblue')
plt.title('RMSE Comparison')
plt.xlabel('Models')
plt.ylabel('RMSE')

# Ajout des annotations pour afficher les valeurs de RMSE sur chaque barre
for bar in bars_rmse:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, '{:.4f}'.format(height), ha='center', va='bottom')

# Plot R² values
plt.subplot(1, 2, 2)
bars_r2 = plt.bar(models, r2_values, color='lightgreen')
plt.title('R² Comparison')
plt.xlabel('Models')
plt.ylabel('R²')

# Ajout des annotations pour afficher les valeurs de R² sur chaque barre
for bar in bars_r2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, '{:.4f}'.format(height), ha='center', va='bottom')

# Affichage du graphique
plt.tight_layout()
plt.show()
