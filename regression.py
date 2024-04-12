from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Import de RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor  # Import de DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras import Sequential #Import du réseau de Neurone
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
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
tree = DecisionTreeRegressor(max_depth=3, random_state=1)  # Utilisation de DecisionTreeRegressor avec une profondeur maximale de 3

# Entraînement du modèle sur l'ensemble d'apprentissage
tree.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred_tree = tree.predict(X_test)

# Calcul de l'erreur quadratique moyenne (MSE) et du coefficient de détermination (R²) pour le modèle Gradient Boosting
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)


# Visualisation de l'arbre de régression DecisionTreeRegressor
plt.figure(figsize=(20,10))
plot_tree(tree, filled=True, feature_names=X_train.columns, max_depth=3, fontsize=10)
plt.title('Visualisation de l\'arbre de décision de régression')  # Ajouter un titre différent
plt.show()

# Affichage des résultats
print('Erreur quadratique moyenne (MSE) du modèle Decision Tree:', mse_tree,"\n")
print('Coefficient de détermination (R²) du modèle Decision Tree:', r2_tree,"\n")

"""Gradient Boosting"""

# Initialisation du modèle GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1)

# Entraînement du modèle sur l'ensemble d'apprentissage
gbr.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred_gbr = gbr.predict(X_test)

# Calcul de l'erreur quadratique moyenne (MSE) et du coefficient de détermination (R²) pour le modèle Gradient Boosting
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)

# Affichage des résultats
print('Erreur quadratique moyenne (MSE) du modèle Gradient Boosting:', mse_gbr,"\n")
print('Coefficient de détermination (R²) du modèle Gradient Boosting:', r2_gbr,"\n")

# Tracé de l'importance des caractéristiques
plt.figure(figsize=(10, 8))
feature_importance = gbr.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(X.columns)[sorted_idx])
plt.title('Importance des caractéristiques pour le modèle Gradient Boosting')
plt.show()

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
