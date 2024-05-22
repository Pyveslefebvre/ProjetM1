import flwr as fl
import tensorflow as tf

#Definition du moèle
model = tf.keras.applications.MobileNetV2(
    input_shape=(32, 32, 3),
    classes=10,
    weights=None
)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#Chargement du dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        print("Entraînement du modèle...")
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        print("Évaluation du modèle...")
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


# Start Flower Server
try:
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=CifarClient()
    )
except Exception as e:
    print(f"Erreur lors de la connexion au serveur Flower: {e}")

"""
Résumé du dataset CIFAR-10:

CIFAR-10 est un jeu de données couramment utilisé en vision par ordinateur.
Il se compose de 60 000 images couleur de 32x32 pixels réparties en 10 classes différentes.
Chaque classe représente une catégorie d'objets courants.
Les classes sont les suivantes :Avion, Automobile, Oiseau, Chat, Cerf, Chien, Grenouille, Cheval, Navire, Camion
Chaque classe contient 6 000 images. 
Le jeu de données est divisé en un ensemble d'entraînement de 50 000 images et un ensemble de test de 10 000 images.

On utilise ici un modèle de réseau de neurones convolutif adapté à la taille des images 32x32x3 
"""
