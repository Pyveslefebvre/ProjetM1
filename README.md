# <p align="center">Bienvenue dans notre projet ING2 2 Majeure Intelligence Artificielle</p>

## <p align="center">Le Federated Learning, première approche d'une révolution de l'apprentissage supervisé</p>


<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*nMG4NRWj8hp3qTgTtOZwyA.png" alt="Federated Learning">
</p>

## Description
Ce projet vise à réaliser notre première implémentation de Federated Learning via PyTorch, TensorFlow et un serveur Flower.

## Caractéristiques
- Créer plusieurs réseaux de neurones 
- Entraînement sur le dataset CIFAR-10
- Communication et agrégation des modèles via un serveur Flower

### Étapes
1. Cloner le répertoire:
```bash
git clone https://github.com/Pyveslefebvre/ProjetM1.git
```
2. Naviguer jusqu'au repertoire:
```bash
cd ProjetM1
```
3. Installer les dépendances:
```bash
pip install -r requirements.txt
```
4. Lancez les applications souhaitées:

a. Tests sur la régression:
```bash
python regression.py
```
b. Tests sur la classification :
```bash
python classification.py
```
c. Federated Learning avec PyTorch (première exécution):
```bash
python centralized.py
```
  Ensuite, dans différents terminaux :
  
  Terminal A:
```bash
cd pytorch
python pytorchserver.py
```
  Terminal B:
```bash
cd pytorch
bash start_clients.sh
```
d. Federated Learning avec TensorFlow
  Terminal A:
```bash
cd flower
python flowerserver.py
```
  Terminal B:
```bash
cd flower
bash start_clients.sh
```

## Auteurs

👤 **Pierre-Yves Lefebvre, Virgile Louvel & Louison Beranger**
