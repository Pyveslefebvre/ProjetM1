#!/bin/bash

# Nombre de clients à lancer
num_clients=5

# Adresse du serveur Flower
server_address="127.0.0.1:8080"
for ((i=1; i<$num_clients+1; i++))
do
   echo "Lancement du client $i"
   python pytorchclient.py $server_address &
done

wait
