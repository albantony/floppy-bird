# Floppy Bird Project

Ce dépôt contient une implémentation de Flappy Bird développée avec Pygame, couplée à un agent d'apprentissage par renforcement de type Deep Q-Network (DQN) utilisant PyTorch.

## Description

Le projet simule un environnement de jeu où un agent (l'oiseau) apprend à naviguer entre des tuyaux en optimisant ses décisions (sauter ou ne rien faire) pour maximiser son score et sa survie.

## Fonctionnalités

* **Moteur de Jeu** : Développé avec Pygame, incluant la gestion de la gravité, des collisions et du défilement des obstacles.
* **Agent IA** : Utilisation d'un modèle DQN (Deep Q-Learning) pour l'apprentissage autonome.
* **Système de Récompenses** : Gain de points pour chaque tuyau franchi et pénalité en cas de collision.
* **Visualisation** : Rendu graphique activé périodiquement (tous les 50 épisodes) pour observer la progression de l'IA.

## Utilisation
Le fichier flappy_user.py contient une interface jouable pour l'utilisateur
Les fichiers restants sont des essais d'implémentation d'un agent autonome qui n'ont pour le moment pas aboutis

## Prérequis

* Python 3.x
* Pygame
* PyTorch
* NumPy

```bash
pip install pygame torch numpy

