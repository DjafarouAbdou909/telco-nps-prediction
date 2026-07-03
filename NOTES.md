# Projet de prédiction NPS – Notes

## 1. Vue d’ensemble du projet

Ce projet vise à prédire la catégorie NPS (Détracteur, Passif, Promoteur) des clients d’un opérateur télécom à partir de données structurées liées à leur compte et à leur comportement.

L’objectif est de soutenir les stratégies de rétention client en identifiant les clients à risque et en comprenant les facteurs d’insatisfaction.

---

## 2. Problème métier

Seulement 15 % des clients répondent aux enquêtes NPS, ce qui crée une zone d’ombre importante pour l’entreprise.

Le défi consiste à étendre cette visibilité à l’ensemble de la base client grâce au machine learning.

---

## 3. Problème de machine learning

Il s’agit d’un problème de classification supervisée à trois classes ordonnées :

- Détracteur
- Passif
- Promoteur

La variable cible est construite à partir du Score de Satisfaction (1–5).

---

## 4. Hypothèses clés

- Le Score de Satisfaction est un proxy acceptable du NPS.
- Les répondants aux enquêtes sont partiellement représentatifs de l’ensemble des clients.
- Les variables structurées contiennent suffisamment d’information pour prédire la satisfaction client.

---

## 5. Risques et limites

- Risque de fuite de données avec Churn Score et Churn Value.
- Déséquilibre des classes (en particulier les Détracteurs).
- Incertitude liée à la construction de la variable cible.
- Décalage de distribution entre répondants et non-répondants.
- Corrélation ≠ causalité.

---

## 6. Critères de succès

- Bon rappel sur la classe Détracteur.
- Utilisation de métriques équilibrées (F1 macro, kappa pondéré, etc.).
- Modèle interprétable.
- Résultats exploitables par des équipes non techniques.

---

## 7. Prochaines étapes

- Nettoyage et préparation des données
- Ingénierie des variables
- Modèle baseline
- Modélisation avancée
- Évaluation et analyse d’équité
- Prototype de déploiement