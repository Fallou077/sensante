---
title: SenSante
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# SenSante
Assistant de pré-diagnostic médical pour le Sénégal.

## 🚀 Démo en ligne
Tu peux tester l'application directement à cette adresse :
[https://fallou077-sensante.hf.space](https://fallou077-sensante.hf.space)

## 🛠️ Stack Technique
* **scikit-learn** : Entraînement et sérialisation du modèle de Machine Learning (Random Forest).
* **FastAPI** : Backend performant pour exposer les routes de pré-diagnostic (`/predict`) et de santé (`/health`).
* **Tailwind CSS** : Interface utilisateur responsive, moderne et épurée pour les patients.
* **Groq / Llama 3** : Intégration d'un LLM pour générer des explications médicales personnalisées (avec des expressions en wolof).
* **Docker** : Conteneurisation complète de l'application pour un déploiement fluide sur Hugging Face Spaces.

## 📝 Structure du Projet
* `data/` : Données patients (fichiers CSV).
* `models/` : Modèle de Machine Learning sérialisé (`.pkl`).
* `api/` : Code source de l'API FastAPI (`main.py`).
* `frontend/` : Interface web (fichiers HTML/CSS/JS).
* `notebooks/` : Scripts d'exploration et d'entraînement du modèle.

## 👤 Auteur
* **Moussa Diallo** - L2 GLSI - ESP / UCAD - 2026

## 🎓 Cours
* **Intégration de Modèles IA** - Dr. El Hadji Bassirou TOURÉ