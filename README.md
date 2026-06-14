# Nepal Trekking Analytics & Recommendation System

This repository contains the source code and materials for the MINspire Capstone 2026 project: a Data-Driven Trek Recommendation and Analytics System for Nepal Trekking.

Existing trekking recommendations are largely informal, making it difficult to account for preferences such as difficulty, duration, altitude, cost, and season simultaneously. This project addresses these challenges using the Nepal Trekking Dataset to make the trekking landscape more accessible and navigable for both novice and experienced trekkers.

## Core Features

* **Trek Personas & Recommendation:** Segments treks into 5 distinct "personas" (such as Extreme Altitude Expeditions and Short Budget Escapes) using K-Means clustering. It suggests the most suitable treks based on user preferences using Cosine Similarity and a KNN-based classifier.
* **Cost Prediction:** Models trek characteristics to estimate costs using Decision Tree regressors and Linear Regression with regularization.
* **Hidden Gem Detection:** Identifies underrated, high-value treks (like the Tsum Valley and Manaslu Trek) that are statistically anomalous using Isolation Forest.

## Model Performance Highlights

* **Persona Classification (KNN):** Achieved an accuracy of 93.75% in predicting trek personas.
* **Cost Prediction (Decision Tree):** Captured nonlinear patterns successfully, achieving an R² score of 0.755.

## Team Pringle:
* Sudip Tamang 
* Aaditya Sapkota
* Anuj Sapkota 
* Monika Rana 
* Acchyut Bhandari 
