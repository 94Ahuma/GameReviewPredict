# Game Review Cross-Domain Project

## Project Overview
This project investigates whether models trained on one game genre (RPG vs casual) can generalize to another.  
Compare model performance in **in-domain** (same dataset) and **cross-domain** (different dataset) settings.

---

## Current Progress
- Data cleaning completed for BG3 and Animal Crossing datasets  
- Data standardized into a unified format 
- Initial models built using TF-IDF + Logistic Regression / Naive Bayes / Linear SVM  
- In-domain and cross-domain evaluations completed  

Current findings:  
Models perform well in-domain (F1 ~0.87–0.89), but performance drops in cross-domain settings, indicating domain-specific language differences.

---


## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
