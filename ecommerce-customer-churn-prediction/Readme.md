# 🛒 Ecommerce Customer Churn Analysis and Prediction
**`Machine Learning | Classification | Real-World Business Problem`**

A production-oriented Machine Learning system built to predict **customer churn in e-commerce businesses** using modern preprocessing pipelines, hyperparameter optimization, and comparative model benchmarking.

Unlike the Titanic project, this project focuses on a **real-world industry problem** — helping businesses identify customers likely to leave so retention strategies can be applied before churn happens.

---

## ⚡ Project Overview

Customer churn prediction is one of the most important business problems in industries such as:

- E-commerce
- Banking
- Telecom
- Subscription Services
- SaaS Platforms

Retaining an existing customer is significantly cheaper than acquiring a new one.

This project builds an **end-to-end churn prediction system** using multiple Machine Learning models and selects the best production-ready model through comparative evaluation.

The project emphasizes:

✅ Modular Architecture  
✅ Data Preprocessing Pipelines  
✅ Hyperparameter Optimization  
✅ Cross-Validated Evaluation  
✅ Real-World Business Problem Solving

---

## 🎯 Objective

The objective of this project is to predict **which customers are likely to churn** using behavioral and transactional customer data.

The system helps businesses:

- Detect high-risk customers early
- Improve customer retention
- Reduce revenue loss
- Make data-driven decisions

---

## 🧠 Machine Learning Architecture

This project follows a **modular ML engineering workflow** by separating preprocessing, pipeline construction, model optimization, and execution into reusable components.

### 🔹 `preprocessing.py`
Handles:

- Data cleaning
- Feature engineering
- Dataset preparation
- Target variable separation

### 🔹 `pipelines.py`
Contains reusable preprocessing pipelines using:

- `ColumnTransformer`
- `Pipeline`
- `SimpleImputer`
- `RobustScaler`
- `StandardScaler`
- `OneHotEncoder`

### 🔹 `models_optuna.py`
Handles:

- Hyperparameter tuning
- Model optimization using **Optuna**
- Training logic for:
  - Logistic Regression
  - Random Forest
  - XGBoost

### 🔹 `main.py`
Acts as the execution entry point for the entire project.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-green)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter_Optimization-blueviolet)
![Pandas](https://img.shields.io/badge/Pandas-Data_Intelligence-150458?logo=pandas)

---

## 📊 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|-----------|------------|---------|-----------|
| Logistic Regression | 89.04% | 76.44% | 51.23% | 61.34% |
| Random Forest | 96.39% | 97.86% | 80.35% | 88.25% |
| **XGBoost** | **98.28%** | **97.76%** | **91.93%** | **94.76%** |

---

## 🏆 Final Model Selection

### Selected Model: XGBoost

After comparative evaluation, **XGBoost was selected as the final production model**.

Why?

✅ Highest Accuracy (**98.28%**)  
✅ Highest Recall (**91.93%**)  
✅ Highest F1 Score (**94.76%**)  
✅ Excellent Precision (**97.76%**)  
✅ Strong handling of imbalanced classification

This makes XGBoost the most reliable model for predicting customer churn in this dataset.

---

## 📂 Project Structure

```bash
Ecommerce-Customer-Churn-Analysis-and-Prediction/
│── dataset/
│   └── E_Commerce_Dataset.xlsx
│
│── src/
│   ├── preprocessing.py
│   ├── pipelines.py
│   └── models_optuna.py
│
│── main.py
│── requirements.txt
│── README.md
```

---

## 🚀 Installation & Usage

### 1️⃣ Clone Repository

```bash
git clone https://github.com/NaramCharan/Ecommerce-Customer-Churn-Analysis-and-Prediction.git
```

### 2️⃣ Navigate Into Project

```bash
cd Ecommerce-Customer-Churn-Analysis-and-Prediction
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Project

Execute:

```bash
python main.py
```

The workflow includes:

- Data preprocessing
- Pipeline creation
- Missing value handling
- Hyperparameter optimization using Optuna
- Model training
- Model benchmarking
- Final evaluation

---

## 🔮 Future Improvements

Planned production-level features:

- [ ] Automated email alerts for customers likely to churn
- [ ] Internal team notifications for high-risk customers
- [ ] Real-time churn prediction dashboard
- [ ] FastAPI deployment for live predictions

---

## 📬 Contact

### 👨‍💻 Naram Charan

**LinkedIn:**  
https://www.linkedin.com/in/naramcharan/

**Email:**  
charannaram1710@gmail.com

---

## ⭐ Support

If you found this project useful, consider giving it a **star** on GitHub!