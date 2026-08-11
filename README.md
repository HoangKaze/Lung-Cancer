# 🫁 Lung Cancer & Pulmonary Disease Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/) [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) [![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

Pulmonary diseases and lung cancer remain primary contributors to global medical mortality. Early detection through clinical screening and risk assessments plays a critical role in proactive treatment and improving patient survival rates.

This repository focuses on building an end-to-end **Data Science & Machine Learning pipeline** centered around the **Random Forest Classifier** to predict pulmonary disease risk levels (`PULMONARY_DISEASE`: `YES` / `NO`) using 5,000 patient clinical records.

---

## 📁 Repository Structure

```text
lung-cancer-prediction/
│
├── images/                   # Visualizations for EDA & README
│   ├── correlation_matrix.png
│   ├── smoking_relationship.png
│   └── oxygen_energy_boxplot.png
│
├── lung_cancer_dataset.csv   # Clinical dataset (5,000 records)
├── code.ipynb                # Main EDA & Model Training Notebook
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation

```

git clone [https://github.com/HoangKaze/Lung-Cancer.git](https://github.com/HoangKaze/Lung-Cancer.git)
cd Lung-Cancer
pip install -r requirements.txt
jupyter notebook code.ipynb
