# 🫁 Lung Cancer & Pulmonary Disease Risk Prediction using Random Forest

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/) [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) [![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Project Overview

Pulmonary diseases and lung cancer remain primary contributors to global medical mortality. Early detection through clinical screening and risk assessments plays a critical role in proactive treatment and improving patient survival rates.

This project focuses on building and evaluating **Machine Learning models**, centered around the **Random Forest Classifier**, to analyze clinical survey attributes, behavioral factors, and biological indicators to predict pulmonary disease risk levels.

---

## Research Objectives

The main objective of this project is to develop an accurate and reliable machine learning pipeline capable of predicting pulmonary disease/lung cancer risk based on patient clinical indicators, lifestyle factors, and physiological measurements.

The prediction task is formulated as a binary classification problem (`PULMONARY_DISEASE`: `YES` / `NO`).

Key project milestones include:

- **Clinical Data Processing:** Cleaning and preprocessing patient records containing physical metrics, behavioral habits, and physiological markers.
- **Exploratory Data Analysis:** Identifying critical risk factors and feature correlations associated with pulmonary risk.
- **Feature Engineering Pipeline:** Encoding categorical features and scaling continuous health indicators .
- **Comparative Model Evaluation:** Benchmarking the **Random Forest** ensemble model against baseline classifiers (**Decision Tree** and **Logistic Regression**) across multiple performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC).
- **Model Interpretation & Feature Importance:** Analyzing key features driving model decisions and identifying the primary risk factors contributing to pulmonary disease prediction.

---

## Dataset Description

The dataset consists of **5,000 clinical records** with **18 attributes** covering demographics, lifestyle, symptoms, and physiological metrics:

- **Demographics & Medical History:** `AGE`, `GENDER`, `FAMILY_HISTORY`, `SMOKING_FAMILY_HISTORY`, `LONG_TERM_ILLNESS`.
- **Behavioral & Environmental Factors:** `SMOKING`, `ALCOHOL_CONSUMPTION`, `EXPOSURE_TO_POLLUTION`.
- **Symptoms & Health Indicators:** `ENERGY_LEVEL`, `OXYGEN_SATURATION`, `BREATHING_ISSUE`, `THROAT_DISCOMFORT`, `CHEST_TIGHTNESS`, `FINGER_DISCOLORATION`, `MENTAL_STRESS`, `IMMUNE_WEAKNESS`, `STRESS_IMMUNE`.
- **Target Variable:** `PULMONARY_DISEASE` (`YES` / `NO`).

---

## System Architecture

The end-to-end Machine Learning pipeline architecture consists of five core layers:

<div align="center">

```text
       [ Raw CSV Data ]
              │
              ▼
[ 1. Data Ingestion & Preprocessing ]
              │
              ▼
[ 2. Exploratory Data Analysis ]
              │
              ▼
[ 3. Feature Engineering & Scaling ]
              │
              ▼
[ 4. Model Training & Benchmarking ]
              │
              ▼
[ 5. Model Evaluation & Insights ]
```

---

## Data Preprocessing & Feature Engineering

Data quality directly impacts model accuracy and generalization. A structured processing pipeline was executed to clean raw clinical records and generate domain-specific composite features prior to model training.

### Target Encoding & Data Hygiene

- **Target Transformation:** Converted categorical diagnosis labels (`PULMONARY_DISEASE`: `YES` / `NO`) into numerical targets (`PULMONARY_DISEASE_NUM`: `1` / `0`).
- **Data Integrity:** Verified zero missing values across all 5,000 clinical records and checked numerical feature ranges.

### Domain-Specific Feature Engineering

To capture cumulative clinical risk effects, **3 composite domain features** were engineered by aggregating correlated clinical indicators:

1. **`SMOKE_RISK`** (`SMOKING` + `SMOKING_FAMILY_HISTORY` + `EXPOSURE_TO_POLLUTION`):
   - Measures overall environmental and behavioral tobacco/pollution exposure level (Scale: $0$ to $3$).
2. **`RESP_SYMPTOMS`** (`BREATHING_ISSUE` + `THROAT_DISCOMFORT` + `CHEST_TIGHTNESS`):
   - Aggregates primary upper/lower respiratory tract physical symptoms (Scale: $0$ to $3$).
3. **`IMMUNE_STRESS`** (`IMMUNE_WEAKNESS` + `MENTAL_STRESS` + `STRESS_IMMUNE`):
   - Captures physiological strain and immune vulnerability interaction (Scale: $0$ to $3$).

After feature generation, the input matrix $X$ expanded to **20 total features**.

#### Sample Output of Engineered Features:

| Row Index | `SMOKE_RISK` | `RESP_SYMPTOMS` | `IMMUNE_STRESS` |
| :-------: | :----------: | :-------------: | :-------------: |
|   **0**   |      2       |        2        |        1        |
|   **1**   |      2       |        2        |        1        |
|   **2**   |      1       |        1        |        0        |
|   **3**   |      2       |        2        |        1        |
|   **4**   |      2       |        2        |        1        |

---

## Exploratory Data Analysis

Exploratory Data Analysis was conducted to uncover key risk factors, understand underlying statistical distributions, and analyze attribute correlations associated with pulmonary disease diagnostic outcomes.

### 1. Risk Factor Analysis: Smoking & Family History

![Smoking and Family History Relationships](./images/smoking_relationship.png)

- **Personal Smoking Habit (`SMOKING`):**
  - Shows the strongest direct relationship with pulmonary disease risk.
  - Non-smokers (`0`) exhibit an extremely low rate of pulmonary disease, with **1,523 healthy cases** vs. only **145 diseased cases**.
  - Smokers (`1`) account for **1,892 positive cases** (representing over **92.8%** of all diagnosed pulmonary disease patients in the dataset).
- **Family Smoking History (`SMOKING_FAMILY_HISTORY`):**
  - Having a family member who smokes significantly elevates personal risk.
  - Patients with a positive family smoking history (`1`) present a high infection/risk rate (**715 positive cases** out of 1,020 individuals, or **~70.1%**).

### 2. Continuous Health Metrics Distribution

![Oxygen Saturation and Energy Level Boxplots](./images/oxygen_energy_boxplot.png)

- **Blood Oxygen Saturation (`OXYGEN_SATURATION`):**
  - Both healthy (`NO`) and diseased (`YES`) cohorts display similar median oxygen saturation levels around **95.0%**, spanning between **90% and 100%**.
  - Outliers exist at both extremes ($<91\%$ and $>99\%$), indicating that static oxygen saturation alone does not serve as a standalone linear separator.
- **Physical Energy Level (`ENERGY_LEVEL`):**
  - Patients diagnosed with pulmonary disease (`YES`) demonstrate a slightly higher median energy level index (**56.75**) compared to non-diseased individuals (**53.72**).
  - Both groups exhibit normal bell-curve distributions spanning from **23.2 to 83.0**.

### 3. Feature Correlation Matrix

![Correlation Heatmap](./images/correlation_matrix.png)

Analysis of the Pearson correlation heatmap with the target variable (`PULMONARY_DISEASE_NUM`) highlights key insights:

**Primary Predictors (Highest Positive Correlation):**

- **`SMOKING` ($r = 0.46$):** The single most dominant linear predictor of disease presence.
- **`SMOKING_FAMILY_HISTORY` ($r = 0.30$):** Strong secondary environmental/hereditary risk indicator.
- **`THROAT_DISCOMFORT` ($r = 0.28$) & `BREATHING_ISSUE` ($r = 0.27$):** Primary physical symptoms linked to positive diagnosis.
- **`STRESS_IMMUNE` ($r = 0.18$) & `ENERGY_LEVEL` ($r = 0.17$):** Secondary physiological and stress-related markers.
  **Collinearity & Interaction Insights:**
- High inter-feature correlation between `FAMILY_HISTORY` and `SMOKING_FAMILY_HISTORY` ($r = 0.77$).
- Strong correlation between `IMMUNE_WEAKNESS` and `STRESS_IMMUNE` ($r = 0.64$), as well as `MENTAL_STRESS` and `STRESS_IMMUNE` ($r = 0.48$), validating the creation of aggregate composite features.
- **Low Correlation Variables:** Demographic variables such as `AGE` ($r = -0.01$) and `GENDER` ($r = -0.00$) show negligible linear correlation with the target variable.

---

## Dataset Partitioning

To ensure an unbiased evaluation of the Machine Learning models and prevent data leakage, the processed dataset ($N = 5,000$) was partitioned into independent training and evaluation subsets.

### 1. Splitting Strategy

- **Training Set ($80\%$ / $4,000$ samples):** Used to fit the parameters of the Random Forest and baseline classifiers during training.
- **Testing Set ($20\%$ / $1,000$ samples):** Reserved exclusively as unseen data to evaluate final generalization performance.
- **Stratified Sampling (`stratify=y`):** Applied to maintain the exact class distribution of the target variable (`PULMONARY_DISEASE_NUM`) in both training and test sets ($59.26\%$ `NO` vs. $40.74\%$ `YES`), avoiding potential class imbalance bias during evaluation.
- **Reproducibility (`random_state=42`):** Set a fixed random seed to ensure consistent data splitting across experiments.

---

### 2. Class Distribution Summary

| Data Subset               | Total Samples | Healthy Class (`0` / `NO`) | Disease Class (`1` / `YES`) |
| :------------------------ | :-----------: | :------------------------: | :-------------------------: |
| **Full Dataset**          |    $5,000$    |    $2,963$ ($59.26\%$)     |     $2,037$ ($40.74\%$)     |
| **Training Set ($80\%$)** |    $4,000$    |    $2,370$ ($59.25\%$)     |     $1,630$ ($40.75\%$)     |
| **Testing Set ($20\%$)**  |    $1,000$    |     $593$ ($59.30\%$)      |      $407$ ($40.70\%$)      |

---

## Machine Learning Models

Five classification algorithms were trained and evaluated using **Scikit-Learn, XGBoost, and LightGBM** pipelines.

| Model                   | Description                                                                         |
| :---------------------- | :---------------------------------------------------------------------------------- |
| **Logistic Regression** | Baseline linear classifier evaluating log-odds relationships                        |
| **Decision Tree**       | Rule-based non-linear tree classifier                                               |
| **XGBoost**             | Optimized gradient boosting framework designed for speed and performance            |
| **LightGBM**            | Fast, leaf-wise tree boosting classifier optimized for large feature spaces         |
| **Random Forest**       | Ensemble tree-based bagging classifier reducing variance and preventing overfitting |

---

## Experimental Results & Model Benchmarking

To evaluate the predictive capability of the models, all five algorithms were benchmarked on the independent test set ($20\%$ holdout) across key classification metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.

---

### Comparative Performance Summary

| Model                   | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |  ROC-AUC   |
| :---------------------- | :----------: | :-----------: | :--------: | :----------: | :--------: |
| **Decision Tree**       |    80.50%    |    76.24%     |   75.68%   |    75.96%    |   0.7974   |
| **Logistic Regression** |    88.80%    |    85.04%     |   87.96%   |    86.47%    |   0.9260   |
| **XGBoost**             |    90.40%    |    88.40%     |   87.96%   |    88.18%    |   0.9139   |
| **LightGBM**            |    90.40%    |    88.59%     |   87.71%   |    88.15%    |   0.9199   |
| **Random Forest ⭐**    |  **90.60%**  |  **88.45%**   | **88.45%** |  **88.45%**  | **0.9232** |

### Key Empirical Insights

1. **Top Predictive Performance:**
   **Random Forest** achieved the highest overall performance with an **Accuracy of 90.60%** and an **F1-Score of 88.45%**, proving to be the most reliable model for pulmonary disease prediction.
2. **Balanced Sensitivity & Specificity:**
   **Random Forest** demonstrated an optimal balance between **Precision (88.45%)** and **Recall (88.45%)**, ensuring minimal false negatives (missing sick patients) while avoiding high false alarm rates.
3. **Gradient Boosting Benchmarks:**
   Both **XGBoost** and **LightGBM** delivered strong competitive results (**90.40% Accuracy**), slightly behind Random Forest, highlighting the effectiveness of ensemble learning architectures on clinical feature spaces.
4. **Baseline Model Evaluation:**
   **Logistic Regression** produced a solid linear baseline with a strong ROC-AUC score ($0.9260$), whereas a single **Decision Tree** struggled with high variance, yielding the lowest accuracy ($80.50\%$).

---

## Repository Structure

```text
lung-cancer-prediction/
│
├── images/
│   ├── correlation_matrix.png
│   ├── smoking_relationship.png
│   └── oxygen_energy_boxplot.png
│
├── Lung Cancer Dataset.csv
├── code.ipynb
└── README.md
```

---

## Technologies Used

| Category                    | Technology                                                           |
| :-------------------------- | :------------------------------------------------------------------- |
| **Language**                | Python 3.10+                                                         |
| **Machine Learning Models** | Decision Tree, Logistic Regression, XGBoost, LightGBM, Random Forest |
| **ML Frameworks**           | Scikit-Learn, XGBoost, LightGBM                                      |
| **Visualization**           | Matplotlib, Seaborn                                                  |
| **Scientific Computing**    | NumPy, Pandas                                                        |
| **Notebook**                | Jupyter Notebook                                                     |
| **Development**             | Git, GitHub, VS Code                                                 |

---

## Conclusion & Future Work

### Conclusion

This project successfully developed an end-to-end Machine Learning pipeline to predict pulmonary disease risk using clinical, behavioral, and physiological indicators.

Key takeaways include:

- **Feature Engineering Impact:** Aggregating domain-specific composite features (`SMOKE_RISK`, `RESP_SYMPTOMS`, `IMMUNE_STRESS`) significantly enhanced model feature representation.
- **Model Superiority:** Among the five evaluated algorithms, **Random Forest Classifier** emerged as the top-performing model, achieving **90.60% Accuracy**, **88.45% Precision**, **88.45% Recall**, and an **88.45% F1-Score**.
- **Clinical Insights:** Behavioral factors (specifically personal and family smoking history) alongside respiratory symptoms (`THROAT_DISCOMFORT`, `BREATHING_ISSUE`) were identified as the strongest risk predictors for pulmonary conditions.

### Future Work

- **Web Application Deployment:** Deploy the trained Random Forest model as an interactive web API using **Streamlit** or **Flask** for real-time clinical decision support.
- **Advanced Hyperparameter Tuning:** Explore Bayesian Optimization (`Optuna`) to further fine-tune model parameters.
- **Model Interpretability:** Integrate SHAP values to provide local explainability for individual patient risk predictions.
