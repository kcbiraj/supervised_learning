# 📈 Human Development Index (GDP) — Supervised Learning

This project explores the relationship between various socio-economic factors and their influence on a country's **GDP (Gross Domestic Product)** and **Human Development Index (HDI)**. By applying supervised learning models, we aim to build predictive models that quantify the effects of these variables and offer insights into national development trends.

---

## 🎯 Project Objective

The objectives of this project are:

- To analyze the relationship between socio-economic indicators and GDP/HDI.
- To build regression models that can predict GDP and HDI using machine learning.
- To evaluate and compare different regression techniques using performance metrics like RMSE.
- To apply cross-validation for robust model validation.

---

## 📊 Dataset Overview

The dataset includes the following key variables:

- **Total Cases** — Total confirmed disease-related cases.
- **Total Deaths** — Total deaths due to health-related issues.
- **Population** — Country’s population.
- **GDP Per Capita** — Gross Domestic Product per person.
- **Location** — Country name.
- **HDI (Human Development Index)** — A composite index measuring average achievement in key dimensions of human development.

---

## 🧪 Methodology

The dataset is divided into:

- **70% Training Data**  
- **30% Testing Data**

To enhance the reliability of the results, **K-Fold Cross Validation** is used during training. This helps reduce bias and variance in model evaluation.

### 🧠 Machine Learning Models Used:

- **Linear Regression (Vanilla)**
- **Lasso Regression** — L1 regularization (helps with feature selection)
- **Ridge Regression** — L2 regularization (shrinks coefficients)
- **ElasticNet Regression** — Combines L1 and L2

Each model is trained and validated to assess predictive performance.

---

## 🧮 Libraries Used

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression, Lasso, Ridge, RidgeCV, LassoCV, ElasticNetCV
from sklearn.pipeline import Pipeline

%matplotlib inline

## 📈 Model Performance (RMSE)

The **Root Mean Squared Error (RMSE)** is used to evaluate the model performance. A lower RMSE value indicates better predictive accuracy.

| Model       | RMSE      |
|-------------|-----------|
| Linear      | 1.397796  |
| Lasso       | 1.397007  |
| Ridge       | 1.397331  |
| ElasticNet  | 1.397227  |

---

### 📌 Interpretation

- All models exhibit **very close RMSE values**, indicating consistent performance across methods.
- **Lasso Regression** achieved the lowest RMSE, making it slightly more effective in this case.
- The similarity in performance suggests that:
  - The dataset is well-processed and scaled.
  - There is no major multicollinearity or overfitting.
  - All four models are appropriate for this problem space.

---

## ✅ Conclusion

This project demonstrates that multiple regression models — **Linear**, **Lasso**, **Ridge**, and **ElasticNet** — perform similarly well on this dataset. Regularized models offer additional robustness and potential for feature selection, with **Lasso Regression** showing a slight edge.
