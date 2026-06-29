# Machine Failure Prediction Using Machine Learning

## 📌 Project Overview

This project predicts whether an industrial machine is likely to fail using **Machine Learning**. The model is trained on a predictive maintenance dataset containing machine operating parameters such as temperature, rotational speed, torque, and tool wear.

The project uses **Logistic Regression** with both **L1 (Lasso)** and **L2 (Ridge)** regularization techniques and compares their performance.

---

# 📂 Dataset

The dataset used is:

**predictive_maintenance_dataset.csv**

### Dataset Features

| Feature | Description |
|---------|-------------|
| UDI | Unique Identifier |
| Product ID | Product Identifier |
| Type | Machine Type (L, M, H) |
| Air temperature [K] | Air temperature in Kelvin |
| Process temperature [K] | Process temperature in Kelvin |
| Rotational speed [rpm] | Machine rotational speed |
| Torque [Nm] | Torque produced |
| Tool wear [min] | Tool wear in minutes |
| Machine failure | Target Variable |
| TWF | Tool Wear Failure |
| HDF | Heat Dissipation Failure |
| PWF | Power Failure |
| OSF | Overstrain Failure |
| RNF | Random Failure |

---

# 🎯 Objective

The main objective of this project is to:

- Predict machine failures
- Perform data preprocessing
- Train Logistic Regression models
- Compare L1 and L2 Regularization
- Evaluate model performance

---

# 🛠 Technologies Used

- Python
- Google Colab
- NumPy
- Pandas
- Scikit-Learn

---

# 📚 Libraries Used

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
```

---

# ⚙️ Workflow

## 1. Import Libraries

Necessary Python libraries are imported for data preprocessing, model training, and evaluation.

---

## 2. Upload Dataset

Dataset is uploaded using Google Colab.

```python
from google.colab import files
uploaded = files.upload()
```

---

## 3. Load Dataset

The CSV dataset is loaded into a Pandas DataFrame.

```python
data = pd.read_csv("predictive_maintenance_dataset.csv")
```

---

## 4. Data Exploration

The first few rows of the dataset are displayed.

```python
data.head()
```

---

## 5. Feature Selection

Target column:

```
Machine failure
```

Removed columns:

- Product ID
- UDI
- TWF
- HDF
- PWF
- OSF

---

## 6. One-Hot Encoding

The categorical column **Type** is converted into numerical values using:

```python
pd.get_dummies()
```

---

## 7. Train-Test Split

Dataset is divided into training and testing sets.

```python
train_test_split()
```

Training: 80%

Testing: 20%

---

## 8. Feature Scaling

Features are standardized using:

```python
StandardScaler()
```

---

## 9. Model Training

Two Logistic Regression models are trained.

### Logistic Regression (L1 Regularization)

```python
penalty='l1'
solver='liblinear'
```

---

### Logistic Regression (L2 Regularization)

```python
penalty='l2'
solver='lbfgs'
```

---

# 📈 Model Evaluation

The models are evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score
- Classification Report

Example:

```python
accuracy_score()
confusion_matrix()
classification_report()
precision_score()
recall_score()
f1_score()
```

---

# 📊 Machine Learning Concepts Covered

This project covers the following topics:

- Supervised Learning
- Binary Classification
- Logistic Regression
- L1 Regularization (Lasso)
- L2 Regularization (Ridge)
- Feature Engineering
- Feature Selection
- One-Hot Encoding
- Standardization
- Train-Test Split
- Model Evaluation
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1 Score
- Predictive Maintenance
- Industrial AI

---

---

# 🚀 Future Improvements

- Support Vector Machine (SVM)
- Random Forest Classifier
- Decision Tree
- XGBoost
- LightGBM
- Hyperparameter Tuning
- Cross Validation
- Feature Importance Visualization
- ROC Curve
- Precision-Recall Curve
- Model Deployment using Flask
- Streamlit Web Application

---

# 📌 Results

The Logistic Regression model successfully predicts machine failures after preprocessing and feature scaling.

Both:

- Logistic Regression (L1)
- Logistic Regression (L2)

were trained and evaluated using the testing dataset.

Performance metrics such as Accuracy and Confusion Matrix were used to compare the models.

---

# 🎓 Learning Outcomes

After completing this project, you will understand:

- How predictive maintenance works
- Data preprocessing techniques
- Handling categorical variables
- Feature scaling
- Logistic Regression
- L1 vs L2 Regularization
- Binary Classification
- Model Evaluation Metrics
- End-to-end Machine Learning workflow

---

# 👨‍💻 Author

**Vivek Sharma**

Machine Learning Project

Machine Failure Prediction Using Logistic Regression

---

# ⭐ If you found this project useful, consider giving it a Star!
