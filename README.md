#  Credit Card Default Risk Prediction - Behavior Score Model

## 📌 Problem Statement

Bank A aims to enhance its risk management framework by developing a **Behavior Score** model—a machine learning system to predict the **probability of default** by existing credit card customers. This model is crucial for:

- Reducing customer default rates  
- Improving financial stability  
- Supporting informed credit decisions and portfolio risk management  

---

## 📊 Dataset Overview

The dataset is a **large-scale credit card transaction dataset** with a **very high number of features**. It includes both numerical and categorical features, and targets whether a customer is likely to **default** (`bad_flag = 1`).

---

## 🧹 Data Preprocessing & Cleaning

Due to the dataset's complexity and size, extensive preprocessing was done:

### 🧱 Feature Reduction
- Removed **non-informative** columns  
- Dropped features with **>50% missing values**  
- Removed **highly correlated features** (correlation > 0.85)  
- Used **XGBoost** for feature importance and selected **top 50** features  

### 📉 Handling Missing Values
- Used **skewness-based strategy**:
  - **Skewed** → Median  
  - **Normal** → Mean  

---

## ⚖️ Class Imbalance Handling

The dataset had a **highly imbalanced target** (default vs non-default):

- Used **oversampling** instead of undersampling to prevent data loss  
- Tried additional balancing techniques:
  - **Tomek Links**, **Edited Nearest Neighbors (ENN)**, **Random Undersampling**, **SMOTE**
  - **ClusterCentroids** for synthetic balancing  

---

## 📈 Exploratory Data Analysis (EDA)

- Visualized distributions with **histograms**  
- Used **correlation heatmaps** to reduce redundancy  
- Identified features with strong separation between classes (e.g., `onus_attribute_2`)  

---

## 🧠 Models Used

### ✅ Final Model: **XGBoost**
- Best performing model with:
  - **AUC-ROC**: 0.9845  
  - **Accuracy**: 93.56%  
  - **F1 Score**: 0.9383  

### 🧪 Other Models:
- **Balanced Random Forest** on different preprocessed datasets  
- Compared using:
  - **ROC-AUC**
  - **Precision, Recall, F1-Score**
  - **PR-AUC** (important for imbalanced datasets)

---

## 📌 Final Results & Insights

- **XGBoost** provided strong classification performance  
- Excellent detection of defaulters (**Recall = 97%**)  
- High precision for non-defaulters (**Precision = 97%**)  
- Balanced and generalizable model suitable for deployment  

---

## 🏁 How to Run

1. Place the dataset in the correct directory: `train_data/train_data.csv`  
2. Open `FinalApproach.ipynb` using Jupyter or any compatible IDE  
3. Run the notebook step by step to observe preprocessing, training, and evaluation  

---

## 📁 Project Structure

