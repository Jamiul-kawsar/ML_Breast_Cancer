# Breast Cancer Diagnostic Classification Pipeline

An end-to-end machine learning classification pipeline benchmarking multiple core algorithms on the **Breast Cancer Wisconsin Dataset**. This framework implements strict feature engineering safety standards to eradicate **data leakage** and produces automated performance visualizations.

---

## 📊 Project Architecture & Workflow

The pipeline is structurally engineered to guarantee unbiased testing parameters across data manipulation boundaries:

1. **Feature Pruning & Label Encoding**: Drops database keys (`id`) and raw metadata structural columns (`Unnamed: 32`), transforming textual target tags (`Malignant` / `Benign`) into unified numerical formats (`1` / `0`).
2. **Data Leakage Isolation**: Executes a stratified train-test split *prior* to calculating statistical feature parameters.
3. **Empirical Standardisation**: Applies `StandardScaler` to calculate variance scaling arrays strictly from the training cohort, avoiding real-world prediction shifts.
4. **Classifier Evaluation Suite**: Simultaneously trains and benchmarks 4 diverse algorithmic structures using identical environmental states.
5. **Metric Normalization**: Converts raw fractional decimal scores into standardized categorical percentages for visualization layouts.

---

## 🛠️ Evaluated Algorithms

- **Support Vector Machine (Linear SVM)**: Maps features onto high-dimensional spaces to find the optimal maximum-margin dividing hyperplane.
- **Logistic Regression**: Computes regularized maximum-likelihood probabilities via continuous sigmoidal curves.
- **K-Nearest Neighbors (KNN)**: Classifies test samples based on localized spatial proximity and distance cluster majorities.
- **Naive Bayes (Gaussian)**: Utilizes a conditional probabilistic approach assuming normal metric distributions across feature variables.

---

## 📈 Performance Analysis Benchmark

When executed on the test dataset split, the classifiers consistently evaluate to the following baseline structural metrics:

| Model | Test Accuracy (%) | Boundary Classification | Primary Sensitivity |
| :--- | :--- | :--- | :--- |
| **SVM (Linear)** | **~98.2%** | Linear Hyperplane | High stability, resilient to feature scaling |
| **Logistic Regression** | **~97.4%** | Probability Curve | Highly interpretable feature log-odds |
| **K-Nearest Neighbors** | **~96.5%** | Instance Neighborhood | Sensitive to local coordinate density |
| **Naive Bayes (Gaussian)** | **~94.7%** | Conditional Probability | Marginally affected by correlated metrics |

> 💡 **Medical Domain Note:** In clinical tumor classifications, minimizing **False Negatives (Recall)** is crucial to ensure malignant cases are not missed. While SVM yields the highest overall accuracy, individual class recall metrics should remain the focal parameter for final model deployment strategies.

---

## 🚀 Getting Started & Local Setup

### 1. Prerequisites
Ensure your local machine has Python installed. Clone this repository and run the installation string to set up all underlying data, computing, and plotting engines:

```bash
git clone https://github.com
cd YOUR_REPOSITORY_NAME
pip install -r requirements.txt
```

### 2. Execution Loop
To run the preprocessing, execute model training, and generate the comparative evaluation bar chart, run the controller file from your terminal:

```bash
python main.py
```

---
`
