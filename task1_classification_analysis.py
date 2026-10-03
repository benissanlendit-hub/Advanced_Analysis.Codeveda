"""
Task 1: Predictive Modeling (Classification)
Objective: Build and evaluate a classification model to predict
categorical outcomes (customer churn: True/False).
Tools: Python, scikit-learn, pandas, matplotlib

Dataset: BigML Telecom Churn dataset, already split by the provider into:
  - churn-bigml-80.csv  (2666 rows) -> training set
  - churn-bigml-20.csv  (667 rows)  -> testing set
We use this official split directly instead of re-splitting ourselves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# ---------------------------------------------------------------------
# 1. Load the official train/test split
# ---------------------------------------------------------------------
train_df = pd.read_csv("churn-bigml-80.csv")
test_df = pd.read_csv("churn-bigml-20.csv")

print("Training set shape:", train_df.shape)
print("Testing set shape: ", test_df.shape)
print("\nChurn distribution (train):")
print(train_df["Churn"].value_counts(normalize=True).round(3))

# ---------------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------------

def preprocess(df):
    df = df.copy()
    # Target: bool -> int (0/1)
    df["Churn"] = df["Churn"].astype(int)
    # Binary categorical variables: Yes/No -> 1/0
    df["International plan"] = df["International plan"].map({"Yes": 1, "No": 0})
    df["Voice mail plan"] = df["Voice mail plan"].map({"Yes": 1, "No": 0})
    return df

train_df = preprocess(train_df)
test_df = preprocess(test_df)

# One-hot encode multi-category variables (State, Area code)
train_encoded = pd.get_dummies(train_df, columns=["State", "Area code"], drop_first=True)
test_encoded = pd.get_dummies(test_df, columns=["State", "Area code"], drop_first=True)

# Align test columns to training columns (in case a State value is missing
# from one of the two files), filling any missing dummy column with 0
test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

# Separate features (X) and target (y)
X_train = train_encoded.drop(columns=["Churn"])
y_train = train_encoded["Churn"]
X_test = test_encoded.drop(columns=["Churn"])
y_test = test_encoded["Churn"]

# Feature scaling: standardize the originally continuous numeric columns
# (mean 0, std 1) - important for Logistic Regression, harmless for trees
continuous_cols = [
    "Account length", "Number vmail messages", "Total day minutes",
    "Total day calls", "Total day charge", "Total eve minutes",
    "Total eve calls", "Total eve charge", "Total night minutes",
    "Total night calls", "Total night charge", "Total intl minutes",
    "Total intl calls", "Total intl charge", "Customer service calls",
]

scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

print("\nFinal feature matrix shape (train):", X_train.shape)
print("Final feature matrix shape (test): ", X_test.shape)

# ---------------------------------------------------------------------
# 3. Train and evaluate multiple baseline classification models
# ---------------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
    }

baseline_results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    result = evaluate(name, model, X_test, y_test)
    baseline_results.append(result)

baseline_df = pd.DataFrame(baseline_results).set_index("Model").round(4)
print("\n--- Baseline Model Comparison (default hyperparameters) ---")
print(baseline_df)

# ---------------------------------------------------------------------
# 4. Hyperparameter tuning with GridSearchCV
# ---------------------------------------------------------------------

param_grids = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"],
    },
    "Decision Tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    },
}

base_estimators = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

tuned_results = []
best_models = {}

for name, estimator in base_estimators.items():
    print(f"\nRunning GridSearchCV for {name}...")
    grid = GridSearchCV(
        estimator,
        param_grids[name],
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_

    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validated F1-score: {grid.best_score_:.4f}")

    result = evaluate(name, grid.best_estimator_, X_test, y_test)
    tuned_results.append(result)

tuned_df = pd.DataFrame(tuned_results).set_index("Model").round(4)
print("\n--- Tuned Model Comparison (best hyperparameters, test set) ---")
print(tuned_df)

# ---------------------------------------------------------------------
# 5. Detailed report for the best model (highest F1-score after tuning)
# ---------------------------------------------------------------------
best_model_name = tuned_df["F1-score"].idxmax()
best_model = best_models[best_model_name]

print(f"\n--- Best Model: {best_model_name} ---")
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best, target_names=["No Churn", "Churn"]))

cm = confusion_matrix(y_test, y_pred_best)
print("Confusion matrix:")
print(cm)

# ---------------------------------------------------------------------
# 6. Visualizations
# ---------------------------------------------------------------------

# 6a. Bar chart: baseline vs tuned F1-score per model
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(baseline_df.index))
width = 0.35
ax.bar(x - width/2, baseline_df["F1-score"], width, label="Baseline (default params)", color="lightsteelblue")
ax.bar(x + width/2, tuned_df["F1-score"], width, label="Tuned (GridSearchCV)", color="steelblue")
ax.set_xticks(x)
ax.set_xticklabels(baseline_df.index)
ax.set_ylabel("F1-score")
ax.set_title("Model Comparison: Baseline vs Tuned (F1-score)")
ax.legend()
plt.tight_layout()
plt.savefig("01_model_comparison_f1.png", dpi=150)
plt.show()

# 6b. Confusion matrix heatmap for the best model
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["No Churn", "Churn"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["No Churn", "Churn"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_model_name}")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
plt.tight_layout()
plt.savefig("02_confusion_matrix.png", dpi=150)
plt.show()

print("\nPlots saved: 01_model_comparison_f1.png, 02_confusion_matrix.png")
print(f"\nBest performing model after tuning: {best_model_name}")
