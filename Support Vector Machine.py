import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("schizophrenia_dataset.csv")

# Select features and target
selected_features = [
    "Age", "Education_Level", "Marital_Status",
    "Occupation", "Suicide_Attempt", "Family_History"
]
X = df[selected_features]
y = df["Diagnosis"]

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=["Education_Level", "Marital_Status", "Occupation"], drop_first=True)

# Fill any potential missing values
X = X.fillna(X.mean())

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1],
    'gamma': ['auto'],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

### 1. Train SVM WITHOUT SMOTE
grid_no_smote = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1
)
grid_no_smote.fit(X_train_scaled, y_train)
best_no_smote = grid_no_smote.best_estimator_
y_pred_no_smote = best_no_smote.predict(X_test_scaled)
y_proba_no_smote = best_no_smote.predict_proba(X_test_scaled)[:, 1]

# Print classification report and confusion matrix
print("=== Classification Report WITHOUT SMOTE ===")
print(classification_report(y_test, y_pred_no_smote))
cm_no_smote = confusion_matrix(y_test, y_pred_no_smote)

### 2. Train SVM WITH SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
X_train_sm_scaled = scaler.fit_transform(X_train_sm)

grid_smote = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1
)
grid_smote.fit(X_train_sm_scaled, y_train_sm)
best_smote = grid_smote.best_estimator_
y_pred_smote = best_smote.predict(X_test_scaled)
y_proba_smote = best_smote.predict_proba(X_test_scaled)[:, 1]

# Print classification report and confusion matrix
print("=== Classification Report WITH SMOTE ===")
print(classification_report(y_test, y_pred_smote))
cm_smote = confusion_matrix(y_test, y_pred_smote)

# --- Plot Confusion Matrices ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.heatmap(cm_no_smote, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Schizophrenia", "Schizophrenia"],
            yticklabels=["No Schizophrenia", "Schizophrenia"])
plt.title("Confusion Matrix - Without SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 2, 2)
sns.heatmap(cm_smote, annot=True, fmt="d", cmap="Greens",
            xticklabels=["No Schizophrenia", "Schizophrenia"],
            yticklabels=["No Schizophrenia", "Schizophrenia"])
plt.title("Confusion Matrix - With SMOTE")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# --- Plot ROC Curves ---
fpr_no_smote, tpr_no_smote, _ = roc_curve(y_test, y_proba_no_smote)
fpr_smote, tpr_smote, _ = roc_curve(y_test, y_proba_smote)
auc_no_smote = auc(fpr_no_smote, tpr_no_smote)
auc_smote = auc(fpr_smote, tpr_smote)

plt.figure(figsize=(6, 5))
plt.plot(fpr_no_smote, tpr_no_smote, label=f"Without SMOTE (AUC = {auc_no_smote:.2f})", color='blue')
plt.plot(fpr_smote, tpr_smote, label=f"With SMOTE (AUC = {auc_smote:.2f})", color='green')
plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()
