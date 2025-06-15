import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


#load data
df = pd.read_csv('../Group project/2025_schizophrenia_dataset.csv',index_col=0)



selected_features = [
    "Age", "Gender", "Education_Level", "Marital_Status", "Occupation",
    "Income_Level", "Living_Area", "Family_History", "Substance_Use",
    "Suicide_Attempt", "Social_Support", "Stress_Factors"
]


X = df[selected_features]
Y = df['Diagnosis']

#onehotencoding
X = pd.get_dummies(X)


#data splitting
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train,X_Val,Y_train,Y_val =  train_test_split(X_train_val, Y_train_val, test_size=0.25, random_state=42)

#feature selection
rf = RandomForestClassifier(random_state=0, class_weight='balanced')
rf.fit(X_train, Y_train)

importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("Feature Importances:\n", importances)
top_features = importances.head(6).index.tolist()


#reduction of featureset
X_train_sel = X_train[top_features]
X_val_sel = X_Val[top_features]
X_test_sel = X_test[top_features]

#hyperparametertuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}
#model without SMOTE
grid_search_nosmote = GridSearchCV(
    estimator=RandomForestClassifier(random_state=0, class_weight='balanced'),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

grid_search_nosmote.fit(X_train_sel, Y_train)
best_rf_nosmote = grid_search_nosmote.best_estimator_
print("Best parameter no SMOTE:", grid_search_nosmote.best_params_)

rf_nosmote = RandomForestClassifier(random_state=0, class_weight='balanced')
rf_nosmote.fit(X_train_sel, Y_train)

y_pred_nosmote = rf_nosmote.predict(X_test_sel)
y_proba_nosmote = rf_nosmote.predict_proba(X_test_sel)[:, 1]

print("\n Model without SMOTE:")
print("Precision Score:", precision_score(Y_test, y_pred_nosmote))
print("ROC AUC Score:", roc_auc_score(Y_test, y_proba_nosmote))
print(classification_report(Y_test, y_pred_nosmote, target_names=['No Schizophrenia', 'Schizophrenia']))


# Confusion Matrix without SMOTE
cm_nosmote = confusion_matrix(Y_test, y_pred_nosmote)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_nosmote, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Schizophrenia", "Schizophrenia"],
            yticklabels=["No Schizophrenia", "Schizophrenia"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve no SMOTE
fpr_ns, tpr_ns, _ = roc_curve(Y_test, y_proba_nosmote)
plt.figure(figsize=(8, 6))
plt.plot(fpr_ns, tpr_ns, label=f'without SMOTE (AUC = {roc_auc_score(Y_test, y_proba_nosmote):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (without SMOTE)")
plt.legend()
plt.grid()
plt.show()



#SMOTE
smote = SMOTE(random_state=42)
X_train_smote, Y_train_smote = smote.fit_resample(X_train_sel, Y_train)

#scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test_sel)


grid_search_smote = GridSearchCV(
    estimator=RandomForestClassifier(random_state=0, class_weight='balanced'),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

grid_search_smote.fit(X_train_scaled, Y_train_smote)
best_rf_smote = grid_search_smote.best_estimator_
print("Best parameter with SMOTE:", grid_search_smote.best_params_)


#evaluation
y_pred_smote = best_rf_smote.predict(X_test_sel)
y_proba_smote = best_rf_smote.predict_proba(X_test_sel)[:, 1]

# Scores
print("\n Model with SMOTE:")
print("Precision Score:", precision_score(Y_test, y_pred_smote))
print("ROC AUC Score:", roc_auc_score(Y_test, y_proba_smote))
print(classification_report(Y_test, y_pred_smote, target_names=['No Schizophrenia', 'Schizophrenia']))




# ROC-curve
fpr, tpr, _ = roc_curve(Y_test, y_proba_smote)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(Y_test, y_proba_smote):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()


#confusion matrix
cm = confusion_matrix(Y_test, y_pred_smote)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Schizophrenia", "Schizophrenia"],
            yticklabels=["No Schizophrenia", "Schizophrenia"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
