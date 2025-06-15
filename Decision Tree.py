import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from imblearn.pipeline import Pipeline


# Load dataset
df = pd.read_csv("../data/2025_schizophrenia_dataset.csv")

# Drop non-predictive identifier
df = df.drop(columns=['Patient_ID'])
df = df.drop(columns=['Disease_Duration'])


# Define features and target
X = df[["Age", "Gender", "Education_Level", "Marital_Status", "Occupation",
"Income_Level", "Living_Area", "Family_History", "Substance_Use",
"Suicide_Attempt", "Social_Support", "Stress_Factors"]]  # All features
y = df['Diagnosis']  # Target variable

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


#Hyperparameter tuning
dt = DecisionTreeClassifier(random_state=42)

param_grid = {
    'max_depth':      [None, 5, 10, 20],
    'min_samples_leaf': [5, 10, 25],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy","log_loss"]
}

grid = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best parameters  :", grid.best_params_)
print("Best CV F1 bal.acc. :", round(grid.best_score_, 3))

best_clf = grid.best_estimator_

# Make predictions on the held-out test set
y_pred = best_clf.predict(X_test)



# ROC-AUC curve
y_prob = best_clf.predict_proba(X_test)[:, 1]   # get probability for class 1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

plt.savefig("../output/ROC_Curve.png", dpi=100)
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=best_clf.classes_)

# Plot
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Schizophrenia", "Schizophrenia"],
            yticklabels=["No Schizophrenia", "Schizophrenia"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("../output/ConfusionMatrix.png", dpi=100)


# Get feature importances
importances = best_clf.feature_importances_
feature_names = X.columns

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df.head(10))  # Top 10 important features

print("Best parameters  :", grid.best_params_)


# y_test = true labels, y_pred = predictions from the model
report = classification_report(y_test, y_pred, target_names=["No Schizophrenia", "Schizophrenia"])

print(report)






#apply smote and do it all again


#Hyperparameter tuning
pipe = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("tree", DecisionTreeClassifier(random_state=42))
])

param_grid = {
    'tree__max_depth':      [None, 5, 10, 20],
    'tree__min_samples_leaf': [ 5, 10, 25],
    "tree__min_samples_split": [2, 5, 10],
    "tree__criterion": ["gini", "entropy","log_loss"]
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best parameters  :", grid.best_params_)
print("Best CV F1 bal.acc. :", round(grid.best_score_, 3))

best_clf = grid.best_estimator_
tree = best_clf.named_steps["tree"]

# Make predictions on the held-out test set
y_pred = best_clf.predict(X_test)



# ROC-AUC curve
y_prob = best_clf.predict_proba(X_test)[:, 1]   # get probability for class 1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

plt.savefig("../output/ROC_Curve_SMOTE.png", dpi=100)
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=tree.classes_)

# Plot
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Schizophrenia", "Schizophrenia"],
            yticklabels=["No Schizophrenia", "Schizophrenia"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("../output/ConfusionMatrixSMOTE.png", dpi=100)


# Get feature importances
importances = tree.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df.head(10))  # Top 10 important features

print("Best parameters  :", grid.best_params_)


# y_test = true labels, y_pred = predictions from the model
report = classification_report(y_test, y_pred, target_names=["No Schizophrenia", "Schizophrenia"])

print(report)