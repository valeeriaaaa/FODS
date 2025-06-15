import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, make_scorer, f1_score
from imblearn.over_sampling import SMOTE

def get_confusion_matrix(y,y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y)):
        if y.iloc[i] == 0 and y_pred [i] == 0:
            tn += 1
        if y.iloc[i] == 0 and y_pred[i] == 1:
            fp += 1
        if y.iloc[i] == 1 and y_pred[i] == 0:
            fn += 1
        if y.iloc[i] == 1 and y_pred[i] == 1:
            tp += 1


    return tn, fp, fn, tp

def evaluation_metrics(model,y, y_pred, X):
    tn, fp, fn, tp = get_confusion_matrix(y,y_pred)

    precision   = tp / (tp + fp)
    specificity = tn / (tn + fp)
    accuracy    = (tn + tp) / (tn + tp + fn + fp)
    recall      = tp / (tp + fn)
    f1          = tp / (tp + 0.5 * (fn + fp))

    y_predict_proba  = model.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y, y_predict_proba)

    roc_auc = auc(fpr, tpr)

    return [accuracy, precision, recall, specificity, f1, roc_auc], (fpr, tpr)


### inspection of data ###
df = pd.read_csv("schizophrenia_dataset.csv")
print(data.head())
print(data.columns)
print(data.info())
for col in data.columns:
    print(col, data[col].unique())
print(data.describe())
print(data.isnull().sum())


### cleaning ###
data = data.drop(['Patient_ID', 'Disease_Duration', 'Hospitalizations',
                  'Positive_Symptom_Score', 'Negative_Symptom_Score', 'GAF_Score'], axis=1)
# Patient_ID has no valuable information and the other columns are already associated with a diagnosis,
# so they won't be considered in this project.


### Data Splitting ###
X = data.drop("Diagnosis", axis=1)      # axis=1 muss sein weil bei data["Diagnosis"] gibts value error
y = data["Diagnosis"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_train_raw, y_train, test_size=0.25, random_state=42, stratify=y_train)


### one-hot encoding ###
nominal_columns = ['Marital_Status', 'Occupation'] # all the others are ordinal or binary
X_train_encoded = pd.get_dummies(X_train_raw, columns=nominal_columns, drop_first=False, dtype=int)
#X_val_encoded  = pd.get_dummies(X_val_raw, columns=nominal_columns, drop_first=False, dtype=int)
X_test_encoded = pd.get_dummies(X_test_raw, columns=nominal_columns, drop_first=False, dtype=int)

nominal_columns_after_ohe = ['Marital_Status_0', 'Marital_Status_1', 'Marital_Status_2', 'Marital_Status_3',
                             'Occupation_0', 'Occupation_1', 'Occupation_2', 'Occupation_3']

num_columns = ['Age', 'Gender', 'Education_Level', 'Income_Level', 'Living_Area', 'Family_History',
               'Substance_Use', 'Suicide_Attempt', 'Social_Support', 'Stress_Factors', 'Medication_Adherence']

### Scaling ###
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_encoded[num_columns]), columns=num_columns)
#X_val_scaled = pd.DataFrame(scaler.transform(X_val_encoded[num_columns]), columns=num_columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_encoded[num_columns]), columns=num_columns)


### put together ###
X_train = pd.concat([X_train_scaled.reset_index(drop=True), X_train_encoded[nominal_columns_after_ohe].reset_index(drop=True)], axis=1)
#X_val = pd.concat([X_val_scaled.reset_index(drop=True), X_val_encoded[nominal_columns_after_ohe].reset_index(drop=True)], axis=1)
X_test = pd.concat([X_test_scaled.reset_index(drop=True), X_test_encoded[nominal_columns_after_ohe].reset_index(drop=True)], axis=1)



### Model training ###
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("Baseline test accuracy:", model.score(X_test, y_test))


### Hyperparameter tuning ###
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [1, 10, 100, 1000],
    'solver': ['liblinear', 'saga'],
    'random_state': [42],
    'max_iter': [100]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=make_scorer(f1_score, pos_label=1),
    n_jobs=-1,
    cv=5,
    verbose=1,
    refit=True
)


grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)


y_pred = best_model.predict(X_test)
names = ["No Schizophrenia", "Scizophrenia"]
print(classification_report(y_test, y_pred, target_names= names))
print(evaluation_metrics(best_model,y_test, y_pred, X_test))



disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred), display_labels = ["No Schizophrenia", "Schizophrenia"])
fig, ax = plt.subplots()
disp.plot(cmap="Blues", values_format='d', ax=ax)
plt.title("Confusion Matrix")
ax.set_yticklabels(disp.display_labels, rotation=90, va='center')
plt.savefig('../output/confusion_matrix.png')
plt.show()


### SMOTE ###
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)


### Model training ###
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("Baseline test accuracy:", model.score(X_test, y_test))


### Hyperparameter tuning ###
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)


y_pred = best_model.predict(X_test)
names = ["No Schizophrenia", "Scizophrenia"]
print(classification_report(y_test, y_pred, target_names= names))
print(evaluation_metrics(best_model,y_test, y_pred, X_test))


disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred), display_labels = ["No Schizophrenia", "Schizophrenia"])
fig, ax = plt.subplots()
disp.plot(cmap="Blues", values_format='d', ax=ax)
plt.title("Confusion Matrix")
ax.set_yticklabels(disp.display_labels, rotation=90, va='center')
plt.savefig('../output/confusion_matrix_with_SMOTE.png')
plt.show()
