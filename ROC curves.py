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
df = pd.read_csv("schizophrenia_dataset.csv")

# Drop non-predictive identifier
df = df.drop(columns=['Patient_ID'])
df = df.drop(columns=['Disease_Duration'])


# Define features and target
X = df[["Age",  "Education_Level", "Marital_Status", "Occupation",
 "Family_History",
"Suicide_Attempt", ]]
y = df['Diagnosis']

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

fpr_decision_tree, tpr_decision_tree, thresholds = roc_curve(y_test, y_prob)
roc_auc_decision_tree = auc(fpr_decision_tree, tpr_decision_tree)




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

best_clf = grid.best_estimator_
tree = best_clf.named_steps["tree"]

# Make predictions on the held-out test set
y_pred = best_clf.predict(X_test)

# ROC-AUC curve
y_prob = best_clf.predict_proba(X_test)[:, 1]   # get probability for class 1

fpr_decision_tree_smote, tpr_decision_tree_smote, thresholds = roc_curve(y_test, y_prob)
roc_auc_decision_tree_smote = auc(fpr_decision_tree_smote, tpr_decision_tree_smote)






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
data = pd.read_csv("schizophrenia_dataset.csv")
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

metrics, (fpr_lr, tpr_lr) = (evaluation_metrics(best_model,y_test, y_pred, X_test))


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
metrics, (fpr_lr_smote, tpr_lr_smote) = (evaluation_metrics(best_model,y_test, y_pred, X_test))











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
df = pd.read_csv("schizophrenia_dataset.csv")



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
# ROC Curve no SMOTE
rf_fpr_ns, rf_tpr_ns, _ = roc_curve(Y_test, y_proba_nosmote)



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
fpr_rf, tpr_rf, _ = roc_curve(Y_test, y_proba_smote)






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


# --- Plot ROC Curves ---
fpr_no_smote_svm, tpr_no_smote_svm, _ = roc_curve(y_test, y_proba_no_smote)
fpr_smote_svm, tpr_smote_svm, _ = roc_curve(y_test, y_proba_smote)


plt.figure(figsize=(6, 5))
plt.plot(fpr_no_smote_svm, tpr_no_smote_svm, label=f"SVM")
plt.plot(fpr_smote_svm, tpr_smote_svm, label=f"SVM + SMOTE")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest + SMOTE")
plt.plot(rf_fpr_ns, rf_tpr_ns, label=f"Random Forest")
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression")
plt.plot(fpr_lr_smote, tpr_lr_smote, label=f"Logistic Regression + SMOTE")
plt.plot(fpr_decision_tree_smote, tpr_decision_tree_smote, label=f'Decision Tree + SMOTE')
plt.plot(fpr_decision_tree, tpr_decision_tree, label=f'Decision Tree')
plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.savefig("ROC.png", dpi=200)
plt.show()




























