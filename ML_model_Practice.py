import numpy as np
import pandas as pd
import os
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    precision_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc
)



file_path = 'Churn_Modelling.csv'
if not os.path.exists(file_path):
    print(f"Error : {file_path} is not found")

print("Loading Dataset...\n")
df = pd.read_csv(file_path)
print("Loaded Successfully\n")

print(f"Printing First 5 rows :\n{df.head()}")
print(f"\nPrinting Last 5 rows :\n{df.tail()}")
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

le = LabelEncoder()

df["Geography"] = le.fit_transform(df["Geography"])
df["Gender"] = le.fit_transform(df["Gender"])
print(df.info())

X = df.drop("Exited",axis=1)
y = df['Exited']

print(df.describe())
print(df.info())
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)

print("Accuracy Score:",accuracy_score(y_test,rf_pred))
print("Classification_report:\n",classification_report(y_test,rf_pred))

xgb = XGBClassifier()
xgb.fit(X_train,y_train)
xgb_pred = xgb.predict(X_test)

print("Accuracy Score:",accuracy_score(y_test,xgb_pred))
print("Classification_report:\n",classification_report(y_test,xgb_pred))

lgb = LGBMClassifier()
lgb.fit(X_train,y_train)
lgb_pred = lgb.predict(X_test)

print("Accuracy Score:",accuracy_score(y_test,lgb_pred))
print("Classification_report:\n",classification_report(y_test,lgb_pred))
#-------------------------------------------------
precision = precision_score(y_test,rf_pred)
recall = recall_score(y_test,rf_pred)
f1s = f1_score(y_test,rf_pred)

print(f"precison for Random Forest :{precision}")
print(f"recall for Random Forest :{recall}")
print(f"f1-score for Random Forest :{f1s}\n")
#------------------------------------------------
precision1 = precision_score(y_test,xgb_pred)
recall1 = recall_score(y_test,xgb_pred)
f1s1 = f1_score(y_test,xgb_pred)

print(f"precison for XGBoost :{precision1}")
print(f"recall for XGBoost :{recall1}")
print(f"f1-score for XGBoost :{f1s1}\n")
#----------------------------------------------
precision2 = precision_score(y_test,lgb_pred)
recall2 = recall_score(y_test,lgb_pred)
f1s2 = f1_score(y_test,lgb_pred)

print(f"precison for Lightgbm :{precision2}")
print(f"recall for Lightgbm :{recall2}")
print(f"f1-score for Lightgbm :{f1s2}\n")
#-----------------------------------------------
cm = confusion_matrix(
    y_test,
    rf_pred
)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='YlOrRd')
plt.title("Confusion Matrix for Random Forest")
plt.show()
#--------------------------------------------------
cm1 = confusion_matrix(
    y_test,
    xgb_pred
)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1)
disp1.plot(cmap='YlOrRd')
plt.title("Confusion Matrix for XGBoost")
plt.show()
#----------------------------------------------------
cm2 = confusion_matrix(
    y_test,
    lgb_pred
)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp.plot(cmap='YlOrRd')
plt.title("Confusion Matrix for Lightgbm")
plt.show()
#----------------------------------------------------
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]
roc_score = roc_auc_score(
    y_test,
    y_prob
)
print(f"ROC AUC Score for Random Forest :{roc_score}")
#----------------------------------------------------
y_pred1 = xgb.predict(X_test)
y_prob1 = xgb.predict_proba(X_test)[:,1]
roc_score1 = roc_auc_score(
    y_test,
    y_prob1
)
print(f"ROC AUC Score for XGBoost :{roc_score1}")
#----------------------------------------------------
y_pred2 = lgb.predict(X_test)
y_prob2 = lgb.predict_proba(X_test)[:,1]
roc_score2 = roc_auc_score(
    y_test,
    y_prob2
)
print(f"ROC AUC Score for Lightgbm :{roc_score2}")
#----------------------------------------------------
fpr , tpr, thresholds = roc_curve(
    y_test,
    y_prob
)
plt.figure(figsize=(7,4))
plt.plot(fpr,tpr,label=f"AUC = {roc_score:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Random Forest")
plt.legend()
plt.show()
#--------------------------------------------------------
fpr1 , tpr1, thresholds1 = roc_curve(
    y_test,
    y_prob1
)
plt.figure(figsize=(7,4))
plt.plot(fpr1,tpr1,label=f"AUC = {roc_score1:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for XGBoost")
plt.legend()
plt.show()
#--------------------------------------------------------
fpr2 , tpr2, thresholds2 = roc_curve(
    y_test,
    y_prob2
)
plt.figure(figsize=(7,4))
plt.plot(fpr2,tpr2,label=f"AUC = {roc_score2:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Lightgbm")
plt.legend()
plt.show()
#--------------------------------------------------------
comparison = pd.DataFrame({
    "mopdel":[
        "Random forest:",
        "Xgboost :",
        "lightgm :"
    ],
    "Accuracy":[
         accuracy_score(y_test,rf_pred),
        accuracy_score(y_test,xgb_pred),
        accuracy_score(y_test,lgb_pred)
    ]
})

print(comparison)

important = pd.Series(
    rf.feature_importances_,
    index=X.columns
)
plt.figure(figsize=(7,4))
important.sort_values(
    ascending=False
).head(10).plot.barh()
plt.title("Important Features")
plt.xlabel("Important")
plt.ylabel("Features")
plt.show()

#1. Baseline Model
model = XGBClassifier(random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Baseline Accuracy:",accuracy_score(y_test,y_pred))

#2. Grid Search (simple)
param_grid = {
    'n_estimators':[50,100,15,],
    'max_depth':[3,5,7],
    'learning_rate':[0.01,0.1]
}
grid = GridSearchCV(
    estimator = XGBClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy'
)
grid.fit(X_train,y_train)
print("Best Parameters")
print(grid.best_params_)
best_model = grid.best_estimator_
prediction = best_model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,prediction))