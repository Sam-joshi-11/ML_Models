import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import(
    accuracy_score,
    classification_report
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
