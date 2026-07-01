import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import(
    accuracy_score,
    classification_report
)

data = load_breast_cancer(as_frame=True)
df = data.frame

print(df.head())

X = df.drop("target",axis=1)
y = df['target']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)
rf_predict = rf.predict(X_test)

print("Accuracy Score:",accuracy_score(y_test,rf_predict))
print("Classification_report:\n",classification_report(y_test,rf_predict))

xgb = XGBClassifier()
xgb.fit(X_train,y_train)
xgb_predict = xgb.predict(X_test)

print("Accuracy Score:",accuracy_score(y_test,xgb_predict))
print("classification_report:\n",classification_report(y_test,xgb_predict))

lgb = LGBMClassifier()
lgb.fit(X_train,y_train)
lgb_predict = lgb.predict(X_test)

print("Accuracy Score:",accuracy_score(y_test,lgb_predict))
print("Classification_report:\n",classification_report(y_test,lgb_predict))


comparison = pd.DataFrame({
    "model":[
        "Random forest",
        "XGboost",
        "LightGBM"
    ],
    "Accuracy":[
        accuracy_score(y_test,rf_predict),
        accuracy_score(y_test,xgb_predict),
        accuracy_score(y_test,lgb_predict)
    ]
})

print(comparison)