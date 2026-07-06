import re ,string,nltk,spacy
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
nltk.download("punkt_tab")
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("IMDB Dataset.csv")

stop_words = stopwords.words("english")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>',' ',text)
    text = re.sub(r'http\S+|www\S+',' ',text)
    text = re.sub(r'[^a-z\s]',' ',text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    doc = nlp(' '.join(tokens))
    lemmas = [tok.lemma_ for tok in doc]
    return ' '.join(lemmas)

df['clean_review'] = df['review'].apply(preprocess)
df[['review','clean_review']].head()

df['label'] = df['sentiment'].map({'negative':0,'positive':1})
X_train,X_test,Y_train,Y_test = train_test_split(
    df['clean_review'],df['label'],test_size=0.2,
    random_state=42,stratify=df['label']
)

tfidf = TfidfVectorizer(max_features=5000)
X_train_t = tfidf.fit_transform(X_train)
X_test_t = tfidf.transform(X_test)
print(X_train_t.shape)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_t,Y_train)
pred_lr = lr.predict(X_test_t)
print("LR Accuracy:",accuracy_score(Y_test,pred_lr))
print("Confusion Matrix:",confusion_matrix(Y_test,pred_lr))
print("Classification Repoort:",classification_report(Y_test,pred_lr))


xgb = XGBClassifier(n_estimators = 100,max_depth = 6, learnin_rate=0.1,
                    eval_metric='logloss',random_state=42)
xgb.fit(X_train_t,Y_train)
pred_xgb = xgb.predict(X_test_t)
print('XGBoost Accuracy:',accuracy_score(Y_test,pred_xgb))
print("confusion matrix:",confusion_matrix(Y_test,pred_xgb))
print("Classification Report:",classification_report(Y_test,pred_xgb))

results = pd.DataFrame({
    'Model':['Logistic Regression','XGBoost'],
    "Accuracy":[accuracy_score(Y_test,pred_lr),
                accuracy_score(Y_test,pred_xgb)]
})
print(results)

def predict_review(review,model):
    clean=preprocess(review)
    vec=tfidf.transform([clean])
    pred=model.predict(vec)[0]
    return 'Positive' if pred==1 else 'Negative'

print(predict_review('This movie was absolutely fantastic!',lr))
print(predict_review('Worst movie I have ever watched.',xgb))