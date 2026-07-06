import re
text='NLP,, Session is.. going on!!!'
clean = re.sub(r'[^a-zA-Z]',' ',text)
print(clean)

text1 = 'The Climate is beautiful'
tokens = text1.split()
print(tokens)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab")
nltk.download("stopwords")
tokens = word_tokenize(text1)
filtered = [word for word in tokens if word.lower() not in stopwords.words("english")]
print(filtered)

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
words = ['played','reading']
for word in words :
    print(stemmer.stem(word))


from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running",pos='v'))

from sklearn.feature_extraction.text import CountVectorizer
text2 = ['The Climate is beautiful','It is raining']
vector = CountVectorizer()
x = vector.fit_transform(text2)
print(x.toarray())
print(vector.get_feature_names_out())


from gensim.models import Word2Vec
sentences = ['climate', 'beautiful'],['raning','climate']
model = Word2Vec(sentences,vector_size = 20,window = 2,min_count = 1)
print(model.wv["climate"])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
text3 = ["i love this phone","I hate the phone's battery"]
labels=[1,0]
vectorizer = TfidfVectorizer()
x=vectorizer.fit_transform(text3)
model = LogisticRegression()
model.fit(x,labels)
new_text = ["i hate this phone"]
new_x = vectorizer.transform(new_text)
predictions = model.predict(new_x)
print(predictions)