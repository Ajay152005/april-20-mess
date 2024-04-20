from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# load the 20 Newsgroups dataset

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))

#Preprocess the text data

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

X = vectorizer.fit_transform(newsgroups_train.data)

y = newsgroups_train.target

#Split the data into trainig and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)

# Initialize the Naive BAYES classifier
classifier = MultinomialNB()

#Train the classifier
classifier.fit(X_test, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

#Evaluate the model
print(classification_report(y_test, y_pred, target_names = newsgroups_train.target_names))