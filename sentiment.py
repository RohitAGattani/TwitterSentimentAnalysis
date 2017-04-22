import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt

stopWords = stopwords.words('english')

# read csv data file and keep only required columns
data = pd.read_csv('input/Sentiment.csv')
data = data[['text', 'sentiment']]

# split the dataset into train and test set
trainData, testData = train_test_split(data, test_size = 0.2)


# ****************************************************************
# preprocessing
regexRules = [(r'(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@[A-Za-z]+[A-Za-z0-9_]+', ''),
			  (r'(?<=^|(?<=[^a-zA-Z0-9-_\\.]))#[A-Za-z]+[A-Za-z0-9_]+', ''),
			  (r'(http|https|ftp)\S+', ''),
			  (r'((\w)\2{2,})', r'\2')]

emoticons = [	
		('__EMOT_SMILEY',	[':-)', ':)', '(:', '(-:', ] )	,
		('__EMOT_LAUGH',		[':-D', ':D', 'X-D', 'XD', 'xD', ] )	,
		('__EMOT_LOVE',		['<3', ':\*', ] )	,
		('__EMOT_WINK',		[';-)', ';)', ';-D', ';D', '(;', '(-;', ] )	,
		('__EMOT_FROWN',		[':-(', ':(', '(:', '(-:', ] )	,
		('__EMOT_CRY',		[':,(', ':\'(', ':"(', ':(('] )	,
	]

#For emoticon regexes
def escape_paren(arr):
	return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

def regex_union(arr):
	return '(' + '|'.join( arr ) + ')'

emoticons_regex = [ (re.compile(regex_union(escape_paren(regx))), repl) for (repl, regx) in emoticons ]

# remove user-mentions, hashtags, links and replace repeated characters (>2)
def preprocessData(data, regexRules):
	tweets = []
	for (index, row) in data.iterrows():
		text = row.text.lower()
		for (pattern, replace) in regexRules:
			text = re.sub(pattern, replace, text)
		for (pattern, replace) in emoticons_regex:
			text = re.sub(pattern, replace, text)
		tweets.append(text)
	return tweets

trainingTweets = preprocessData(trainData, regexRules)
testingTweets = preprocessData(testData, regexRules)


# ****************************************************************
## training a classifier (using a pipeline)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics


# Naive Bayes
print('Naive Bayes')
from sklearn.naive_bayes import MultinomialNB
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
					('tfidf', TfidfTransformer()),
					('clf', MultinomialNB())
			])
_ = text_clf.fit(trainingTweets, trainData['sentiment'])
predicted = text_clf.predict(testingTweets)
#for tweet, sentimentCategory in zip(testData['text'], predicted):
#	print('%s => %s' %(tweet, sentimentCategory))
print('MEAN: ', np.mean(predicted == testData['sentiment']))

print(metrics.classification_report(testData['sentiment'], predicted))
print(metrics.confusion_matrix(testData['sentiment'], predicted))


## parameters tuining using grid search
from sklearn.model_selection import GridSearchCV
parameters = {	
				'vect__ngram_range':[(1, 1), (1, 2)],
				'tfidf__use_idf':(True, False),
				'clf__alpha':(1e-2, 1e-3)
			}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(trainingTweets, trainData['sentiment'])
print('Best score: ', gs_clf.best_score_)

for param_name in sorted(parameters.keys()):
	print("%s: %r" %(param_name, gs_clf.best_params_[param_name]))


# ****************************************************************
# svm
print('svm')
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
					('tfidf', TfidfTransformer()),
					('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=10))
			])
_ = text_clf.fit(trainingTweets, trainData['sentiment'])
predicted = text_clf.predict(testingTweets)
#for tweet, sentimentCategory in zip(testData['text'], predicted):
#	print('%s => %s' %(tweet, sentimentCategory))
print('MEAN: ', np.mean(predicted == testData['sentiment']))

print(metrics.classification_report(testData['sentiment'], predicted))
print(metrics.confusion_matrix(testData['sentiment'], predicted))


## parameters tuining using grid search
from sklearn.model_selection import GridSearchCV
parameters = {	
				'vect__ngram_range':[(1, 1), (1, 2)],
				'tfidf__use_idf':(True, False),
				'clf__alpha':(1e-2, 1e-3)
			}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(trainingTweets, trainData['sentiment'])
print('Best score: ', gs_clf.best_score_)

for param_name in sorted(parameters.keys()):
	print("%s: %r" %(param_name, gs_clf.best_params_[param_name]))


# ****************************************************************
# Multi-layer perceptron
print('MLP')
from sklearn.neural_network import MLPClassifier
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
					('tfidf', TfidfTransformer()),
					('clf', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1))
			])
_ = text_clf.fit(trainingTweets, trainData['sentiment'])
predicted = text_clf.predict(testingTweets)
#for tweet, sentimentCategory in zip(testData['text'], predicted):
#	print('%s => %s' %(tweet, sentimentCategory))
print('MEAN: ', np.mean(predicted == testData['sentiment']))

print(metrics.classification_report(testData['sentiment'], predicted))
print(metrics.confusion_matrix(testData['sentiment'], predicted))
