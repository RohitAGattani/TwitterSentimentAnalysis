#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics

# display training and test data size
def sizeMB(docs):
	return sum(len(s.encode('utf-8')) for s in docs) / 1e6

# display categorical details of dataset 
def printGroupByDetails(data, groupByColumn):
	groupedData = data.groupby(by=groupByColumn)
	for name, group in groupedData:
		print(" %s => %d " %(name, len(group)))

# read csv data file and keep only required columns
print('[INFO]: loading data...')
data = pd.read_csv('input/US_Airline.csv')
data = data[['text', 'sentiment']]
print('[INFO]: data loaded')
print()

print('[INFO]: (full set) %d tweets - %0.3fMB' %(len(data['text']), sizeMB(data['text'])))
print()

# display details of full dataset
print('[INFO]: full set details')
printGroupByDetails(data, 'sentiment')
print()

# split the dataset into a training set and a test set
trainingData, testData = train_test_split(data, test_size = 0.2, random_state = 42)

# store tweets and labels in separate variables
trainingTweets = trainingData['text']
testTweets = testData['text']
trainingLabels = trainingData['sentiment']
testLabels = testData['sentiment']

# display details of training set and test set
print('[INFO]: (training set) %d tweets - %0.3fMB' %(len(trainingTweets), sizeMB(trainingTweets)))
print('[INFO]:     (test set) %d tweets - %0.3fMB' %(len(testTweets), sizeMB(testTweets)))
print()

print('[INFO]: training set details')
printGroupByDetails(trainingData, 'sentiment')
print()

# preprocessing
tokenRegexRules = [
			  # twitter tokens
			  (r'(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@[A-Za-z]+[A-Za-z0-9_]+', 'HASHTAG'),
			  (r'(?<=^|(?<=[^a-zA-Z0-9-_\\.]))#[A-Za-z]+[A-Za-z0-9_]+', 'USERNAME'),
			  (r'(http|https|ftp)\S+', 'URL'),
			  (r'(rt)', ''),
			  # acronyms, slang
			  (r'(ab|abt)', 'about'),
			  (r'(ama)', 'ask me anything'),
			  (r'(ab|abt)', 'about'),
			  (r'(b4)', 'before'),
			  (r'(bfn)', 'bye for now'),
			  (r'(btw)', 'by the way'),
			  (r'(bgd)', 'background'),
			  (r'(chk)', 'check'),
			  (r'(cr8|cre8)', 'create'),
			  (r'(cld)', 'could'),
			  (r'(clk)', 'click'),
			  (r'(dae)', 'does anyone else'),
			  (r'(dm)', 'direct message'),
			  (r'(f2f)', 'face to face'),
			  (r'(fab)', 'fabulous'),
			  (r'(fomo)', 'fear of missing out'),
			  (r'(ftl)', 'for the loss'),
			  (r'(ftw)', 'for the win'),
			  (r'(ftfy)', 'fixed that for you'),
			  (r'(hifw)', 'how i felt when'),
			  (r'(ic)', 'i see'),
			  (r'(icymi)', 'in case you missed it'),
			  (r'(idk)', "i don't know"),
			  (r'(imo)', 'in my opinion'),
			  (r'(imho)', 'in my humble opinion'),
			  (r'(irl)', 'in real life'),
			  (r'(jsyk)', 'just so you know'),
			  (r'(lol)', 'laughing out loud'),
			  (r'(mfw)', 'my face when'),
			  (r'(mrw)', 'my reaction when'),
			  (r'(mirl)', 'me in real life'),
			  (r'(mtf)', 'more to follow'),
			  (r'(nsfw)', 'not safe for work'),
			  (r'(nsfl)', 'not safe for life'),
			  (r'(nts)', 'note to self'),
			  (r'(paw)', 'parents are watching'),
			  (r'(prt)', 'please retweet'),
			  (r'(qft)', 'quote for truth'),
			  (r'(smh)', 'shaking my head'),
			  (r'(til)', 'today i learned'),
			  (r'(tbh)', 'to be honest'),
			  (r'(tmb)', 'tweet me back'),
			  (r'(u)', 'you'),
			  (r'(woz)', 'was'),
			  (r'(wtv)', 'whatever'),
			  (r'(ymmv)', 'your mileage may vary'),
			  (r'(yolo)', 'you only live once'),
			  # repeated letters
			  (r'((\w)\2{2,})', r'\2')]

# emoticons
emoticonsDictionary = [	
		([':-)', ':)', '(:', '(-:',], 'EMOT_SMILEY'),
		([':-D', ':D', 'X-D', 'XD', 'xD',], 'EMOT_LAUGH'),
		(['<3', ':\*', ], 'EMOT_LOVE'),
		([';-)', ';)', ';-D', ';D', '(;', '(-;',], 'EMOT_WINK'),
		([':-(', ':(', '(:', '(-:',], 'EMOT_FROWN'),
		([':,(', ':\'(', ':"(', ':((',], 'EMOT_CRY')]

# for emoticon regexes
def buildEmoticonRegex(emoticons):
	emoticons = [emoticon.replace(')', '[)}\]]').replace('(', '[({\[]') for emoticon in emoticons]
	emoticons = '(' + '|'.join(emoticons) + ')'
	return re.compile(emoticons)	

emoticonsRegexRules = [ (buildEmoticonRegex(emoticons), replace) for (emoticons, replace) in emoticonsDictionary]

# remove user-mentions, hashtags, links and replace repeated characters, emoticons
def preprocessData(data):
	tweets = []
	for (index, row) in data.iterrows():
		text = row.text.lower()
		for (pattern, replace) in tokenRegexRules:
			text = re.sub(pattern, replace, text)
		for (pattern, replace) in emoticonsRegexRules:
			text = re.sub(pattern, replace, text)
		tweets.append(text)
	return tweets

processedTrainingTweets = preprocessData(trainingData)
processedTestTweets = preprocessData(testData)

# feature extraction
print('[INFO]: extracting features from the training data...')
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,1), sublinear_tf=True, max_df=0.5)
trainingDtm = vectorizer.fit_transform(processedTrainingTweets)

print('[INFO]: extracting features from the test data...')
testDtm = vectorizer.transform(processedTestTweets)


# compare and contrast classifiers
def benchmark(classifier, classifierName):
	# train the classifier and predict class-labels
	print('[INFO]: Training the classifier on the training set')
	startTime = time()
	classifier.fit(trainingDtm, trainingLabels)
	trainTime = time() - startTime
	print('[INFO]: Training took %0.3fs' %(trainTime))

	print('[INFO]: Predicting the outcome of the test set')
	startTime = time()
	predictedLabels = classifier.predict(testDtm)
	testTime = time() - startTime
	print('[INFO]: Prediction took %0.3fs' %(testTime))

	# calculate accuracy score, display report and build confusion matrix
	score = metrics.accuracy_score(testLabels, predictedLabels)
	print('[INFO]: Accuracy: %0.3f' % score)

	print('[INFO]: Classification report:')
	print(metrics.classification_report(testLabels, predictedLabels))

	confusionMatrix = metrics.confusion_matrix(testLabels, predictedLabels)
	print('[INFO]: Confusion matrix:')
	print(confusionMatrix)

	# return benchmark parameters
	return classifierName, score, trainTime, testTime

# selected classifiers for testing
classifiers = [
	        (KNeighborsClassifier(n_neighbors=10), "K Nearest Neighbors"),
	        (MultinomialNB(alpha=1e-2), "Multinomial Naive Bayes"),
	        (LinearSVC(penalty='l2', dual=False, tol=1e-4), "Linear SVC"),
	        (SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=50), "SGD Classifier-l2"),
	        (SGDClassifier(loss='hinge', penalty='elasticnet', alpha=1e-3, n_iter=50), "SGD Classifier-elasticnet"),
	        (MLPClassifier(alpha=1e-2), "Multi-layer Perceptron")]

# benchmark classifiers and store results
results = []
for classifier, classifierName in classifiers:
	print('=' * 80)
	print(classifierName)
	results.append(benchmark(classifier, classifierName))

# make some plots
# plot code referenced from scikit-learn tutorials
indices = np.arange(len(results))
results = [[result[idx] for result in results] for idx in range(4)]
classifierNames, score, trainingTime, testTime = results

trainingTime = np.array(trainingTime) / np.max(trainingTime)
testTime = np.array(testTime) / np.max(testTime)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, trainingTime, .2, label="training time", color='c')
plt.barh(indices + .6, testTime, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for idx, classifierName in zip(indices, classifierNames):
    plt.text(-.3, idx, classifierName)

plt.show()
