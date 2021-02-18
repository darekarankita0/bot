# -*- coding: utf-8 -*-
import os
import glob
import datetime
import time
import _pickle as cPickle
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from sklearn.metrics.pairwise import linear_kernel


cosValProcessLimit=0.1
simDiffProcessLimit=0.9

scriptDir = os.path.dirname(__file__)
picklePath = os.path.join(scriptDir, '..','model')
logFile = os.path.join(scriptDir, '..','log',str(datetime.datetime.today().strftime('%b'))+'_training.txt')
corpusItems = len(glob.glob1(os.path.join(scriptDir, '..','data','domain'),'*.txt'))
#intent = cPickle.load(open((os.path.join(picklePath, 'intent.m')),'rb'))
corpus=cPickle.load(open((os.path.join(picklePath, 'corpus.m')),'rb'))
faq=cPickle.load(open((os.path.join(picklePath, 'faq.m')),'rb'))
tfidfvec=cPickle.load(open((os.path.join(picklePath, 'tfidfvec.m')),'rb'))
svd=cPickle.load(open((os.path.join(picklePath, 'svd.m')),'rb'))
trainLSA=cPickle.load(open((os.path.join(picklePath, 'trainLSA.m')),'rb'))


stemmer = SnowballStemmer('english').stem

def stem_tokenize(text):
	return [stemmer(i) for i in word_tokenize(text)]

def replaceNth(string, sub, repl, nth):
	find = string.find(sub)
	i = find != -1
	while find != -1 and i != nth:
		find = string.find(sub, find + 1)
		i += 1
	if i == nth:
		return string[:find] + repl + string[find + len(sub):]
	return string

def wordReplacer(utterance, matchedDict, combinations):
	matchedDict = matchedDict.copy()
	while(len(matchedDict) > 0):
		replacement = matchedDict.popitem()
		for wordReplacement in replacement[1]['synonym']:
			newUtterance = utterance.replace(replacement[0], wordReplacement)
			combinations.append(newUtterance)
			wordReplacer(newUtterance, matchedDict, combinations)

def genSentences(utterance, matchedDict, combinations):
	matchedDict = matchedDict.copy() 
	while(len(matchedDict) > 0):
		replacement = matchedDict.popitem()
		for count in range(replacement[1]['count']):
			for wordReplacement in replacement[1]['synonym']:
				newUtterance = replaceNth(utterance, replacement[0], wordReplacement, count + 1)
				combinations.append(newUtterance)
				wordReplacer(newUtterance, matchedDict, combinations)   
	
def processUtterances(utterance,lang):
	result = {}
	for query in utterance:
		query = query.lower()
		query = [query]
		test = tfidfvec.transform(query).toarray()
		LSATest = svd.transform(test)
		cosineSimilarity = linear_kernel(LSATest, trainLSA).flatten() 
		relatedDocsIndex = cosineSimilarity.argsort()[:-4:-1]

		for fID in relatedDocsIndex:
			fScore = cosineSimilarity[fID]
			if (fID in result):
				result[fID] = max(fScore, result[fID])
			else:
				result[fID] = fScore
	noOfTopResults = 3
	maxCosineVal = max(result.values())

	result = {key: result[key] for key in result if key in sorted(result, key=result.get, reverse=True)[:noOfTopResults]} 
	#print ('----Result:', result)
	finalDocIndices = {id:{'cosineVal': cosineVal, 'sim_diff': maxCosineVal - cosineVal} for (id, cosineVal) in result.items() if (id >= corpusItems) & (cosineVal > cosValProcessLimit) & (maxCosineVal - cosineVal < simDiffProcessLimit)}
	#print ('----finalDoc:', finalDocIndices)
	#finalDocIndices[id]['cosineVal'] = Cosine Score   finalDocIndices[id]['sim_diff'] = Similarity Difference Score
	return finalDocIndices
	
def logTraining(logDetails):
    #check if lofFile exists
	if os.path.isfile(logFile):
		dFile = time.strftime("%m/%d/%Y",time.localtime(os.path.getmtime(logFile)))
		dCurrent = datetime.datetime.today().strftime('%m/%d/%Y')
		dateFile = datetime.datetime.strptime(dFile, '%m/%d/%Y')
		dateCurrent = datetime.datetime.strptime(dCurrent, '%m/%d/%Y')
		diff = (dateCurrent - dateFile).days
		if (diff>31):
			f1 = open(logFile, 'w')
			f1.write(str(datetime.datetime.now(datetime.timezone.utc).astimezone())+" "+logDetails+"\n")
			f1.close()
		else:
			f1 = open(logFile, 'a')
			f1.write(str(datetime.datetime.now(datetime.timezone.utc).astimezone())+" "+logDetails+"\n")
			f1.close()
	else:
		f1 = open(logFile, 'w')
		f1.write(str(datetime.datetime.now(datetime.timezone.utc).astimezone())+" "+logDetails+"\n")
		f1.close()

synonymFile = os.path.join(scriptDir,'..','data','dictionary','synonyms.txt')
synonymsList = []
with open(synonymFile,'r') as rawSynonymFileObj:
	rawSynonyms = rawSynonymFileObj.read()
	rawSynonyms = rawSynonyms.split('\n')

for i in rawSynonyms:
	synonymsList.append(i.split(','))
	
def gen_utterances(utterance):
	matched = {}
	utteranceSet = set(utterance.split())
	for synonym in synonymsList:
		for word in set(synonym) & utteranceSet:
			count = utterance.split().count(word)
			matched[word] = {'synonym':list(set(synonym) - set([word])), 'count':count}
	combinations = [utterance]
	genSentences(utterance,matched, combinations)
	combinations.sort()
	return combinations
