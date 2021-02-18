# -*- coding: utf-8 -*-
import os
import sys
import codecs
import glob
import codecs
import re
import _pickle as cPickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from module import Stemmer
from warnings import simplefilter

# ignore all warnings
simplefilter(action='ignore')

script_dir=os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
picklePath=os.path.join(script_dir,'model')
corpusPath=os.path.join(script_dir,'data','domain','*.txt')
listOfCorpusFiles=glob.glob(corpusPath)

print("Processing Corpus files:",*listOfCorpusFiles,sep='\n')
corpus=[]
faq=[]

for fileName in listOfCorpusFiles:
	FI=codecs.open(fileName,'r',encoding='utf-8')
	corpus.append(FI.read())

corpusItems=len(corpus)

faqPath=os.path.join(script_dir,'data','faq','*.txt')
listOfFaqFiles=glob.glob(faqPath)

print("\nProcessing FAQ files:",*listOfFaqFiles,sep='\n')

for fileName in listOfFaqFiles:
	FI=codecs.open(fileName,'r',encoding='utf-8').read()
	i=1
	for line in FI.split('\n'):
			if(line.count('?')>1):
				print("SEVERE:Found multiple questions in %s at line %d."%(os.path.basename(fileName),i))
				print("SEVERE:Aborting the process..!!!")
				sys.exit("Aborting...")
			line=line.replace('$','USD')
			line=line.replace('"','\'')
			que,ans=line.split('?')
			corpus.append(que+' ?')
			faq.append(line)
			i+=1

print('\nTotal no of questions for training: %s'%len(corpus))

stopListFile=os.path.join(script_dir,'data','dictionary','stopwords.txt')
stopWords=[line.rstrip('\n')for line in open(stopListFile,'r+')]
extraStopWords=set(stopWords)
stops=set(stopwords.words('english'))|extraStopWords
tfidfvec=TfidfVectorizer(corpus,decode_error='ignore',stop_words=stops,ngram_range=(1,5),tokenizer=Stemmer.stemTokenize)
transientIdfVectorizer=tfidfvec.fit_transform(corpus).toarray()
vLength=len(transientIdfVectorizer[1])

nDimension=200 

if vLength<=200:
	nDimension=vLength-1
svd=TruncatedSVD(n_components=nDimension,algorithm='randomized',n_iter=15,random_state=42)
trainLSA=svd.fit_transform(transientIdfVectorizer)
fileName=os.path.join(picklePath,'corpus.m')
fileObject=open(fileName,'wb')
cPickle.dump(corpus,fileObject) 
fileObject.close()
fileName=os.path.join(picklePath,'faq.m')
fileObject=open(fileName,'wb')
cPickle.dump(faq,fileObject) 
fileObject.close()
fileName=os.path.join(picklePath,'svd.m')
fileObject=open(fileName,'wb')
cPickle.dump(svd,fileObject) 
fileObject.close()
fileName=os.path.join(picklePath,'tfidfvec.m')
fileObject=open(fileName,'wb')
cPickle.dump(tfidfvec,fileObject) 
fileObject.close()
fileName=os.path.join(picklePath,'trainLSA.m')
fileObject=open(fileName,'wb')
cPickle.dump(trainLSA,fileObject) 
fileObject.close()
