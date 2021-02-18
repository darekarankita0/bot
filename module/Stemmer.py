# -*- coding: utf-8 -*-
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer

class PorterStemming(object):
	def __init__(self):
			self.wnl=PorterStemmer()
	def __call__(self,doc):
			return[self.wnl.stem(t)for t in word_tokenize(doc)]
stemmer=SnowballStemmer('english').stem

def stemTokenize(text):
	return[stemmer(i)for i in word_tokenize(text)]