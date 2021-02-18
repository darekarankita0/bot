# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 02:28:08 2017
"""
import os
import re
import glob
import _pickle as cPickle 
from flask import Flask, jsonify
from flask import request
from flask import abort
from flask import make_response
from module import ProcessQuery as pq
from warnings import simplefilter
# ignore all warnings
simplefilter(action='ignore')

cosValResultLimit=0.4
app = Flask(__name__)
#get absolute dir the script is in
scriptDir = os.path.dirname(__file__)
#get the number of corpus items 
corpusItems = len(glob.glob1(os.path.join(scriptDir, 'data','domain'),'*.txt'))

picklePath = os.path.join(scriptDir,'model')
faq=cPickle.load(open((os.path.join(picklePath, 'faq.m')),'rb'))

@app.route('/faq', methods=['GET'])
def get_query():
	if not (request.args.get('userUtterance')):
		abort(404)
	if (request.args.get('language')):
		lang = request.args.get('language')
	else:
		lang = 'en'
	if (request.args.get('info')):
		info = request.args.get('info')
	else:
		info = ""

	query= request.args.get('userUtterance')
	query = re.sub(r'[^a-zA-Z ]', '', query)
	print("\n\n Query: ", query)
	combinations = pq.gen_utterances(query)
	result = pq.processUtterances(combinations,lang)

	#Check if result exists
	if result:
		#Taking the first best ANS
		bestScoreID = next(iter(result))
		if (result[bestScoreID]['cosineVal']>=cosValResultLimit):
			q, a = faq[int(bestScoreID) - corpusItems].split('?')
			ans = a.strip()
		else:
			ans = "NA"
			pq.logTraining(query)
		#Formatting O/P
		count = 1
		info = {}
		for resultID, scores in result.items():
			info['id_'+str(count)] = str(resultID)
			info['score_'+str(count)] = format(scores['cosineVal'], '.2f')
			count += 1
		del count
		result = jsonify({'response': ans,'info': info})
		return result, 201
	else:
		result = jsonify({'response': 'NA', 'info': 'NA'})
		pq.logTraining(query)
		return result, 201

@app.errorhandler(404)
def not_found(error):
	return make_response(jsonify({'error': 'Please enter a query'}), 404)

if __name__ == '__main__':
	app.run(debug=False)
