#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:22:59 2017

@author: ahadmushir
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 00:45:15 2017

@author: ahadmushir
"""

from flask import Flask, jsonify
#from flask_restful import Resource, Api
from flask import request
from flask import json
#import getCommon
#import setCommon
import pandas as pd
from sklearn.externals import joblib


app = Flask(__name__)
#api = Api(app)

@app.route('/predict', methods = ['POST'])
def predicting():
    json_ = request.json
    print ( json.dumps(request.json))

    query_df = pd.DataFrame(json_)
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'other', 'purpose'] = 0
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'Other', 'purpose'] = 0
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'major_purchase', 'purpose'] = 0
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'moving', 'purpose'] = 1
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'vacation', 'purpose'] = 1
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'Take a Trip', 'purpose'] = 1
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'Business Loan', 'purpose'] = 2
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'small_business', 'purpose'] = 2
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'Home Improvements', 'purpose'] = 3
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'Buy a Car', 'purpose'] = 3
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'Buy House', 'purpose'] = 3
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'Debt Consolidation', 'purpose'] = 4
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'Educational Expenses', 'purpose'] = 5
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'wedding', 'purpose'] = 5
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'Medical Bills', 'purpose'] = 5
    query_df['purpose_bin'] = query_df.loc[query_df.purpose == 'renewable_energy', 'purpose'] = 5
    query_df['purpose_bin'] = query_df['purpose'].convert_objects(convert_numeric=True)
    query_df['purpose'] = query_df['purpose_bin']

    terms_df = pd.get_dummies(query_df['term'])
    ownership_df = pd.get_dummies(query_df['homeownership'])
    final_df = pd.concat([query_df, terms_df], axis=1)
    final_df = pd.concat([query_df, ownership_df], axis=1)

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    temp = final_df.select_dtypes(include=numerics)
    l1 = list(temp.columns)
    l1.remove('purpose_bin')

    print (query_df)
    prediction = clf.predict(final_df[l1])

#    if request.headers['Content-Type'] == 'application/json':
#        setCommon.PostUserInfo(request.json)

    return jsonify({'prediction': list(prediction)})

@app.route('/alive', methods = ['GET'])
def alive():
    return jsonify({'prediction': 'About to come'})


if __name__ == '__main__':
    clf = joblib.load('modelFinal.pkl')
    model_columns = joblib.load('modelFinal.pkl')
    app.run(host='0.0.0.0')
