# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 17:26:36 2018

@author: mohsin.aslam
"""

import urllib.request as urllib2
# If you are using Python 3+, import urllib instead of urllib2

import json 
import pandas as pd 
import numpy as np

def Read_file(pathname = "", filename = ""):
    import os
    filepath = os.path.join(pathname,filename)
    print (filepath)
    frame1 = pd.read_csv(filepath)
    return frame1
#testdata = Read_file('F:\\Downloads\\','Student Performance.csv')
#pp = testdata.values
#a = list(list(pp))

data =  {

        "Inputs": {

                "input1":
                {
                    "ColumnNames": ["gender", "NationalITy", "PlaceofBirth", "StageID", "GradeID", "SectionID", "Topic", "Semester", "Relation", "raisedhands", "VisITedResources", "AnnouncementsView", "Discussion", "ParentAnsweringSurvey", "ParentschoolSatisfaction", "StudentAbsenceDays", "Class"],
                    "Values": [['F', 'KW', 'KuwaIT', 'HighSchool', 'G-12', 'A', 'English', 'F',
        'Father', 55, 90, 16, 40, 'No', 'Bad', 'Under-7', 'M'],
       ['M', 'Jordan', 'Jordan', 'MiddleSchool', 'G-08', 'A', 'Chemistry',
        'S', 'Mum', 70, 82, 75, 29, 'Yes', 'Good', 'Under-7', 'M'],
       ['F', 'KW', 'KuwaIT', 'lowerlevel', 'G-04', 'A', 'IT', 'F',
        'Father', 42, 30, 13, 70, 'Yes', 'Bad', 'Above-7', 'M'],
       ['M', 'Jordan', 'Jordan', 'lowerlevel', 'G-02', 'B', 'French', 'S',
        'Father', 50, 62, 13, 33, 'No', 'Bad', 'Above-7', 'L'],
       ['F', 'KW', 'KuwaIT', 'lowerlevel', 'G-02', 'C', 'IT', 'F',
        'Father', 10, 3, 0, 30, 'No', 'Bad', 'Under-7', 'M'],
       ['M', 'Syria', 'Syria', 'lowerlevel', 'G-02', 'B', 'French', 'S',
        'Mum', 20, 52, 23, 33, 'Yes', 'Good', 'Above-7', 'L'],
       ['M', 'KW', 'KuwaIT', 'MiddleSchool', 'G-08', 'B', 'Arabic', 'S',
        'Mum', 15, 90, 52, 83, 'Yes', 'Bad', 'Under-7', 'H'],
       ['M', 'Jordan', 'Jordan', 'lowerlevel', 'G-02', 'A', 'Arabic', 'F',
        'Father', 10, 17, 50, 21, 'No', 'Bad', 'Under-7', 'M'],
       ['F', 'Jordan', 'KuwaIT', 'MiddleSchool', 'G-08', 'C', 'Spanish',
        'S', 'Father', 87, 88, 40, 10, 'Yes', 'Good', 'Under-7', 'M'],
       ['M', 'Iraq', 'Iraq', 'lowerlevel', 'G-02', 'B', 'Arabic', 'F',
        'Mum', 69, 82, 20, 28, 'Yes', 'Good', 'Under-7', 'H'],
       ['F', 'SaudiArabia', 'SaudiArabia', 'lowerlevel', 'G-02', 'B',
        'French', 'S', 'Father', 60, 70, 63, 93, 'Yes', 'Bad', 'Under-7',
        'H'],
       ['M', 'Jordan', 'SaudiArabia', 'lowerlevel', 'G-02', 'B', 'Arabic',
        'S', 'Mum', 5, 0, 1, 8, 'No', 'Good', 'Above-7', 'L'],
       ['F', 'USA', 'USA', 'HighSchool', 'G-12', 'A', 'English', 'F',
        'Mum', 65, 75, 23, 80, 'Yes', 'Good', 'Under-7', 'H'],
       ['M', 'KW', 'KuwaIT', 'lowerlevel', 'G-04', 'A', 'IT', 'F',
        'Father', 40, 50, 12, 50, 'No', 'Bad', 'Above-7', 'M'],
       ['F', 'Jordan', 'Jordan', 'MiddleSchool', 'G-08', 'A', 'History',
        'S', 'Father', 39, 88, 43, 72, 'Yes', 'Good', 'Under-7', 'M'],
       ['F', 'Iraq', 'Iraq', 'lowerlevel', 'G-02', 'B', 'Arabic', 'S',
        'Mum', 79, 93, 49, 23, 'Yes', 'Good', 'Under-7', 'H'],
       ['M', 'KW', 'KuwaIT', 'MiddleSchool', 'G-08', 'C', 'Spanish', 'S',
        'Mum', 57, 51, 46, 34, 'Yes', 'Good', 'Under-7', 'M'],
       ['M', 'Syria', 'Syria', 'lowerlevel', 'G-02', 'A', 'French', 'F',
        'Father', 24, 35, 18, 31, 'No', 'Bad', 'Under-7', 'M'],
       ['F', 'KW', 'KuwaIT', 'lowerlevel', 'G-02', 'B', 'IT', 'F',
        'Father', 66, 90, 55, 12, 'Yes', 'Good', 'Above-7', 'M'],
       ['M', 'Egypt', 'Egypt', 'lowerlevel', 'G-04', 'A', 'Math', 'S',
        'Mum', 49, 94, 42, 7, 'No', 'Bad', 'Above-7', 'M'],
       ['F', 'Jordan', 'Jordan', 'lowerlevel', 'G-04', 'B', 'Science', 'S',
        'Mum', 12, 44, 25, 39, 'No', 'Bad', 'Under-7', 'M'],
       ['M', 'Jordan', 'Jordan', 'MiddleSchool', 'G-07', 'A', 'Biology',
        'F', 'Father', 39, 71, 40, 26, 'No', 'Good', 'Above-7', 'M'],
       ['M', 'Jordan', 'SaudiArabia', 'MiddleSchool', 'G-08', 'A',
        'Geology', 'S', 'Father', 80, 89, 23, 68, 'No', 'Bad', 'Under-7',
        'H'],
       ['F', 'Syria', 'Syria', 'MiddleSchool', 'G-07', 'A', 'Biology', 'S',
        'Mum', 80, 91, 87, 72, 'Yes', 'Good', 'Under-7', 'H']]
                },        },
            "GlobalParameters": {
}
    }

body = str.encode(json.dumps(data))
#url = 'https://ussouthcentral.services.azureml.net/workspaces/fb8f085099c94b9d815dda244f2eb1a9/services/dcf67b8be850446bbd0c2f22cfd34202/execute?api-version=2.0&details=true'   # My URL
#api_key = 'D4EbvrMYRirszvYv8HcJBU7dGUDCd9c+h+XZ/t+3IB/ERvbtjOfbfDj+oTqnVSQDU960raOnah9sts5bd/Vk6Q==' # My API

api_key = 'E+d3f1hiYyCV4z49jZ6pCeuD2CC66+k2m7/0DVJkBa+Ou1tvdFlj87qfCDE596qkZVqYVqNBT9hr7ytZp49Q+A=='
url = 'https://ussouthcentral.services.azureml.net/workspaces/89a890388444492faa6e5f2a9b83d3b2/services/8beff44062ac4853af81b5fc00f66cd5/execute?api-version=2.0&details=true'
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib2.Request(url, body, headers) 

try:
    response = urllib2.urlopen(req)

    # If you are using Python 3+, replace urllib2 with urllib.request in the above code:
    # req = urllib.request.Request(url, body, headers) 
    # response = urllib.request.urlopen(req)

    result = response.read()
    print(result) 
except (urllib2.HTTPError):
    '''
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())

    print(json.loads(error.read()))    
    '''
    a = urllib2.HTTPError
    print ('ss')             
count = 0


import json
my_json = result.decode('utf8').replace("'", '"')
data1 = json.loads(my_json)
for i in range(0 , len(data1['Results']['output1']['value']['Values']) ):
    clabel = data1['Results']['output1']['value']['Values'][i][0]
    
    slabel = data1['Results']['output1']['value']['Values'][i][len(data1['Results']['output1']['value']['Values'][0]) - 1]
    if clabel == slabel:
        count = count +1
accuracy = float(count)/len(data1['Results']['output1']['value']['Values'])
print ('Accuracy = ',accuracy)


