##Exploring the RDD syntax and data frames

from __future__ import print_function

import sys
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import HiveContext
from pyspark.sql.functions import when
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import numpy as np
import pandas as pd

# import numpy as np


sc = SparkContext(appName="testingSql")
sqlContext = HiveContext(sc)

log4j = sc._jvm.org.apache.log4j
log4j.LogManager.getRootLogger().setLevel(log4j.Level.ERROR)
# Create a dataframe from a Hive query
df = sqlContext.sql("""SELECT * FROM loangrant""")
pdf = df.toPandas()
print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)

pdf.drop_duplicates(subset=['loanid'], keep='first')

print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)

pdf.loc[pdf['creditscore'] >= 1000, 'creditscore'] = pdf['creditscore'] / 10
print('*********************************///////**************************************')
'''
print(pdf.bankruptcies.unique())
pdf.loc[pdf['bankruptcies'] == None, 'bankruptcies'] = pdf['bankruptcies'].mean()
print(pdf.bankruptcies.unique())
'''
print('*********************************///////**************************************')
print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)

print('***********************************************************************')
print(pdf.bankruptcies.unique())
pdf['bankruptcies'] = pdf['bankruptcies'].fillna(pdf['bankruptcies'].mean())
print(pdf.bankruptcies.unique())
print('***********************************************************************')

print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)

pdf.creditscore = pdf.creditscore.fillna(pdf.creditscore.median())
pdf.taxliens = pdf.taxliens.fillna(pdf.taxliens.median())

print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)

pdf.annualincome = pdf.annualincome.fillna(pdf.annualincome.median())
print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)

pdf.loc[pdf['monthssincelastdelinquent'] == 'NA', 'monthssincelastdelinquent'] = "0"
print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)
pdf['monthlydebt'] = pdf['monthlydebt'].convert_objects(convert_numeric=True)
pdf['maximumopencredit'] = pdf['maximumopencredit'].convert_objects(convert_numeric=True)
pdf['monthssincelastdelinquent'] = pdf['monthssincelastdelinquent'].convert_objects(convert_numeric=True)
print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)
pdf.drop_duplicates(subset=['loanid'], keep='first')
print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)

abc = pdf.yearsincurrentjob.unique()
print(abc)
pdf['temp1'] = pdf['yearsincurrentjob']

pdf['yearsincurrentjob_bin'] = pdf.loc[pdf.yearsincurrentjob == '< 1 year', 'yearsincurrentjob'] = 0
pdf['yearsincurrentjob_bin'] = pdf.loc[pdf.yearsincurrentjob == 'n/a', 'yearsincurrentjob'] = 0
pdf['yearsincurrentjob_bin'] = pdf.loc[pdf.yearsincurrentjob == '1 year', 'yearsincurrentjob'] = 1
pdf['yearsincurrentjob_bin'] = pdf.loc[pdf.yearsincurrentjob == '2 years', 'yearsincurrentjob'] = 1
pdf['yearsincurrentjob_bin'] = pdf.loc[pdf.yearsincurrentjob == '3 years', 'yearsincurrentjob'] = 1
pdf['yearsincurrentjob_bin'] = pdf.loc[pdf.yearsincurrentjob == '4 years', 'yearsincurrentjob'] = 1
pdf['yearsincurrentjob_bin'] = pdf.loc[pdf.yearsincurrentjob == '5 years', 'yearsincurrentjob'] = 2
pdf['yearsincurrentjob_bin'] = pdf.loc[pdf.yearsincurrentjob == '6 years', 'yearsincurrentjob'] = 2
pdf['yearsincurrentjob_bin'] = pdf.loc[pdf.yearsincurrentjob == '7 years', 'yearsincurrentjob'] = 2
pdf['yearsincurrentjob_bin'] = pdf.loc[pdf.yearsincurrentjob == '8 years', 'yearsincurrentjob'] = 3
pdf['yearsincurrentjob_bin'] = pdf.loc[pdf.yearsincurrentjob == '9 years', 'yearsincurrentjob'] = 3
pdf['yearsincurrentjob_bin'] = pdf.loc[pdf.yearsincurrentjob == '10+ years', 'yearsincurrentjob'] = 4
pdf['yearsincurrentjob_bin'] = pdf['yearsincurrentjob'].convert_objects(convert_numeric=True)
print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)
pdf['yearsincurrentjob'] = pdf['temp1']

pdf['temp'] = pdf['purpose']

pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'other', 'purpose'] = 0
pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'Other', 'purpose'] = 0
pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'major_purchase', 'purpose'] = 0
pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'moving', 'purpose'] = 1
pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'vacation', 'purpose'] = 1
pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'Take a Trip', 'purpose'] = 1
pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'Business Loan', 'purpose'] = 2
pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'small_business', 'purpose'] = 2
pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'Home Improvements', 'purpose'] = 3
pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'Buy a Car', 'purpose'] = 3
pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'Buy House', 'purpose'] = 3
pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'Debt Consolidation', 'purpose'] = 4
pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'Educational Expenses', 'purpose'] = 5
pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'wedding', 'purpose'] = 5
pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'Medical Bills', 'purpose'] = 5
pdf['purpose_bin'] = pdf.loc[pdf.purpose == 'renewable_energy', 'purpose'] = 5
pdf['purpose_bin'] = pdf['purpose'].convert_objects(convert_numeric=True)

pdf['purpose'] = pdf['temp']

print(pdf.purpose_bin.describe())
print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)

# tyt = pdf['currentloanamount'].dtypes

# print(type(tyt))
# print(tyt)
# print (allcolumns)

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
newdf = pdf.select_dtypes(include=numerics)

print(newdf.head(1))

allcolumns = list(newdf.columns.values)
for k in allcolumns:
    print(k)
    print(type(k))

for i in allcolumns:
    print(i)
    nameIs = str(i) + '_norm'
    pdf[nameIs] = (newdf[i] - newdf[i].mean()) / (newdf[i].max() - newdf[i].min())

allcolumns1 = list(pdf.columns.values)
print(allcolumns1)
print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)

print('******************************')

obj = ['object']
newdf1 = pdf.select_dtypes(include=obj)
allcolumns1 = list(newdf1.columns.values)
for i in allcolumns1:
    text = 'unique column in ' + str(i) + ':'
    print(text)
    print(pdf[i].unique())

file_name = '/home/cloudera/Documents/preprocessing1.csv'
pdf.to_csv(file_name, encoding='utf-8')

