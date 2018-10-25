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

#import numpy as np


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

pdf.drop_duplicates(subset=['loanid'], keep ='first')

print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)

if(pdf['creditscore'] >= 1000):
    pdf['creditscore']= pdf['creditscore']/10

print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)


pdf.creditscore.fillna(pdf.creditscore.median())

print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)

pdf.annualincome.fillna(pdf.creditscore.median())
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
pdf.drop_duplicates(subset=['loanid'], keep ='first')
print(pdf.describe())
print(pdf.count())
print(pdf.dtypes)


'''
#instead of using pd.head()
df.show(1)

#using count()
print (df.count())

#Exploring the dtypes
print ('the types are...')
print (df.schema)

#using describe
print ('describing the data')
df[['annualincome','creditscore']].describe().show()

print ('Checking...')
dfNew = df[['creditscore']]
dfNew.show(5)

## dot notation does not work in pyspark sql
# print ('checking other...')
# df2 = df.creditscore
# df2.show(5)
##
print ('Checking for .loc')
dfCheck = df.where((df.creditscore < 900))
dfCheck[['creditscore']].show(10)
print ('the count is...')
print (dfCheck.count())
####
# print ('experimenting..')
# df.where((df.creditscore < 900 )) = 0
# dfCheck = df.where((df.creditscore < 900))
# print (dfCheck.count())



###checking
# dfA = df
# print ('original data set')
# dfA[['creditscore']].show(10)
#
# dfA.withColumn('c1', when(dfA.c1.isNotNull(), 1).otherwise(0))
#   .withColumn('c2', when(dfA.c2.isNotNull(), 1).otherwise(0))
#   .withColumn('c3', when(dfA.c3.isNotNull(), 1).otherwise(0))

#TODO: testing

###
#Use this for changing values matching a certain criteria...
print ('testing .loc methods......')
dfA = df
dfA = dfA.withColumn('creditscore',
    F.when(dfA['creditscore'] >=900,-1).
    otherwise(dfA['creditscore']))

dfCheck2 = dfA.where((dfA.creditscore == -1))
dfCheck2[['creditscore']].show(10)

##############
#Casting
##Converting column dtypes
print ('checking casting')
changedTypedf = df.withColumn("monthlydebt", df["monthlydebt"].cast("double"))
changedTypedf = df.withColumn("maximumopencredit", df["maximumopencredit"].cast("double"))

changedTypedf[['maximumopencredit']].show(10)
print ('schema is ...')
print (df.schema)

###
print ('the type is... ')
print (type(df))


meanbef2 = F.mean(df['creditscore'])
df.select(meanbef2).show()
abc = np.median(df['creditscore'])
print(abc)

dfD = df
dfD = dfD.withColumn('creditscore',
    F.when(dfD['creditscore'] >=1000, dfD['creditscore']/10).
    otherwise(dfD['creditscore']))

meanaf2 = F.mean(dfD['creditscore'])
dfD.select(meanaf2).show()



meanbef = F.mean(df['annualincome'])
print(type(meanbef))
df.select(meanbef).show()


dfB = df
dfB = dfB.withColumn('annualincome',F.when(dfB['annualincome'] == '',72498.11727800309).otherwise(dfB['annualincome']))

meanaf = F.mean(dfB['annualincome'])
dfB.select(meanaf).show()

meanbef1 = F.mean(df['creditscore'])
df.select(meanbef1).show()

dfC = df
dfC = dfC.withColumn('creditscore',F.when(dfC['creditscore'] == '', abc).otherwise(dfC['creditscore']))

meanaf1 = F.mean(dfC['creditscore'])
dfC.select(meanaf1).show() '''
