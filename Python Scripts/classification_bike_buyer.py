# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:36:32 2017

@author: mohsin.aslam
"""

import pandas as pd
import numpy as np

# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame

##Calc Age
def azureml_main(dataframe1 = None, dataframe2 = None):

    import datetime as DT
    print(dataframe1.head())
    dataframe1['BirthDateNew'] = pd.to_datetime(dataframe1.BirthDate)
    now = pd.Timestamp(DT.datetime.now())
    dataframe1['BirthDateNew'] = dataframe1['BirthDateNew'].where(dataframe1['BirthDateNew'] < now, dataframe1['BirthDateNew'] -  np.timedelta64(100, 'Y'))
    dataframe1['age'] = (now - dataframe1['BirthDateNew']).astype('<m8[Y]')
    return dataframe1

##Location merge
import pandas as pd

# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1 = None, dataframe2 = None):

    dataframe1['Location'] = dataframe1['City'] + " " +  dataframe1['StateProvinceName'] + " " + dataframe1['CountryRegionName']
    return dataframe1

    

##Age Bucket
import pandas as pd

# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dfnew = None, dataframe2 = None):

    dfnew['Age_Bucket'] = dfnew.age
    dfnew['Age_Bucket'] = dfnew['Age_Bucket'].convert_objects(convert_numeric=True)

    dfnew['Age_Bucket3'] = dfnew.loc[dfnew.Age_Bucket <= 23.0 , 'Age_Bucket'] = 0
    dfnew['Age_Bucket3'] = dfnew.loc[(dfnew.Age_Bucket > 23.0) & (dfnew.Age_Bucket <= 30.0) , 'Age_Bucket'] = 1
    dfnew['Age_Bucket3'] = dfnew.loc[(dfnew.Age_Bucket > 30.0) & (dfnew.Age_Bucket <= 37.0) , 'Age_Bucket'] = 2
    dfnew['Age_Bucket3'] = dfnew.loc[(dfnew.Age_Bucket > 37.0) & (dfnew.Age_Bucket <= 44.0) , 'Age_Bucket'] = 3
    dfnew['Age_Bucket3'] = dfnew.loc[(dfnew.Age_Bucket > 44.0) & (dfnew.Age_Bucket <= 51.0), 'Age_Bucket'] = 4
    dfnew['Age_Bucket3'] = dfnew.loc[(dfnew.Age_Bucket > 51.0) & (dfnew.Age_Bucket <= 58.0), 'Age_Bucket'] = 5
    dfnew['Age_Bucket3'] = dfnew.loc[(dfnew.Age_Bucket > 58.0) & (dfnew.Age_Bucket <= 65.0) , 'Age_Bucket'] = 6
    dfnew['Age_Bucket3'] = dfnew.loc[dfnew.Age_Bucket > 65.0 , 'Age_Bucket'] = 7
    return dfnew


##Flags
import pandas as pd

# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dfnew = None, dataframe2 = None):

    dfnew['HomeOwnerFlagCategorical'] = dfnew['HomeOwnerFlag']
    dfnew['HomeOwnerFlagCategorical'] = dfnew['HomeOwnerFlagCategorical'].replace(1,"Yes")
    dfnew['HomeOwnerFlagCategorical'] = dfnew['HomeOwnerFlagCategorical'].replace(0,"No")
    return dfnew

    