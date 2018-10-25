# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:12:39 2017

@author: mohsin.aslam
"""
import pandas as pd 
def Read_file(pathname = "", filename = ""):
    
    import os

    filepath = os.path.join(pathname,filename)
    frame1 = pd.read_csv(filepath)
    return frame1

df = Read_file(pathname = "D:\FinalProjectDataScience", filename = "final.csv")    

from datetime import date

def calculate_age(df): 
    import datetime as DT
    import numpy as np
    df['BirthDateNew'] = pd.to_datetime(df.BirthDate)
    now = pd.Timestamp(DT.datetime.now())
    df['BirthDateNew'] = df['BirthDateNew'].where(df['BirthDateNew'] < now, df['BirthDateNew'] -  np.timedelta64(100, 'Y'))
    df['age'] = (now - df['BirthDateNew']).astype('<m8[Y]')
    return df
    
#df['Age'][i] = today.year - df['BirthDateNew'][i].year - ((today.month, today.day) < (df['BirthDateNew'][i].month, df['BirthDateNew'][i].day))
dfnew = calculate_age(df)

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


##dfnew['YearlyIncome_Bucket'] = dfnew.YearlyIncome

dfnew['YearlyIncome_Bucket2'] = dfnew.loc[dfnew.YearlyIncome_Bucket <= 45000 , 'YearlyIncome_Bucket'] = 0
dfnew['YearlyIncome_Bucket2'] = dfnew.loc[(dfnew.YearlyIncome_Bucket > 45000) & (dfnew.YearlyIncome_Bucket <= 70000) , 'YearlyIncome_Bucket'] = 1
dfnew['YearlyIncome_Bucket2'] = dfnew.loc[(dfnew.YearlyIncome_Bucket > 70000) & (dfnew.YearlyIncome_Bucket <= 90000) , 'YearlyIncome_Bucket'] = 2
dfnew['YearlyIncome_Bucket2'] = dfnew.loc[(dfnew.YearlyIncome_Bucket > 90000) & (dfnew.YearlyIncome_Bucket <= 120000) , 'YearlyIncome_Bucket'] = 3
dfnew['YearlyIncome_Bucket2'] = dfnew.loc[dfnew.YearlyIncome_Bucket > 120000 , 'YearlyIncome_Bucket'] = 4


