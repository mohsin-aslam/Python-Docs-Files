# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 12:49:25 2017

@author: mohsin.aslam
"""

def Read_file(pathname = "", filename = ""):
    import pandas as pd 
    import os

    filepath = os.path.join(pathname,filename)
    frame1 = pd.read_csv(filepath)
    return frame1
    

df = Read_file("D:\FinalProjectDataScience\AWTest","AWTest-Classification.csv")
l1 = list()

def ApiCall (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x):

    import requests
    g = "kk"

    url = "http://10.1.1.179:13335/predict"

    payload = "{ \"CustomerID\" : ["+a+"] , \"Title\" : [\""+b+"\"] , \"FirstName\" : [\""+c+"\"] , \"MiddleName\" : [\""+d+"\"], \"LastName\" : [\""+e+"\"] , \"Suffix\": [\""+f+"\"], \"AddressLine1\": [\""+g+"\"], \"AddressLine2\" : [\""+h+"\"] , \"City\" : [\""+i+"\"], \"StateProvinceName\" : [\""+j+"\"], \"CountryRegionName\" : [\""+k+"\"], \"PostalCode\" : [\""+l+"\"], \"PhoneNumber\" : [\""+m+"\"], \"BirthDate\" : [\""+n+"\"], \"Education\" : [\""+o+"\"], \"Occupation\" : [\""+p+"\"] , \"Gender\" : [\""+q+"\"], \"MaritalStatus\" : [\""+r+"\"], \"HomeOwnerFlag\" : ["+s+"], \"NumberCarsOwned\":["+t+"], \"NumberChildrenAtHome\": ["+u+"], \"TotalChildren\": ["+v+"] ,\"YearlyIncome\" :["+w+"], \"LastUpdated\": [\""+x+"\"] }\n"
    headers = {
    'content-type': "application/json",
    'cache-control': "no-cache",
    'postman-token': "491385bc-9097-2cc7-2aa5-502d257d34f2"
    }

    response = requests.request("POST", url, data=payload, headers=headers)
    l1.append(response.text)
    print(response.text)



def BatchApiCall (df):
    count = len(df)
    colcount = len(df.columns)
    listCols = df.columns
    counter = 0
    while counter != count :
        for col in listCols:
            if(col == "CustomerID"):
                CustomerID = str(df[col][counter])
            elif(col == "Title"):
                Title = str(df[col][counter] )
            elif(col == "FirstName"):
                FirstName = str(df[col][counter])
            elif(col == "MiddleName"):
                MiddleName = str(df[col][counter])
            elif(col == "LastName"):
                LastName = str(df[col][counter])
            elif(col == "Suffix"):
                Suffix = str(df[col][counter])
            elif(col == "AddressLine1"):
                AddressLine1 = str(df[col][counter])
            elif(col == "AddressLine2"):
                AddressLine2 = str(df[col][counter])
            elif(col == "City"):
                City = str(df[col][counter])
            elif(col == "StateProvinceName"):
                StateProvinceName = str(df[col][counter])
            elif(col == "CountryRegionName"):
                CountryRegionName = str(df[col][counter])
            elif(col == "PostalCode"):
                PostalCode = str(df[col][counter])
            elif(col == "PhoneNumber"):
                PhoneNumber = str(df[col][counter])
            elif(col == "BirthDate"):
                BirthDate = str(df[col][counter])
            elif(col == "Education"):
                Education = str(df[col][counter])
            elif(col == "Occupation"):
                Occupation = str(df[col][counter])
            elif(col == "Gender"):
                Gender = str(df[col][counter])
            elif(col == "MaritalStatus"):
                MaritalStatus = str(df[col][counter])
            elif(col == "HomeOwnerFlag"):
                HomeOwnerFlag = str(df[col][counter])
            elif(col == "NumberCarsOwned"):
                NumberCarsOwned = str(df[col][counter])
            elif(col == "NumberChildrenAtHome"):
                NumberChildrenAtHome = str(df[col][counter])
            elif(col == "TotalChildren"):
                TotalChildren = str(df[col][counter])
            elif(col == "YearlyIncome"):
                YearlyIncome = str(df[col][counter])
            elif(col == "LastUpdated"):
                LastUpdated = str(df[col][counter])
#        print (CustomerID,Title,FirstName,MiddleName,LastName,Suffix,AddressLine1,AddressLine2,City,StateProvinceName,CountryRegionName,PostalCode,PhoneNumber,BirthDate,Education,Occupation,Gender,MaritalStatus,HomeOwnerFlag,NumberCarsOwned,NumberChildrenAtHome,TotalChildren,YearlyIncome,LastUpdated)
#        print (type(CustomerID),type(Title),type(FirstName),type(MiddleName),type(LastName),type(Suffix),type(AddressLine1),type(AddressLine2),type(City),type(StateProvinceName),type(CountryRegionName),type(PostalCode),type(PhoneNumber),type(BirthDate),type(Education),type(Occupation),type(Gender),type(MaritalStatus),type(HomeOwnerFlag),type(NumberCarsOwned),type(NumberChildrenAtHome),type(TotalChildren),type(YearlyIncome),type(LastUpdated))
        
        ApiCall(CustomerID,Title,FirstName,MiddleName,LastName,Suffix,AddressLine1,AddressLine2,City,StateProvinceName,CountryRegionName,PostalCode,PhoneNumber,BirthDate,Education,Occupation,Gender,MaritalStatus,HomeOwnerFlag,NumberCarsOwned,NumberChildrenAtHome,TotalChildren,YearlyIncome,LastUpdated)
        counter = counter + 1



BatchApiCall(df)
#def ApiCall (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x):
#
#    import requests
#
#    url = "http://10.1.1.179:13335/predict"
#
#    payload = "{ \"CustomerID\" : [a] , \"Title\" : [\"b\"] , \"FirstName\" : [\"c\"] , \"MiddleName\" : [\"d\"], \"LastName\" : [\"e\"] , \"Suffix\": [\"f\"], \"AddressLine1\": [\"g\"], \"AddressLine2\" : [\"h\"] , \"City\" : [\"i\"], \"StateProvinceName\" : [\"j\"], \"CountryRegionName\" : [\"k\"], \"PostalCode\" : [\"l\"], \"PhoneNumber\" : [\"m\"], \"BirthDate\" : [\"n\"], \"Education\" : [\"o\"], \"Occupation\" : [\"p\"] , \"Gender\" : [\"q\"], \"MaritalStatus\" : [\"r\"], \"HomeOwnerFlag\" : [s], \"NumberCarsOwned\":[t], \"NumberChildrenAtHome\": [u], \"TotalChildren\": [v] ,\"YearlyIncome\" :[w], \"LastUpdated\": [\"x\"] }\n"
#    headers = {
#    'content-type': "application/json",
#    'cache-control': "no-cache",
#    'postman-token': "491385bc-9097-2cc7-2aa5-502d257d34f2"
#    }
#
#    response = requests.request("POST", url, data=payload, headers=headers)
#
#    print(response.text)

    