# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:38:36 2018

@author: mohsin.aslam
"""
sentence = 'my name is MOHSIN'
import numpy as np
abc = np.matrix([[-1,0,1,2,3,4,5], ['M',1,0,0,0,0,0],['O',0,2,0,0,0,0],['H',0,0,3,0,0,0],['S',0,0,0,4,0,0],['I',0,0,0,0,5,0],['N',0,0,0,0,0,6]])
def my_fsa (sentence,abc):
   words = sentence.split(' ')
   MS = 0
   list1 =[]
   for s in range(1,abc.shape[0]):
       list1.append(abc[s,0])
    
   for word in words:
       MS =0       
       for i in word:
           
           for j in range(1,abc.shape[0]):
               if i not in list1 or i == ' ':
                   
                   MS =0
                   continue
               
               if i == abc[j,0]:                  
                   for k in range(1,abc.shape[1]):                       
                       if MS == int(abc[0,k]):
                                         
                           MS = int(abc[j,k])
                           
                           if MS == abc[1:,1:].astype(int).max():
                               print("Matched")
                           break               
                   
       #if MS == abc[1:,1:].astype(int).max():
           #print("Matched")
       #print (MS)
   if MS != abc[1:,1:].astype(int).max():
       print("Not Matched")
       
        
my_fsa (sentence,abc)          