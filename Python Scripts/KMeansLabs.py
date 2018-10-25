# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:44:56 2018

@author: mohsin.aslam
"""

def Read_file(pathname = "", filename = ""):
    import pandas as pd 
    import os
    filepath = os.path.join(pathname,filename)
    print (filepath)
    frame1 = pd.read_csv(filepath)
    return frame1
    
    
