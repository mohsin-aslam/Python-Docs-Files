# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:15:41 2017

@author: mohsin.aslam
"""

def Read_file(pathname = "", filename = ""):
    import pandas as pd 
    import os
    
    os.chdir("F:")
    filepath = os.path.join(pathname,filename)
    print (filepath)
    frame1 = pd.read_csv(filepath)
    return frame1