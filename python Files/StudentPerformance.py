# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:03:40 2017

@author: mohsin.aslam
"""

import pandas as pd 
def Read_file(pathname = "", filename = ""):
    
    import os

    filepath = os.path.join(pathname,filename)
    frame1 = pd.read_csv(filepath)
    return frame1
    
    
df = Read_file("D:\StudentPerformance","SmoteDataset.csv")


df1 = df.gender
df1.replace("M","1",True)
df1.replace("F","0",True)
df["GenderBinary"] = df1

df2 = df.Semester
df2.replace("S","1",True)
df2.replace("F","0",True)
df["SemesterBinary"] = df2

df3 = df.SectionID
df3.replace("A","0",True)
df3.replace("B","1",True)
df3.replace("c","2",True)
df["SectionIDBinary"] = df3

df4 = df.Relation
df4.replace("Father","0",True)
df4.replace("Mum","1",True)
df["RelationBinary"] = df4

df5 = df.StudentAbsenceDays
df5.replace("Under-7","0",True)
df5.replace("Above-7","1",True)
df["StudentAbsenceDaysBinary"] = df5

df6 = df.ParentschoolSatisfaction
df6.replace("Bad","0",True)
df6.replace("Good","1",True)
df["ParentschoolSatisfactionBinary"] = df6

df6 = df.ParentAnsweringSurvey
df6.replace("No","0",True)
df6.replace("Yes","1",True)
df["ParentAnsweringSurveyBinary"] = df6

df7 = df.StageID
df7.replace("lowerlevel","0",True)
df7.replace("MiddleSchool","1",True)
df7.replace("HighSchool","2",True)
df["StageIDBinary"] = df7


df.Class.replace("H","High",True)
df.Class.replace("M","Medium",True)
df.Class.replace("L","Low",True)

df['raisehandBin'] = df.raisedhands
df['raisehandBin1'] = df.loc[df.raisehandBin <= 10 , 'raisehandBin'] = 0
df['raisehandBin1'] = df.loc[(df.raisehandBin > 10) & (df.raisehandBin <= 20) , 'raisehandBin'] = 1
df['raisehandBin1'] = df.loc[(df.raisehandBin > 20) & (df.raisehandBin <= 30) , 'raisehandBin'] = 2
df['raisehandBin1'] = df.loc[(df.raisehandBin > 30) & (df.raisehandBin <= 40) , 'raisehandBin'] = 3
df['raisehandBin1'] = df.loc[(df.raisehandBin > 40) & (df.raisehandBin <= 50) , 'raisehandBin'] = 4
df['raisehandBin1'] = df.loc[(df.raisehandBin > 50) & (df.raisehandBin <= 60) , 'raisehandBin'] = 5
df['raisehandBin1'] = df.loc[(df.raisehandBin > 60) & (df.raisehandBin <= 70) , 'raisehandBin'] = 6
df['raisehandBin1'] = df.loc[(df.raisehandBin > 70) & (df.raisehandBin <= 80) , 'raisehandBin'] = 7
df['raisehandBin1'] = df.loc[(df.raisehandBin > 80) & (df.raisehandBin <= 90) , 'raisehandBin'] = 8
df['raisehandBin1'] = df.loc[(df.raisehandBin > 90) & (df.raisehandBin <= 100) , 'raisehandBin'] = 9
df['raisehandBin1'] = df.loc[df.raisehandBin > 100 , 'raisehandBin'] = 10


df['VisitedResourcesBin'] = df.VisITedResources
df['VisitedResourcesBin1'] = df.loc[df.VisitedResourcesBin <= 10 , 'VisitedResourcesBin'] = 0
df['VisitedResourcesBin1'] = df.loc[(df.VisitedResourcesBin > 10) & (df.VisitedResourcesBin <= 20) , 'VisitedResourcesBin'] = 1
df['VisitedResourcesBin1'] = df.loc[(df.VisitedResourcesBin > 20) & (df.VisitedResourcesBin <= 30) , 'VisitedResourcesBin'] = 2
df['VisitedResourcesBin1'] = df.loc[(df.VisitedResourcesBin > 30) & (df.VisitedResourcesBin <= 40) , 'VisitedResourcesBin'] = 3
df['VisitedResourcesBin1'] = df.loc[(df.VisitedResourcesBin > 40) & (df.VisitedResourcesBin <= 50) , 'VisitedResourcesBin'] = 4
df['VisitedResourcesBin1'] = df.loc[(df.VisitedResourcesBin > 50) & (df.VisitedResourcesBin <= 60) , 'VisitedResourcesBin'] = 5
df['VisitedResourcesBin1'] = df.loc[(df.VisitedResourcesBin > 60) & (df.VisitedResourcesBin <= 70) , 'VisitedResourcesBin'] = 6
df['VisitedResourcesBin1'] = df.loc[(df.VisitedResourcesBin > 70) & (df.VisitedResourcesBin <= 80) , 'VisitedResourcesBin'] = 7
df['VisitedResourcesBin1'] = df.loc[(df.VisitedResourcesBin > 80) & (df.VisitedResourcesBin <= 90) , 'VisitedResourcesBin'] = 8
df['VisitedResourcesBin1'] = df.loc[(df.VisitedResourcesBin > 90) & (df.VisitedResourcesBin <= 100) , 'VisitedResourcesBin'] = 9
df['VisitedResourcesBin1'] = df.loc[df.VisitedResourcesBin > 100 , 'VisitedResourcesBin'] = 10

df['AnnouncementsViewBin'] = df.AnnouncementsView
df['AnnouncementsViewBin1'] = df.loc[df.AnnouncementsViewBin <= 10 , 'AnnouncementsViewBin'] = 0
df['AnnouncementsViewBin1'] = df.loc[(df.AnnouncementsViewBin > 10) & (df.AnnouncementsViewBin <= 20) , 'AnnouncementsViewBin'] = 1
df['AnnouncementsViewBin1'] = df.loc[(df.AnnouncementsViewBin > 20) & (df.AnnouncementsViewBin <= 30) , 'AnnouncementsViewBin'] = 2
df['AnnouncementsViewBin1'] = df.loc[(df.AnnouncementsViewBin > 30) & (df.AnnouncementsViewBin <= 40) , 'AnnouncementsViewBin'] = 3
df['AnnouncementsViewBin1'] = df.loc[(df.AnnouncementsViewBin > 40) & (df.AnnouncementsViewBin <= 50) , 'AnnouncementsViewBin'] = 4
df['AnnouncementsViewBin1'] = df.loc[(df.AnnouncementsViewBin > 50) & (df.AnnouncementsViewBin <= 60) , 'AnnouncementsViewBin'] = 5
df['AnnouncementsViewBin1'] = df.loc[(df.AnnouncementsViewBin > 60) & (df.AnnouncementsViewBin <= 70) , 'AnnouncementsViewBin'] = 6
df['AnnouncementsViewBin1'] = df.loc[(df.AnnouncementsViewBin > 70) & (df.AnnouncementsViewBin <= 80) , 'AnnouncementsViewBin'] = 7
df['AnnouncementsViewBin1'] = df.loc[(df.AnnouncementsViewBin > 80) & (df.AnnouncementsViewBin <= 90) , 'AnnouncementsViewBin'] = 8
df['AnnouncementsViewBin1'] = df.loc[(df.AnnouncementsViewBin > 90) & (df.AnnouncementsViewBin <= 100) , 'AnnouncementsViewBin'] = 9
df['AnnouncementsViewBin1'] = df.loc[df.AnnouncementsViewBin > 100 , 'AnnouncementsViewBin'] = 10

df['DiscussionBin'] = df.Discussion
df['DiscussionBin1'] = df.loc[df.DiscussionBin <= 10 , 'DiscussionBin'] = 0
df['DiscussionBin1'] = df.loc[(df.DiscussionBin > 10) & (df.DiscussionBin <= 20) , 'DiscussionBin'] = 1
df['DiscussionBin1'] = df.loc[(df.DiscussionBin > 20) & (df.DiscussionBin <= 30) , 'DiscussionBin'] = 2
df['DiscussionBin1'] = df.loc[(df.DiscussionBin > 30) & (df.DiscussionBin <= 40) , 'DiscussionBin'] = 3
df['DiscussionBin1'] = df.loc[(df.DiscussionBin > 40) & (df.DiscussionBin <= 50) , 'DiscussionBin'] = 4
df['DiscussionBin1'] = df.loc[(df.DiscussionBin > 50) & (df.DiscussionBin <= 60) , 'DiscussionBin'] = 5
df['DiscussionBin1'] = df.loc[(df.DiscussionBin > 60) & (df.DiscussionBin <= 70) , 'DiscussionBin'] = 6
df['DiscussionBin1'] = df.loc[(df.DiscussionBin > 70) & (df.DiscussionBin <= 80) , 'DiscussionBin'] = 7
df['DiscussionBin1'] = df.loc[(df.DiscussionBin > 80) & (df.DiscussionBin <= 90) , 'DiscussionBin'] = 8
df['DiscussionBin1'] = df.loc[(df.DiscussionBin > 90) & (df.DiscussionBin <= 100) , 'DiscussionBin'] = 9
df['DiscussionBin1'] = df.loc[df.DiscussionBin > 100 , 'DiscussionBin'] = 10

df.drop('raisehandBin1',axis=1,inplace = True)
df.drop('VisitedResourcesBin1',axis=1,inplace = True)
df.drop('AnnouncementsViewBin1',axis=1,inplace = True)
df.drop('DiscussionBin1',axis=1,inplace = True)