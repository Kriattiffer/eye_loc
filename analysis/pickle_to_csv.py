# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 17:49:02 2018

@author: apron
"""

### pickle to csv ###

import pandas as pd
import pickle


#d=pickle.load(open('peaks_av.pickle', 'rb'))
d=pickle.load(open('peaks_prac2018.pickle', 'rb'))

df=pd.DataFrame.from_dict({str(i)+' '+ str(j): d[i][j] for i in d.keys() for j in d[i].keys()}, orient='index')

df.reset_index(level=0, inplace=True)
df['position'], df['mode'] = df['index'].str.split(' ').str
df.drop('index', axis=1, inplace=True)


pdf=df.pivot(columns='position', index='mode')
parts=[pdf[i] for i in range(8)] # NUMBER OF PARTICIPANTS
fin=pd.concat(parts, keys=[str(i) for i in range(8)]) # NUMBER OF PARTICIPANTS
fin.reset_index(level=0, inplace=True)
fin.rename(columns={'level_0':'participant'}, inplace=True)
#fin.to_csv('erp_data.csv')
fin.to_csv('erp_data_prac2018.csv')