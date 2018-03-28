#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Name:	   peak_data_stats.py
# Purpose:   Erp statistics
# Author: Rafael Grigoryan, kriattiffer at gmail.com
# Date: 23.03.18
# ----------------------------------------------------------------------------


import numpy as np
import pandas as pd

from scipy import stats
import sys, itertools, pickle

def wilcox(data):
	output = []
	for comb in itertools.combinations(data.keys(), 2):
		output.append([str(comb),  [ np.mean(data[a]) for a in comb ], stats.wilcoxon(*[ data[a] for a in comb ])])
	return output

def student(data):
	output = []
	for comb in itertools.combinations(data.keys(), 2):
		output.append([str(comb),  [ np.mean(data[a]) for a in comb ], stats.ttest_rel(*[ data[a] for a in comb ])])
	return output


def shapiro(data):
	ret = {}
	for var in data.keys():
		ret[var] = True if stats.shapiro(data[var])[1] >= 0.05 else False
	return ret




if __name__ == '__main__':
	with open('peaks_dot.pickle', 'rb') as file_obj:
		total_data = pickle.load(file_obj)
	eye_data = pd.DataFrame.from_csv('eye_measures_data.csv')
	print eye_data
	# sys.exit()
	


	# sys.exit()
	print total_data
	for channel in total_data.keys():
		normality = shapiro(total_data[channel])
		# print normality
		# pairwise_stats = student(total_data[channel])
		pairwise_stats = wilcox(total_data[channel])
		for ps in pairwise_stats:
			if ps[2].pvalue < 0.05/6:
				print channel, ps
