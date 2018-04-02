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

def corr(data):
	channels = [a for a in data.keys() if a != 'File']
	for channel1 in channels:
		for channel2 in channels:
			if channel1 != channel2:
				for regs in data[channel1].keys():
					# print data[channel1][reg]
					st = stats.kendalltau(data[channel1][reg], data[channel2][reg])
					if st.pvalue < 0.05/(len(channels)**2):
						print channel1, channel2, st

def shapiro(data):
	ret = {}
	for var in data.keys():
		ret[var] = True if stats.shapiro(data[var])[1] >= 0.05 else False
	return ret


def pairwise (total_data, stats = False, norm = False):
	for channel in [a for a in total_data.keys() if a != 'File']:
		if norm:
			normality = shapiro(total_data[channel])
			# print channel, normality
		elif stats:
			if stats == 'student':
				pairwise_stats = student(total_data[channel])
			elif stats == 'wilcoxon':
				pairwise_stats = wilcox(total_data[channel])

			for ps in pairwise_stats:
				if ps[2].pvalue < 0.05/6:
				# if 1:
					print channel, ps


if __name__ == '__main__':
	with open('peaks_av.pickle', 'rb') as file_obj:
		total_data = pickle.load(file_obj)
	eye_data = True
	if eye_data:
		flist =  [a for a in total_data['File']]#.split('.')[0]]
		eye_data = pd.read_csv('eye_measures_data.csv')
		eye_data['User'] = [int(a.split('-')[0]) for a in eye_data['File']]
		eye_data['Reg'] = [a.split('-')[1].split('_')[0].capitalize() for a in eye_data['File']]
		eye_data =  eye_data.sort_values(by=['User'])
		for col in eye_data.columns:
			if col not in ['File', 'User', 'Reg', 'Unnamed: 0']:
				# print eye_data['File'][eye_data['Reg'] == 'Facesnoise']
				dic = {}
				for reg in set(eye_data['Reg']):
					dic[reg] =  list(eye_data[col][eye_data['Reg'] == reg]) 
				total_data[col] = dic

	pairwise(total_data, stats = 'wilcoxon')
	# corr(total_data)

