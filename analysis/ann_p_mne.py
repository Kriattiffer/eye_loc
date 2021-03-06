#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Name:	   ann_v_mne.py
# Purpose:   Erp plotting and analysis -- eye_loc experiment
# Author: Rafael Grigoryan, kriattiffer at gmail.com
# Date: 28.03.18
# ----------------------------------------------------------------------------

from ann_v_mne import Analysis


def aim_detector(stim_id, currrent_aim, interface_type = 'rowcol'):
	def rowcol_aims(stim_id, currrent_aim):
		rows = 	[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11],		  # rows 6x6
				[12, 13, 14, 15, 16, 17], 	 [18, 19, 20, 21, 22, 23],
				[24, 25, 26, 27, 28, 29], 	 [30, 31, 32, 33, 34, 35] ]
		cols = [[0, 6, 12, 18, 24, 30],  [1, 7, 13, 19, 25, 31],
				[2, 8, 14, 20, 26, 32],  [3, 9, 15, 21, 27, 33],
				[4, 10, 16, 22, 28, 34], [5, 11, 17, 23, 29, 35]]

		aim_let_onscreen =  cols + rows
		presencelist = [currrent_aim in a for a in aim_let_onscreen]
		try:
			if presencelist[int (stim_id[1])]:
				# print stim_id[1],  currrent_aim
				return True
			else:
				return False
		except IndexError:
			return False

	def simple_aims(stim_id, currrent_aim):
		if stim_id[1] == currrent_aim:
			# print stim_id[1],  currrent_aim
			return True
		return False

	if interface_type == 'rowcol':
		return rowcol_aims(stim_id, './pics/{}.png'.format(fname))
	elif interface_type == 'simple':
		return simple_aims(stim_id, currrent_aim)




if __name__ == '__main__':
	# data_folder = 'D:/Data/20!8_winter_faces/exp'
	#data_folder = r'..\experimental_data\ann\BCI_EYE_anna_p\valid_eeg_anal'
	data_folder = r'..\experimental_data\prac2018'
	
	# bad_files = 	[]
	# bad_files = [a.split('.')[0] for a in bad_files]
	bad_files = []
	
	Analysis = Analysis(data_folder = data_folder, session = 'play', interface_type = 'rowcol')
	Analysis.bad_files = bad_files

	Analysis.read_fif_if_possible = False
	Analysis.l_freq = 1
	Analysis.h_freq = 35

	
	Analysis.delta = True
	#Analysis.folders = {'small_small': 'SS', 'large_large':'LL', 'large_small':'LS', 'small_large':'SL', 'medium': 'M'}
	#Analysis.folders = {'small_small': 'SS', 'large_small':'LS'}
	Analysis.folders = {'large_small':'LS'}
	#Analysis.plot_colors.update({'small_small': 'red', 'large_small':'blue'})
	Analysis.plot_colors.update({'large_small':'blue'})
	#Analysis.plot_colors.update({'small_small': 'red', 'large_large':'green', 'large_small':'blue', 'small_large':'yellow', 'medium': 'black'})
	#Analysis.dashlist.update({'small_small': (), 'large_large':(), 'large_small':(), 'small_large':(3,1), 'medium': (1,1)})
	#Analysis.plot_colors.update({'small_small': 'black', 'large_large':'gray', 'large_small':'lightgray', 'small_large':'gray', 'medium': 'gray'})
	Analysis.legend_loc= (-1.5, 4.5)
	Analysis.channels = ['time', 'oz', 'o1', 'o2', 'pz', 'p3', 'p4', 'cp1', 'cp2', 'cz', 'ecg', 'a2', 'stim']
	Analysis.ch_types = ['misc', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'ecg', 'misc', 'stim']
	Analysis.reject_eog_artifacts = False
	Analysis.show_raw_eeg = True
	Analysis.show_filtred_eeg = True
	Analysis.test_stats = False


	Analysis.aims = [Analysis.charset.index(a) for a in 'neuroscience'] #[0,5,36,41, 21]
	
	# Analysis.p3average = [2, 3]
	# Analysis.n1average = [1,2]


	Analysis.update_analysis_template()

	#for user in [7]:

	#for user in [1,2,3,4,6,7,8,9]:
	#for user in range(1,11): ### PARTICIPANTS HERE!
	for user in [2]:
		Analysis.user_analysis(user, plot = True)
	#Analysis.grand_average()
	#Analysis.save_peak_data(filename='peaks_ap')
	Analysis.save_peak_data(filename='peaks_prac2018')

