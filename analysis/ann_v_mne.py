#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Name:	   ann_v_mne.py
# Purpose:   Erp plotting and analysis -- eye_loc experiment
# Author: Rafael Grigoryan, kriattiffer at gmail.com
# Date: 13.03.18
# ----------------------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt
import os, sys, pickle
import time, datetime
import mne


mne.set_log_level('WARNING')


def aim_detector(stim_id, currrent_aim, interface_type = 'rowcol'):
	def rowcol_aims(stim_id, currrent_aim):
		rows = 	[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11],		  # rows 6x7
				[12, 13, 14, 15, 16, 17], 	 [18, 19, 20, 21, 22, 23],
				[24, 25, 26, 27, 28, 29], 	 [30, 31, 32, 33, 34, 35], 
				[36, 37, 38, 39, 40, 41]]

		cols = [[0, 6, 12, 18, 24, 30, 36],  [1, 7, 13, 19, 25, 31, 37],
				[2, 8, 14, 20, 26, 32, 38],  [3, 9, 15, 21, 27, 33, 39],
				[4, 10, 16, 22, 28, 34, 40], [5, 11, 17, 23, 29, 35, 41]]

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
		return rowcol_aims(stim_id, currrent_aim)
	elif interface_type == 'simple':
		return simple_aims(stim_id, currrent_aim)

def create_events(eeg, evt, aims, interface_type):
	letter_fragments = np.split(evt, np.where(evt[:,1] == 777.)[0])[1:][5:]
	aims_loc = []
	nonaims_loc = []
	cc = 0
	for fr, aim in zip(letter_fragments, aims):
		for stim_id in fr:	
			if aim_detector(stim_id, aim, interface_type = interface_type):
				loc = np.where(eeg[0,:] ==stim_id[0])
				aims_loc.append([loc[0][0], 0, 2])
			else:
				if stim_id[1] < 777:	# exclude technical markers
					loc = np.where(eeg[0,:] ==stim_id[0])
					nonaims_loc.append([loc[0][0], 0, 1])

	aims_loc, nonaims_loc = np.array(aims_loc), np.array(nonaims_loc)
	events = np.vstack((aims_loc, nonaims_loc))
	return events

def _get_peaks(data, ch_names, times, tmin=None, tmax=None, mode='abs', average_lims = False):
	'''
		edited _get_peak function from mne.evoked
	'''
	modes = ('abs', 'neg', 'pos')
	if mode not in modes:
		raise ValueError('The `mode` parameter must be `{modes}`. You gave '
						 'me `{mode}`'.format(modes='` or `'.join(modes),
											  mode=mode))

	if tmin is None:
		tmin = times[0]
	if tmax is None:
		tmax = times[-1]

	if tmin < times.min():
		raise ValueError('The tmin value is out of bounds. It must be '
						 'within {0} and {1}'.format(times.min(), times.max()))
	if tmax > times.max():
		raise ValueError('The tmin value is out of bounds. It must be '
						 'within {0} and {1}'.format(times.min(), times.max()))
	if tmin >= tmax:
		raise ValueError('The tmin must be smaller than tmax')

	time_win = (times >= tmin) & (times <= tmax)
	mask = np.ones_like(data).astype(np.bool)
	mask[:, time_win] = False

	maxfun = np.argmax
	if mode == 'pos':
		if not np.any(data > 0):
			raise ValueError('No positive values encountered. Cannot '
							 'operate in pos mode.')
	elif mode == 'neg':
		if not np.any(data < 0):
			raise ValueError('No negative values encountered. Cannot '
							 'operate in neg mode.')
		maxfun = np.argmin

	masked_index = np.ma.array(np.abs(data) if mode == 'abs' else data,
							   mask=mask)

	#____________________
	tm =  maxfun(masked_index, axis = 1)

	if average_lims:
		return {ch: [times[tm[n]],  np.mean(masked_index[n, tm[n]-average_lims[0]:tm[n]+average_lims[1]])*1e6] for n, ch  in enumerate(ch_names)}
	else:
		return {ch: [times[tm[n]],  masked_index[n, tm[n]]*1e6] for n, ch  in enumerate(ch_names)}

def get_peaks(self, ch_type=None, tmin=None, tmax=None, mode='abs', time_as_index=False, merge_grads=False, average_lims = False):
	'''
		edited get_peak function from mne.evoked.Evoked - rewrite with functools.partial later
	'''
	supported = ('mag', 'grad', 'eeg', 'seeg', 'ecog', 'misc', 'hbo',
				 'hbr', 'None')
	data_picks = mne.io.pick._pick_data_channels(self.info, with_ref_meg=False)
	types_used = set([mne.io.pick.channel_type(self.info, idx) for idx in data_picks])

	if str(ch_type) not in supported:
		raise ValueError('Channel type must be `{supported}`. You gave me '
						 '`{ch_type}` instead.'
						 .format(ch_type=ch_type,
								 supported='` or `'.join(supported)))

	elif ch_type is not None and ch_type not in types_used:
		raise ValueError('Channel type `{ch_type}` not found in this '
						 'evoked object.'.format(ch_type=ch_type))

	elif len(types_used) > 1 and ch_type is None:
		raise RuntimeError('More than one sensor type found. `ch_type` '
						   'must not be `None`, pass a sensor type '
						   'value instead')

	if merge_grads:
		if ch_type != 'grad':
			raise ValueError('Channel type must be grad for merge_grads')
		elif mode == 'neg':
			raise ValueError('Negative mode (mode=neg) does not make '
							 'sense with merge_grads=True')

	meg = eeg = misc = seeg = ecog = fnirs = False
	picks = None
	if ch_type in ('mag', 'grad'):
		meg = ch_type
	elif ch_type == 'eeg':
		eeg = True
	elif ch_type == 'misc':
		misc = True
	elif ch_type == 'seeg':
		seeg = True
	elif ch_type == 'ecog':
		ecog = True
	elif ch_type in ('hbo', 'hbr'):
		fnirs = ch_type

	if ch_type is not None:
		if merge_grads:
			picks = mne.channels.layout._pair_grad_sensors(self.info, topomap_coords=False)
		else:
			picks = mne.io.pick.pick_types(self.info, meg=meg, eeg=eeg, misc=misc,
							   seeg=seeg, ecog=ecog, ref_meg=False,
							   fnirs=fnirs)
	data = self.data
	ch_names = self.ch_names

	if picks is not None:
		data = data[picks]
		ch_names = [ch_names[k] for k in picks]

	if merge_grads:
		data = _merge_grad_data(data)
		ch_names = [ch_name[:-1] + 'X' for ch_name in ch_names[::2]]

	peaks = _get_peaks(data, ch_names, self.times, tmin,
								 tmax, mode, average_lims)

	return peaks
		

class Analysis():
	"""
		Class, holding all data for the analysis
	"""
	def __init__(self, data_folder, interface_type, session = 'play'):		
		self.sfreq = 500
		self.l_freq = 0.1
		self.h_freq = 35
		self.plot_colors =  {'Faces': 'red', 'Facesnoise':'green', 'Letters':'black', 'Noise':'blue', 'aim':'#e41a1c', 'non_aim':'#377eb8'}

		names = [a for a in u'abcdefghijklmonpqrstuvwxyz_1234567890!@#$%^&*()+=-~[]{};:\"\|?.,/<>½¾¿±®©§£¥¢÷µ¬']
		self.aims = [names.index(a) for a in '@neuroscience!'] #[0,5,36,41, 21]
		self.interface_type = interface_type

		self.channels = ['time', "eyes","ecg",   "a2","f3","fz","f4","c5","p7","c3","cz","c4","c6","cp1","cp2","p3","pz","p4","p8","o1","oz","o2", 'stim']
		self.ch_types = ['misc', 'eog', 'ecg', 'misc']+ ['eeg']*18 + ['stim']

		self.data_folder = data_folder
		self.folders = {'Faces': 'fcs', 'Facesnoise':'fn', 'Letters':'ltrs', 'Noise':'ns'}
		
		self.session = session
		self.extension = '.npy'

		self.evoked_non_aim_saved_for_grand_average = {k:False for k in self.folders.keys()}
		self.evoked_aim_saved_for_grand_average = {k:False for k in self.folders.keys()}
		self.cc_non_aim_evoked = {f:0 for f in self.folders.keys()}
		self.cc_aim_evoked = {f:0 for f in self.folders.keys()}

		self.total_data = {}
		self.tota_evoked = {}

	def isfilebad(self, eegfile, bad_files = []):
		'''
			Check file with list of bad files
			Args:
		  		eegfile	(str): path of filename of file to check
		  		bad_files	(list): list of files to reject (can be empty)

		  	Returns:
		       bool: True if file is invalid, False otherwise.

		'''
		if bad_files:
			if os.path.basename(eegfile).split('.')[0] in bad_files:  
				print 'rejected {}'.format(eegfile)
				return True
		return False


	def read_data_files(self, reg_folder, bad_files = False, read_fif_if_possible = True):
		"""
			read .txt, .npy or .fif raw files, optionally savr preprocessed EEG

		    Args:
		        reg_folder (str): folder with one recording
		        bad_files (bool/list): if not False, reject files from bad_files list. 
		        					Reasons may include broken electrodes, 
		        					exessive amounts of alpha rhytm or high ampltude low-frequency noise
		        read_fif_if_possible (bool): if True, read .fif files (if present) instead of .txt or .npy


		    Returns:
		       bool: True if files in directory are valid, False otherwise.

		   """


		if self.extension == 'txt':
			np.load = np.genfromtxt
		files = os.listdir(reg_folder)
		files = [a for a in files if self.session in a]


		if read_fif_if_possible:
			if len([a for a in files if 'data' in a and '.raw.fif' in a]):
				self.events = np.load(os.path.join(reg_folder, 'selfevents.npy'))
				eegfile = os.path.join(reg_folder, [a for a in files if 'data' in a and '.raw.fif' in a][0])
				print eegfile
				if self.isfilebad(eegfile, bad_files = bad_files):
					return False

				self.eeg_filename = eegfile
				self.raw = mne.io.read_raw_fif(eegfile)
				self.raw.load_data()
				return True

		eegfile = os.path.join(reg_folder, [a for a in files if 'data' in a and self.extension in a][0])
		print eegfile
		if self.isfilebad(eegfile, bad_files = bad_files):
			return False
		self.eeg_filename = eegfile

		evtfile = os.path.join(reg_folder, [a for a in files if 'events' in a and self.extension in a][0])
		evt2file = os.path.join(reg_folder, [a for a in files if 'photocell' in a and self.extension in a][0])
		# aimfile = os.path.join(reg_folder, [a for a in files if 'aims' in a][0])
		eeg = np.load(eegfile).T
		eeg = np.vstack( (eeg, np.zeros(np.shape(eeg)[1])) ) # add stim channel

		evt = np.load(evtfile)
		evt2 = np.load(evt2file)
		self.events = create_events(eeg, evt, self.aims, self.interface_type)
		
		if eegfile.split('\\')[1] == '3':				#Ugly fix for electrode placement error for user 3
			self.ch_types[1], self.ch_types[2] = 'ecg', 'eog'
		else:
			self.ch_types[1], self.ch_types[2] = 'eog', 'ecg'

		info = mne.create_info(ch_names=self.channels, sfreq=self.sfreq, ch_types=self.ch_types)
		self.raw = mne.io.RawArray(eeg, info )
		self.raw.add_events(self.events, stim_channel = 'stim')
		montage = mne.channels.read_montage(kind = 'easycap-M1')
		self.raw.set_montage(montage)
			
		return True

	def raw_filter(self): 
		'''
			Filter EEG in self.raw variable
		'''
		self.raw.filter(l_freq = None, h_freq = self.h_freq, picks = range(1,(len(self.channels)-1)), fir_design = 'firwin2')
		self.raw.filter(l_freq = self.l_freq, h_freq = None, picks = range(1,(len(self.channels)-1)), fir_design = 'firwin2')

	def save_intermediate(self, reg_folder):
		extension = '' if self.eeg_filename.split('.')[-1] == 'fif' else '.raw.fif'
		self.raw.save(self.eeg_filename+ extension, overwrite=True)
		self.events.dump(os.path.join(reg_folder, 'selfevents.npy'))

	def reject_eog_contaminated_events(self, plot_eog = False):
		"""
			detect eog events with mne function and reject events, determining 
			epochs  that overlap with +-250 ms around eog events.

		    Args:
		        plot_eog (bool): if True, plot raw file with eog events (blocking)
		    
		    Returns:
		        bool: True
		"""

		eog_events = mne.preprocessing.find_eog_events(self.raw)

		if plot_eog:	
			n_blinks = len(eog_events)
			onset = eog_events[:, 0] / self.raw.info['sfreq'] - 0.25
			duration = np.repeat(0.5, n_blinks)
			self.raw.annotations = mne.Annotations(onset, duration, ['bad blink'] * n_blinks, orig_time=self.raw.info['meas_date'])
			self.raw.plot(events=eog_events, block = True)  # To see the annotated segments.

		event_ids_purge = []
		for n, event in enumerate(self.events):
			for eog_event in [a for a in eog_events[:,0] if abs(a-event[0])<3000]: # optimisation ?
				if 	(event[0]+0.8*self.sfreq > eog_event-0.25*self.sfreq and  event[0]+0.8*self.sfreq < eog_event+0.25*self.sfreq ) or \
					(event[0]-0.1*self.sfreq > eog_event-0.25*self.sfreq and  event[0]-0.1*self.sfreq < eog_event+0.25*self.sfreq )  	:
					event_ids_purge.append(event[0])
		event_ids_purge = set(event_ids_purge)
		clean_events = [event for event in self.events if event[0] not in event_ids_purge]
		clean_events = np.array(clean_events)
		self.events = clean_events

		return True

	def cut_and_average(self, reg):
		"""
			Cut raw EEG by events, average them and save for grand averages

		    Args:
		        reg (str): session name (from self.folders)
		    
		    Returns:
		        mne.Evoked: evoked_aim and evoked_non_aim average waveforms
		"""
		epochs = mne.Epochs(self.raw, events=self.events, event_id={'aim':2, 'non_aim':1}, tmin=-0.1, tmax=0.8, verbose = 'ERROR')
		evoked_aim = epochs['aim'].average()
		evoked_non_aim = epochs['non_aim'].average()

		# evoked_non_aim.plot_joint()

		self.cc_non_aim_evoked[reg] +=len(epochs['non_aim'].events)
		self.cc_aim_evoked[reg] +=len(epochs['aim'].events)		



		if self.evoked_non_aim_saved_for_grand_average[reg]:
			self.evoked_non_aim_saved_for_grand_average[reg].data += evoked_non_aim.data * len(epochs['non_aim'].events)
		else:
			self.evoked_non_aim_saved_for_grand_average[reg] = evoked_non_aim
			self.evoked_non_aim_saved_for_grand_average[reg].data * len(epochs['non_aim'].events)
			
		
		if self.evoked_aim_saved_for_grand_average[reg]:
			self.evoked_aim_saved_for_grand_average[reg].data += evoked_aim.data * len(epochs['aim'].events)
		else:
			self.evoked_aim_saved_for_grand_average[reg] = evoked_aim
			self.evoked_aim_saved_for_grand_average[reg].data * len(epochs['aim'].events)
		

		
		return evoked_aim, evoked_non_aim
	
	
	def get_peaks_from_evoked(self, evoked_aim, evoked_non_aim, delta = False, p3average = [], n1average = []):
		"""
			Detect P300 and N1 peak ampltudes ans latencies

		    Args:
		        reg_folder (Evoked): Aim average waveforms
		        evoked_non_aim (Evoked): non-aim average waveforms
		        delta (bool): if True detect peaks in differential potentials
		        p3average (list): limits of ep averaging (can be length of 2 or empty)
		        n1average (list): limits of ep averaging (can be length of 2 or empty) 

		    Returns:
		       dict: p3peaks and n1peaks (for plotting) and peaks_dict (for statiscical analysis)
		"""
		if delta:
			evoked_aim.data-= evoked_non_aim.data
		# print evoked_aim.data
		# sys.exit()
		p3peaks =  get_peaks(evoked_aim, ch_type='eeg', tmin = 0.25, tmax = 0.6, mode = 'pos', average_lims = p3average)
		n1peaks =  get_peaks(evoked_aim, ch_type='eeg', tmin = 0.08, tmax = 0.25, mode = 'neg', average_lims = n1average)
		# print p3peaks
		peaks_dict = {}
		for p in p3peaks.keys():
			peaks_dict
			peaks_dict['p3a_{}'.format(p) ] = p3peaks[p][1]
			peaks_dict['p3i_{}'.format(p) ] = p3peaks[p][0]
			if p in ["p7", "p8","o1","oz","o2"]:
				peaks_dict['n1a_{}'.format(p) ] = n1peaks[p][1]
				peaks_dict['n1i_{}'.format(p) ] = n1peaks[p][0]
		
		return p3peaks, n1peaks, peaks_dict



	def plot_evoked_response(self, data = {}, p3peaks = {}, n1peaks = {}, fname = str(time.time()), p300_n1_aim_fill = True, peakdot = True):
		"""
			plot topographic EP maps

		    Args:
		        data (dict): dict with waveforms to plot ({key: mne.Evoked}) 
		        p3peaks (dict): p300 ampltudes and latencies
		        n1peaks (dict): n1 ampltudes and latencies
		        fname (str): file name to save picture
		        peakdot (bool): if True plot P3 and N1 peak
		        p300_n1_aim_fill (bool): if True, fill area around peaks
		"""
		def worker():
			'''
				function for threading // TBD
			'''
			fig = plt.figure(figsize=(16,9))

			tpplt = [a for a in mne.viz.topo._iter_topography(	self.raw.info, layout = None, on_pick = None, fig = fig, layout_scale = 0.945,
																fig_facecolor='white', axis_facecolor='white', axis_spinecolor='white')]
			for ax, idx in tpplt:

				[ax.axvline(vl, color='black', linewidth=0.5, linestyle='--') for vl in [0, 0.1, 0.3]]
				ax.axhline(0, color='black', linewidth=0.5)


				idxc = idx - 4

				for i in data.keys():
					ax.plot(data[i].times, data[i].data[idxc]*1.0e6, color=self.plot_colors[i], label = i)
				# ax.plot(evoked_aim.times, evoked_non_aim.data[idxc], color='', label = 'non_aim')

				if peakdot:
					p3p = p3peaks[self.channels[idx]]
					n1p = n1peaks[self.channels[idx]]
					ax.plot(p3p[0], p3p[1], 'o', color = 'black', zorder = 228)

				if p300_n1_aim_fill:
					fsection = [b for a, b in zip(data['aim'].times, data['aim'].data[idxc]*1.0e6) if a >= p3p[0]-0.05 and a < p3p[0]+0.05]
					section = [a for a in data['aim'].times if a >=p3p[0]-0.05 and a <p3p[0]+0.05]
					ax.fill_between(section, fsection, color = 'y', alpha = 0.6)
					
					if idxc in [15, 16, 17]:
						ax.plot(n1p[0], n1p[1], 'o', color = 'black', zorder = 228)
						fsection = [b for a, b in zip(data['aim'].times, data['aim'].data[idxc]*1.0e6) if a >= n1p[0]-0.015 and a < n1p[0]+0.015]
						section = [a for a in data['aim'].times if a >=n1p[0]-0.015 and a <n1p[0]+0.015]
						ax.fill_between(section, fsection, color = 'green', alpha = 0.6)

			legend = tpplt[0][0].legend( loc= (-1.5,0), prop={'size': 10})
			# plt.show()
			plt.savefig('./pics/{}.png'.format(fname),  dpi = 100)
			plt.close()
			return

		worker()
		


	def grand_average(self):
		'''
			Calculate and plot grand average waveforms 
		'''
		for k in self.evoked_aim_saved_for_grand_average.keys():
			interm = self.evoked_aim_saved_for_grand_average[k].data
			interm /= self.cc_aim_evoked[k]
			self.evoked_aim_saved_for_grand_average[k].data = interm


		self.plot_evoked_response(	{k:self.evoked_aim_saved_for_grand_average[k] for k in self.evoked_aim_saved_for_grand_average.keys()},
									p300_n1_aim_fill = False, peakdot = False,
									fname = '_ga')



		# print self.evoked_aim_saved_for_grand_average['Noise'].data
	
	def save_peak_data(self, filename = 'peaks'):
		'''
		Save dictionary of P300 and N100 data
		{channel:{reg:[ep1, ep2...]}}

		Args:
		    filename (str): basename of pickle datafile 
		'''
		users = self.total_data.keys()
		regs1 = self.total_data[users[0]].keys()
		channels = self.total_data[users[0]][regs1[0]].keys()
		regs = self.folders.keys()

		total_data_comp = {a:{b:[] for b in regs} for a in channels}
		print self.total_data
		for user in users:
			for reg in regs:
				for channel in channels:
					try:
						epdata = self.total_data[user][reg][channel]
					except KeyError:
						epdata = np.NaN
					total_data_comp[channel][reg].append(epdata)

		with open('{}.pickle'.format(filename), 'wb') as file_obj:
			pickle.dump(total_data_comp, file_obj)
		print total_data_comp


	
	def user_analysis(self, user, plot = True, save_intermediate = True, bad_files = []):
		"""
			Main function for one user

		    Args:
		        user (str/int): user folder name
		  		save_intermediate (bool): if True, save raw eeg files with events and applied filters to .fif files
		  		bad_files	(list): list of files to reject (can be empty)
		"""


		peaks = []

		
		user_folder =  os.path.join(self.data_folder, str(user))
		# user_data = {'user':user}
		user_data = {}
		user_peak_data = {}

		for reg in self.folders.keys():
			isfilevalid = self.read_data_files(os.path.join(user_folder, self.folders[reg]), bad_files = bad_files, read_fif_if_possible = True)
			if isfilevalid:
				# self.raw.plot(block = True)
				if np.isclose(self.raw.info['highpass'], self.l_freq) and np.isclose(self.raw.info['lowpass'], self.h_freq): #default values for unfiltered data
					pass
				else:
					print 'filtering'
					self.raw_filter()
					if save_intermediate:
						print 'saving'
						self.save_intermediate(reg_folder = os.path.join(user_folder, self.folders[reg]))


				self.reject_eog_contaminated_events(plot_eog=False)
				evoked_aim, evoked_non_aim = self.cut_and_average(reg)
				user_data[reg] = {'aim':evoked_aim, 'non_aim':evoked_non_aim}

				p3peaks, n1peaks, peaks_dict = self.get_peaks_from_evoked(evoked_aim, evoked_non_aim, 	delta = True)
																										# p3average = [5, 6],
																										# n1average = [1,2]
																										# )
				user_peak_data[reg] = peaks_dict
				if plot:
					self.plot_evoked_response(	{'aim':evoked_aim, 'non_aim':evoked_non_aim}, 
											# p300_n1_aim_fill = False, peakdot = False,
											p3peaks = p3peaks, n1peaks = n1peaks, 
											fname = '{}_{}'.format(user, reg))
		if plot:
			self.plot_evoked_response(	{k:user_data[k]['aim'] for k in user_data.keys()},
									p300_n1_aim_fill = False, peakdot = False,
									
									fname = '_{}'.format(user))

		self.total_data[str(user)] = user_peak_data
		# sys.exit()



if __name__ == '__main__':
	data_folder = 'D:/Data/20!8_winter_faces/exp'
	
	bad_files = 	['_data_facesnoise__play_14_19__31_10.npy',  # low freq
					 '_data_faces__play_16_13_b_22_11.npy',  # low freq
					 '_data_faces__play_14_33__31_10.npy',  # low freq
					 '_data_faces__play_16_26__12_10.npy',  # alpha
					 '_data_faces__play_15_10__01_11.npy',  # alpha
					 '_data_faces__play_16_17__25_11.npy',  # alpha
					 '_data_letters__play_14_05__31_10.npy',  # low freq
					 '_data_noise__play_16_13__12_10.npy',  # alpha
					 '_data_noise__play_16_29__25_11.npy']
	bad_files = [a.split('.')[0] for a in bad_files]
	
	Analysis = Analysis(data_folder = data_folder, session = 'play', interface_type = 'rowcol')
	for user in [1,2,5,6,7,8,9,10,12,14,16,17]:
		Analysis.user_analysis(user, plot = False, bad_files = bad_files)
	Analysis.grand_average()
	Analysis.save_peak_data(filename='peaks_dot')