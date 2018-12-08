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
import os, sys, pickle, copy
import time, datetime
import warnings

import mne

mne.set_log_level('WARNING')
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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

def create_events(eeg, evt, aims, interface_type, skip_n = 5):
	letter_fragments = np.split(evt, np.where(evt[:,1] == 777.)[0])[1:]
	if skip_n:
		letter_fragments = 	letter_fragments[skip_n:]
	else:
		pass
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
		self.delta = True
		self.p3average = []
		self.n1average = []
		self.bad_files = []

		self.aim_word = '@neuroscience!'
		self.plot_colors =  {'aim':'#e41a1c', 'non_aim':'#377eb8', 'delta':'black'}
		self.dashlist={'aim':(), 'non_aim':(), 'delta':()}
		self.legend_loc= (-1.5,0)
		self.charset = [a for a in u'abcdefghijklmonpqrstuvwxyz_1234567890!@#$%^&*()+=-~[]{};:\"\|?.,/<>½¾¿±®©§£¥¢÷µ¬']
		self.aims = [self.charset.index(a) for a in self.aim_word] #[0,5,36,41, 21]
		self.interface_type = interface_type

		self.channels = ['time', "eyes","ecg",   "a2","f3","fz","f4","c5","po7","c3","cz","c4","c6","cp1","cp2","p3","pz","p4","po8","o1","oz","o2", 'stim']
		self.ch_types = ['misc', 'eog', 'ecg', 'misc']+ ['eeg']*18 + ['stim']

		self.data_folder = data_folder
		self.folders = {'Faces': 'fcs', 'Facesnoise':'fn', 'Letters':'ltrs', 'Noise':'ns'}
		
		self.show_raw_eeg = False
		self.show_filtred_eeg = False

		self.session = session
		self.extension = '.npy'
		self.reject_eog_artifacts = False
		self.read_fif_if_possible = True
		
		self.fix_folder_ecg_eog = []
		self.test_stats = False
		self.update_analysis_template()

	def update_analysis_template(self):
		'''
			Update derivatives after changing class variables
			need to be run when importing Analysis module.
		'''
		self.evoked_non_aim_saved_for_grand_average = {k:False for k in self.folders.keys()}
		self.evoked_aim_saved_for_grand_average = {k:False for k in self.folders.keys()}
		self.cc_non_aim_evoked = {f:0 for f in self.folders.keys()}
		self.cc_aim_evoked = {f:0 for f in self.folders.keys()}

		self.total_data = {}
		self.tota_evoked = {}

	def isfilebad(self, eegfile):
		'''
			Check file with list of bad files
			Args:
		  		eegfile	(str): path of filename of file to check

		  	Returns:
		       bool: True if file is invalid, False otherwise.

		'''
		if self.bad_files:
			if os.path.basename(eegfile).split('.')[0] in self.bad_files:  
				print 'rejected {}'.format(eegfile)
				return True
		return False


	def read_data_files(self, reg_folder):
		"""
			read .txt, .npy or .fif raw files, optionally savr preprocessed EEG

		    Args:
		        reg_folder (str): folder with one recording

		    Returns:
		       bool: True if files in directory are valid, False otherwise.

		   """

		if self.extension == 'txt':
			np.load = np.genfromtxt
		files = os.listdir(reg_folder)
		files = [a for a in files if self.session.upper() in a.upper()]


		if self.read_fif_if_possible:
			if len([a for a in files if 'data'.upper() in a.upper() and '.raw.fif'.upper() in a.upper()]):
				self.events = np.load(os.path.join(reg_folder, 'selfevents.npy'))
				eegfile = os.path.join(reg_folder, [a for a in files if 'DATA' in a.upper() and '.raw.fif'.upper() in a.upper()][0])
				print eegfile
				if self.isfilebad(eegfile):
					return False

				self.eeg_filename = eegfile
				self.raw = mne.io.read_raw_fif(eegfile)
				self.raw.load_data()
				return True

		eegfile = os.path.join(reg_folder, [a for a in files if 'DATA' in a.upper() and self.extension.upper() in a.upper()][0])
		print eegfile
		if self.isfilebad(eegfile):
			return False
		self.eeg_filename = eegfile

		evtfile = os.path.join(reg_folder, [a for a in files if 'EVENTS' in a.upper() and self.extension.upper() in a.upper()][0])
		evt2file = os.path.join(reg_folder, [a for a in files if 'PHOTOCELL' in a.upper() and self.extension.upper() in a.upper()][0])
		#aimfile = os.path.join(reg_folder, [a for a in files if 'aims' in a][0])
		eeg = np.load(eegfile).T
		eeg = np.vstack( (eeg, np.zeros(np.shape(eeg)[1])) ) # add stim channel

		evt = np.load(evtfile)
		evt2 = np.load(evt2file)
		# print reg_folder

		self.events = create_events(eeg, evt, self.aims, self.interface_type, skip_n = 5)
		
		if self.fix_folder_ecg_eog:
			if os.path.dirname(eegfile).split('\\')[-2] in self.fix_folder_ecg_eog:				#Ugly fix for electrode placement error for user 3
				print 'fixing'
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
			important -- read from npy files when refiltering -- TBD
		'''
		self.raw.filter(l_freq = None, h_freq = self.h_freq, picks = range(1,(len(self.channels)-1)), fir_design = 'firwin2')
		self.raw.filter(l_freq = self.l_freq, h_freq = None, picks = range(1,(len(self.channels)-1)), fir_design = 'firwin2')

	def save_intermediate(self, reg_folder):
		extension = '' if self.eeg_filename.split('.')[-1] == 'fif' else '.raw.fif'
		self.raw.save(self.eeg_filename+ extension, overwrite=True)
		self.events.dump(os.path.join(reg_folder, 'selfevents.npy'))

	def reject_eog_contaminated_events(self):
		"""
			detect eog events with mne function and reject events, determining 
			epochs  that overlap with +-250 ms around eog events.
		    
		    Returns:
		        bool: True
		"""

		eog_events = mne.preprocessing.find_eog_events(self.raw)

		if self.show_raw_eeg:	
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
		print len(self.events)
		reject = dict(eeg=0.001)
		# reject = None

		epochs = mne.Epochs(self.raw, events=self.events, event_id={'aim':2, 'non_aim':1}, tmin=-0.1, tmax=1, verbose = 'ERROR', reject = reject)
		epochs.drop_bad()
		print([a for a in epochs.drop_log if a !=[]])

		# self.raw.plot(events=self.events, block = True, n_channels = 2, duration = 2,  order = [4, 19, 1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,20,22], scalings =  dict(eeg=20e-4))  # To see the annotated segments.
		epochs.load_data()
		epochs = epochs.apply_baseline(baseline = (0,0))

		# print np.shape(epochs['aim']._data)
		# plt.plot(epochs['aim']._data[:,19,:].T)
		# print np.shape(epochs['aim']._data[:,19,:].T)
		# #plt.plot(epochs['aim']._data[:,12,:].T)
		# plt.plot(np.average(epochs['aim']._data[:,19,:].T, axis = 1), linewidth = 3, color = 'black')
		# #plt.plot(np.average(epochs['aim']._data[:,12,:].T, axis = 1), linewidth = 3, color = 'black')
		# plt.axvline(x=50)
		# plt.suptitle('{}.png'.format(self.user))
		# plt.savefig('{}.png'.format(self.user), dpi=800)
		# plt.show()
		# plt.clf()

		evoked_aim = epochs['aim'].average()
		# evoked_aim.plot_joint()
		evoked_non_aim = epochs['non_aim'].average()
		evoked_aim.apply_baseline(baseline = (0,0))
		evoked_non_aim.apply_baseline(baseline = (0,0))


		if self.test_stats:
			mask = np.zeros(np.shape(evoked_aim.data))
			plusmask = np.ones((np.shape(evoked_aim.data)[0], 50))
			mask[:,150:200] = plusmask
			evoked_aim.data += mask*0.5e-4
			evoked_non_aim.data += mask*0.25e-4


		# d = evoked_aim.plot_joint((0.0))
		# d.savefig('./pics/' + os.path.basename(self.eeg_filename) + '.png')
		# epochs.plot(block=True, picks = [19])
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
	
	
	def get_peaks_from_evoked(self, evoked_aim, evoked_non_aim):
		"""
			Detect P300 and N1 peak ampltudes ans latencies

		    Args:
		        reg_folder (Evoked): Aim average waveforms
		        evoked_non_aim (Evoked): non-aim average waveforms

		    Returns:
		       dict: p3peaks and n1peaks (for plotting) and peaks_dict (for statiscical analysis)
		"""
		evoked = copy.deepcopy(evoked_aim)
		if self.delta:
			evoked.data -= evoked_non_aim.data
		# sys.exit()
		p3peaks =  get_peaks(evoked, ch_type='eeg', tmin = 0.25, tmax = 0.6, mode = 'pos', average_lims = self.p3average)
		n1peaks =  get_peaks(evoked, ch_type='eeg', tmin = 0.08, tmax = 0.25, mode = 'neg', average_lims = self.n1average)
		peaks_dict = {'File':os.path.basename(self.eeg_filename)}
		for p in p3peaks.keys():
			peaks_dict
			peaks_dict['p3a_{}'.format(p) ] = p3peaks[p][1]
			peaks_dict['p3i_{}'.format(p) ] = p3peaks[p][0]
			if p in ["po7", "po8","o1","oz","o2"]:
				peaks_dict['n1a_{}'.format(p) ] = n1peaks[p][1]
				peaks_dict['n1i_{}'.format(p) ] = n1peaks[p][0]
		
		return p3peaks, n1peaks, peaks_dict



	def plot_evoked_response(	self, data = {}, p3peaks = {}, n1peaks = {}, fname = str(time.time()), 
								p300_n1_aim_fill = True, peakdot = True
							):
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

		fig = plt.figure(figsize=(16,9))

		tpplt = [a for a in mne.viz.topo._iter_topography(	self.raw.info, layout = None, on_pick = None, fig = fig, layout_scale = 0.945,
															fig_facecolor='white', axis_facecolor='white', axis_spinecolor='white')]
		for ch in range(len(tpplt)):
			ax = tpplt[ch][0]
			idx = tpplt[ch][1]

			[ax.axvline(vl, color='black', linewidth=0.5, linestyle='--') for vl in [0, 0.1, 0.3]]
			ax.axhline(0, color='black', linewidth=0.5)


			for i in data.keys():
				ax.plot(data[i].times, data[i].data[ch]*1.0e6, color=self.plot_colors[i], label = i) #dashes=self.dashlist[i], 

			ax.set_title(data[data.keys()[0]].ch_names[ch])
			if self.delta and 'non_aim' in data.keys() and  'aim' in data.keys():
				ax.plot(data['aim'].times, data['aim'].data[ch]*1.0e6-data['non_aim'].data[ch]*1.0e6, dashes=self.dashlist['delta'], color=self.plot_colors['delta'], label = 'delta')


			if peakdot:
				p3p = p3peaks[self.channels[idx]]
				n1p = n1peaks[self.channels[idx]]
				ax.plot(p3p[0], p3p[1], 'o', color = 'black', zorder = 228)

			if p300_n1_aim_fill:
				fsection = [b for a, b in zip(data['aim'].times, data['aim'].data[ch]*1.0e6) if a >= p3p[0]-0.05 and a < p3p[0]+0.05]
				section = [a for a in data['aim'].times if a >=p3p[0]-0.05 and a <p3p[0]+0.05]
				ax.fill_between(section, fsection, color = 'y', alpha = 0.6)
				
				if data[data.keys()[0]].ch_names[ch] in ['oz', 'o1', 'o2']:
					ax.plot(n1p[0], n1p[1], 'o', color = 'black', zorder = 228)
					fsection = [b for a, b in zip(data['aim'].times, data['aim'].data[ch]*1.0e6) if a >= n1p[0]-0.015 and a < n1p[0]+0.015]
					section = [a for a in data['aim'].times if a >=n1p[0]-0.015 and a <n1p[0]+0.015]
					ax.fill_between(section, fsection, color = 'green', alpha = 0.6)

		legend = tpplt[0][0].legend( loc= self.legend_loc, prop={'size': 10})
		# plt.show()
		print'./pics/{}.png'.format(fname)
		plt.savefig('./pics/{}.png'.format(fname),  dpi = 400)
		plt.close()
		return

		


	def grand_average(self):
		'''
			Calculate and plot grand average waveforms 
		'''
		for k in self.evoked_aim_saved_for_grand_average.keys():
			interm = self.evoked_aim_saved_for_grand_average[k].data
			interm /= self.cc_aim_evoked[k]
			self.evoked_aim_saved_for_grand_average[k].data = interm



			interm2 = self.evoked_non_aim_saved_for_grand_average[k].data
			interm2 /= self.cc_non_aim_evoked[k]
			self.evoked_non_aim_saved_for_grand_average[k].data = interm2

			#print k	
			# self.evoked_non_aim_saved_for_grand_average[k].plot_joint((0.17, 0.3))

			if self.delta:
				self.evoked_aim_saved_for_grand_average[k].data -= interm2

		self.plot_evoked_response(	{k:self.evoked_aim_saved_for_grand_average[k] for k in self.evoked_aim_saved_for_grand_average.keys()},
									p300_n1_aim_fill = False, peakdot = False,
									fname = '_ga')

	
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
		
		settings = {'lfreq':self.l_freq, 'hfreq':self.h_freq, 
					'p3average':self.p3average, 'n1average':self.n1average, 
					'bad_files':self.bad_files, 'delta': self.delta}
		with open('{}.settings.txt'.format(filename), 'w') as file_obj:
			file_obj.write(str(settings))		
	
	def user_delta_func(self, evoked):
		evoked2 = evoked
		if self.delta:
			evoked2['aim'].data -= evoked['non_aim'].data
		return evoked2

	def user_analysis(self, user, plot = True, save_intermediate = True):
		"""
			Main function for one user

		    Args:
		        user (str/int): user folder name
		  		save_intermediate (bool): if True, save raw eeg files with events and applied filters to .fif files
		"""


		peaks = []
		self.user = user
		
		user_folder =  os.path.join(self.data_folder, str(user))
		user_data = {}
		user_peak_data = {}

		for reg in self.folders.keys():
			isfilevalid = self.read_data_files(os.path.join(user_folder, self.folders[reg]))
			if isfilevalid:
				if self.show_raw_eeg:
					self.raw.plot(block = True)
				if np.isclose(self.raw.info['highpass'], self.l_freq) and np.isclose(self.raw.info['lowpass'], self.h_freq): #default values for unfiltered data
					pass
				else:
					print 'filtering'
					self.raw_filter()
					if self.show_filtred_eeg:
						self.raw.plot(block = True, events  = self.events)
					if save_intermediate:
						print 'saving'
						self.save_intermediate(reg_folder = os.path.join(user_folder, self.folders[reg]))

				if self.reject_eog_artifacts:
					self.reject_eog_contaminated_events()
				
				evoked_aim, evoked_non_aim = self.cut_and_average(reg)
				user_data[reg] = {'aim':evoked_aim, 'non_aim':evoked_non_aim}

				p3peaks, n1peaks, peaks_dict = self.get_peaks_from_evoked(evoked_aim, evoked_non_aim)
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
			self.plot_evoked_response(	{k:self.user_delta_func(user_data[k])['aim'] for k in user_data.keys()},
									p300_n1_aim_fill = False, peakdot = False,
									
									fname = '_{}'.format(user))

		self.total_data[str(user)] = user_peak_data



if __name__ == '__main__':
	# data_folder = 'D:/Data/20!8_winter_faces/exp'B
	data_folder = r'..\exp\valid'
	#data_folder = r'./pt_1'
	
	# bad_files = 	['_data_facesnoise__play_14_19__31_10.npy',  # low freq
	# 				 '_data_faces__play_16_13_b_22_11.npy',  # low freq
	# 				 '_data_faces__play_14_33__31_10.npy',  # low freq
	# 				 '_data_faces__play_16_26__12_10.npy',  # alpha
	# 				 '_data_faces__play_15_10__01_11.npy',  # alpha
	# 				 '_data_faces__play_16_17__25_11.npy',  # alpha
	# 				 '_data_letters__play_14_05__31_10.npy',  # low freq
	# 				 '_data_noise__play_16_13__12_10.npy',  # alpha
	# 				 '_data_noise__play_16_29__25_11.npy']
	# bad_files = [a.split('.')[0] for a in bad_files]
	bad_files = []
	
	Analysis = Analysis(data_folder = data_folder, session = 'play', interface_type = 'rowcol')

	Analysis.aim_word = '@neuroscience!'

	Analysis.bad_files = bad_files
	Analysis.read_fif_if_possible = False
	Analysis.delta = True
	Analysis.l_freq = 0.1
	Analysis.fix_folder_ecg_eog = ['3']

	Analysis.show_raw_eeg = False
	Analysis.show_filtred_eeg = False

	Analysis.plot_colors.update({'Faces': 'red', 'Facesnoise':'green', 'Letters':'black', 'Noise':'blue'})
	# for user in [1,2,5,6,7,8,9,10,12,14,16,17]:
	Analysis.test_stats = False


	# Analysis.show_raw_eeg = True
	# Analysis.show_filtred_eeg = True



	# Analysis.channels = ['time', "eyes","ecg",   "a2",  "f3","fz","f4","c5","po7","c3","cz","c4","c6","cp1","cp2","p3","pz","p4","po8","o1","oz","o2", 'p5', 'p6', 'p7', 'p8', 'tp7', 'tp8', 'stim']

	# Analysis.ch_types = ['misc', 'eog', 'ecg', 'misc']+ ['eeg']*24 + ['stim']

	#Analysis.user_analysis(plot = True)
	#sys.exit()


	for user in [1,2,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22]:
		Analysis.user_analysis(user, plot = True)
	Analysis.grand_average()
	Analysis.save_peak_data(filename='peaks_av')
