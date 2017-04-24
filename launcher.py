#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Name:       launcher.py
# Purpose:    Launch and control of multiple processes
# Author: Rafael Grigoryan, kriattiffer at gmail.com
# Date: December 20, 2016
# ----------------------------------------------------------------------------

import multiprocessing, sys, os, time, argparse
import present, eyetracker, eeg, classify

# config = './configs/letters_table_8x9.bcicfg.py'
data_folder = './experimental_data'
screen = 1
refresh_rate = 60
top_exp_length = 60
device = 'NVX52'
mapnames = {'eeg':'./eegdata.mmap', 
			'markers':'./markers.mmap',
			'photocell': './photocell.mmap'}
# classifier_channels	 = range(8)
# classifier_channels = [a-1 for a in [21, 19, 17, 16, 15, 14, 13, 10]]
classifier_channels = [a for a in range(8)]


savedclass = False
# savedclass = './experimental_data/classifier_1492183246771.cls'

def stims(namespace, ISI_FRAMES = 4, stim_duration_FRAMES = 4, repeats = 10):
	'''Create stimulation window'''
	ENV = present.ENVIRONMENT(namespace = namespace)
	
	ENV.Fullscreen = True 	
	ENV.refresh_rate = refresh_rate
	ENV.shrink_matrix = 1.32
	ENV.build_gui(monitor = present.mymon, 
				  screen = screen, stimuli_number = False)
	if savedclass:
		ENV.LEARN = False
		print 'Using saved classifier from %s' % savedclass
	else:
		print 'Buildindg new classifier'

	ENV.run_exp(stim_duration_FRAMES = stim_duration_FRAMES, ISI_FRAMES = ISI_FRAMES, 
				repetitions = repeats, waitforS = False)
	sys.stdout = open(str(os.getpid()) + ".out", "w") #MAGIC

def eyetrack(namespace, fake_et = False):
	'''Manage Red eyetracker'''
	if fake_et:
		namespace.EYETRACK_CALIB_SUCCESS = True
		return
	else:
		RED = eyetracker.Eyetracker(namespace = namespace, debug = True,
									number_of_points = 9, screen = screen)
		RED.main()
		sys.stdout = open(str(os.getpid()) + ".out", "w") #MAGIC

def rec(namespace):
	''' Create stream class and start recording'''
	STRM = eeg.EEG_STREAM(namespace = namespace, device = device, mapnames = mapnames, top_exp_length = top_exp_length)
	STRM.record()
	sys.stdout = open(str(os.getpid()) + ".out", "w")

def class_(namespace):
	'''Create classifer class and wait for markers from present.py'''
	CLSF = classify.Classifier(namespace = namespace, 
									mapnames = mapnames, online = True,
									top_exp_length = top_exp_length, 
									classifier_channels = classifier_channels, 
									saved_classifier = savedclass)
	CLSF.mainloop()
	sys.stdout = open(str(os.getpid()) + ".out", "w")


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', action='store', dest='isi', type=int, default = 9)
	parser.add_argument('-d', action='store', dest='sdf', type=int, default = 3)
	parser.add_argument('-r', action='store', dest='rpt', type=int, default = 10)
	parser.add_argument('--noeyetrack', action='store_true', dest='fake_et')

	parser.add_argument('--config', action='store', dest='config', type=str, 
						default = './configs/letters.bcicfg.py')
						# default = './configs/hexospell.bcicfg')

	args = vars(parser.parse_args())

	mgr = multiprocessing.Manager()
	namespace = mgr.Namespace()
	namespace.data_folder = data_folder
	namespace.config = args['config']
	print os.path.basename(namespace.config).split('.')[0]


	pgui = multiprocessing.Process(target=stims, args = (namespace,), kwargs = {'ISI_FRAMES':args['isi'], 'repeats':args['rpt'], 
																				'stim_duration_FRAMES':args['sdf'] })
	peye = multiprocessing.Process(target=eyetrack, args = (namespace,), kwargs = {'fake_et':args['fake_et'] })
	prec = multiprocessing.Process(target=rec, args = (namespace,))
	pcls = multiprocessing.Process(target=class_, args = (namespace,)) 


	print 'startig GUI...'
	pgui.start()
	print 'startig eyetracking server...'
	peye.start()
	print 'startig EEG recording...'
	prec.start()
	print 'startig classifier'
	pcls.start()

	prec.join()
	peye.join()
	pgui.join()
	pcls.join()