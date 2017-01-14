#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Name:       launcher.py
# Purpose:    Launch and control of multiple processes
# Author: Rafael Grigoryan, kriattiffer at gmail.com
# Date: December 20, 2016
# ----------------------------------------------------------------------------

import multiprocessing, sys, os, time, argparse
import present, eyetracker


config = './letters_table_5x5.bcicfg'
screen = 1

def stims(namespace, ISI_FRAMES = 4, stim_duration_FRAMES = 4, repeats = 4):
	'''Create stimulation window'''
	ENV = present.ENVIRONMENT(config = config, namespace = namespace)
	
	ENV.Fullscreen = True 	
	ENV.refresh_rate = 120
	ENV.shrink_matrix = 1
	ENV.plot_intervals = True
	ENV.BEGIN_EXP = [True]

	ENV.ROW_COLS = True

	ENV.build_gui(monitor = present.mymon, 
				  screen = screen, stimuli_number = 25)
	ENV.run_exp(stim_duration_FRAMES = stim_duration_FRAMES, ISI_FRAMES = ISI_FRAMES, 
				repetitions = repeats, waitforS = False)

	sys.stdout = open(str(os.getpid()) + ".out", "w") #MAGIC

def eyetrack(namespace):

	RED = eyetracker.Eyetracker(namespace = namespace, debug = True,
								number_of_points = 9)
	RED.main()
	sys.stdout = open(str(os.getpid()) + ".out", "w") #MAGIC

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', action='store', dest='isi', type=int, default = 4)
	parser.add_argument('-f', action='store', dest='sdf', type=int, default = 4)
	parser.add_argument('-r', action='store', dest='rpt', type=int, default = 4)
	args = vars(parser.parse_args())

	mgr = multiprocessing.Manager()
	namespace = mgr.Namespace()

	pgui = multiprocessing.Process(target=stims, args = (namespace,), kwargs = {'ISI_FRAMES':args['isi'], 'repeats':args['rpt'], 
																				'stim_duration_FRAMES':args['sdf'] })
	peye = multiprocessing.Process(target=eyetrack, args = (namespace,))
	print 'startig GUI...'
	pgui.start()
	print 'startig eyetracking server...'
	peye.start()
	peye.join()
	# pgui.join()