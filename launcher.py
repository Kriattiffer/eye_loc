#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Name:       launcher.py
# Purpose:    Launch and control of multiple processes
# Author: Rafael Grigoryan, kriattiffer at gmail.com
# Date: December 20, 2016
# ----------------------------------------------------------------------------

import multiprocessing, sys, os, time
import numpy as np
import present
import eyetracker


config = './letters_table_5x5.bcicfg'
screen = 1

def stims(namespace):
	'''Create stimulation window'''
	ENV = present.ENVIRONMENT(config = config, namespace = namespace)
	
	ENV.Fullscreen = True 	
	ENV.refresh_rate = 120
	ENV.shrink_matrix = 2
	ENV.plot_intervals = True

	ENV.build_gui(monitor = present.mymon, 
				  screen = screen, stimuli_number = 25)
	ENV.run_exp(stim_duration_FRAMES = 6, ISI_FRAMES = 6, 
				repetitions = 100, waitforS = False)

	sys.stdout = open(str(os.getpid()) + ".out", "w") #MAGIC

def eyetrack(namespace):

	RED = eyetracker.Eyetracker(namespace = namespace, debug = True,
								number_of_points = 9)
	RED.main()
	sys.stdout = open(str(os.getpid()) + ".out", "w") #MAGIC

if __name__ == '__main__':

	mgr = multiprocessing.Manager()
	namespace = mgr.Namespace()
	BEGIN_EXP = multiprocessing.Event()
	newstdin = os.dup(sys.stdin.fileno())


	pgui = multiprocessing.Process(target=stims, args = (namespace,))
	peye = multiprocessing.Process(target=eyetrack, args = (namespace,))
	print 'startig GUI...'
	pgui.start()
	print 'startig eyetracking server...'
	peye.start()
	
	peye.join()
	# pgui.join()