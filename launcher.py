# -*- coding: utf-8 -*- 

import multiprocessing, sys, os, time
import numpy as np
import present
import eyetracker


config = './letters_table.bcicfg'
screen = 1

def stims(namespace):
	'''Create stimulation window'''
	ENV = present.ENVIRONMENT(config = config, namespace = namespace)
	ENV.Fullscreen = True 	
	ENV.build_gui(stimuli_number = 3,
					monitor = present.mymon, screen = screen)
	ENV.run_exp(stim_duration_FRAMES = 10, ISI_FRAMES = 5, 
					waitforS = True, repetitions=10)
	sys.stdout = open(str(os.getpid()) + ".out", "w") #MAGIC

def eyetrack(namespace):

	RED = eyetracker.Eyetracker(namespace = namespace, debug = True)
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