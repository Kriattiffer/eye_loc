# -*- coding: utf-8 -*- 

import multiprocessing, sys, os, time
from tempfile import mkdtemp
import numpy as np
import present
import record
import classify


config = './letters_table.bcicfg'
screen = 0

mapnames = {'eeg':'eegdata.mmap', 'markers':'markers.mmap'}
top_exp_length = 60
number_of_channels = 8
classifier_channels = []
savedclass = False
savedclass = 'sample8chann.cls'
# savedclass = 'sample20chann.cls' # 20 CHANN

def view():
	'''Create stimulation window'''
	ENV = present.ENVIRONMENT(config = config)
	ENV.Fullscreen = True 	

	ENV.build_gui(stimuli_number = 3,
					monitor = present.mymon, screen = screen)
	if savedclass:
		ENV.LEARN = False
		print 'Using saved classifier from %s' % savedclass
	else:
		print 'Buildindg new classifier'
	ENV.run_P300_exp(stim_duration_FRAMES = 10, ISI_FRAMES = 5, 
					waitforS = False, repetitions=10)
	sys.stdout = open(str(os.getpid()) + ".out", "w") #MAGIC

def rec():
	''' Create stream class and start recording and plotting'''
	STRM = record.EEG_STREAM(mapnames = mapnames, plot_fft = False, plot_to_second_screen = True,
								top_exp_length = top_exp_length, number_of_channels  = number_of_channels)
	STRM.plot_and_record()
	sys.stdout = open(str(os.getpid()) + ".out", "w")

def class_():
	'''Create classifer class and wait for markers from present.py'''
	CLSF = classify.Classifier(mapnames = mapnames, online = True,
									top_exp_length = top_exp_length, 
									number_of_channels = number_of_channels,
									classifier_channels = classifier_channels, 
									saved_classifier = savedclass,
									config = config)
	CLSF.mainloop()
	sys.stdout = open(str(os.getpid()) + ".out", "w")


if __name__ == '__main__':
	pgui = multiprocessing.Process(target=view)
	prec = multiprocessing.Process(target=rec)
	pclass = multiprocessing.Process(target=class_)
	
	print 'startig backend...'
	prec.start()
	print 'startig GUI...'
	pgui.start()
	time.sleep(4)
	print 'startig classifier...'
	pclass.start()