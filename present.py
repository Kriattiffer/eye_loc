#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Name:       present.py
# Purpose:    Eye tracking experiment
# Author: Rafael Grigoryan, kriattiffer at gmail.com
# Date: December 20, 2016
# ----------------------------------------------------------------------------
import os, sys, random, ast, math
from psychopy import visual, core, event, monitors
from pylsl import StreamInfo, StreamOutlet
import numpy as np

mymon = monitors.Monitor('Eizo', distance=48, width = 52.5)
mymon.setSizePix([1920, 1080])		
# mymon.setSizePix([1024, 768])		


class ENVIRONMENT():
	""" class for visual stimulation during the experiment """
	def __init__(self, namespace, DEMO = False, config = './letters_table_5x5.bcicfg'):

		self.background = '#868686'

		self.Fullscreen = False
		self.window_size = (1920, 1080)
		self.number_of_inputs = 12
		self.BEGIN_EXP = namespace.BEGIN_EXP = [False]

		try:
			self.config =  ast.literal_eval(open(config).read())
		except Exception, e:
			print 'Problem with config file:'
			print e
			self.exit_()


		if DEMO == True:
			self.LSL, self.conn = self.fake_lsl_and_conn()

		elif DEMO == False:
			self.LSL = create_lsl_outlet() # create outlet for sync with NIC
			core.wait(1)		
		
	def fake_lsl_and_conn(self):
		fakeLSL = create_lsl_outlet() # create outlet for sync with NIC - dosen't need connection
		class Fakeconn(object):
			"""dummy connection class with recv() method"""
			def __init__(self):
				pass
			def recv(self, arg):
				'''returns different strings in different enviroments 
					depending on number of bits input'''
				if arg == 1024:
					return 'answer'
				elif arg == 2048:
					return 'startonlinesession'
		return fakeLSL, Fakeconn()
				


	def build_gui(self, stimuli_number = 6, 
					monitor = mymon, fix_size = 1, screen  = 1):
		''' function for creating visual enviroment. Input: various parameters of stimuli, all optional'''
		
		self.stimuli_indices = range(stimuli_number)
		
		active_stims = []
		non_active_stims = []
		# Create window
		self.win = visual.Window(fullscr = self.Fullscreen, 
							rgb = self.background,
							size = self.window_size,	
							monitor = monitor,
							# waitBlanking = True,
							# useFBO=True,
							screen = screen # 1- right, 0 - left
							)
		self.win.setRecordFrameIntervals(True)

		self.refresh_rate =  math.ceil(self.win.monitorFramePeriod**-1)
		self.frame_time = self.win.monitorFramePeriod*1000

		# read image from rescources dir and crate ImageStim objects
		stimpics = os.listdir(self.config['stimuli_dir'])
		for pic in stimpics:
			name = int(pic.split('_')[1])
			pic = os.path.join(self.config['stimuli_dir'], pic)
			if name in self.stimuli_indices:
				stim = visual.ImageStim(self.win, image=pic,
										 name = name,
										size = self.config['size'], units = 'pix')
				if 'non_active' not in pic:
					active_stims.append(stim)
				else:
					non_active_stims.append(stim)
			else:
				pass
		active_stims  = active_stims
		non_active_stims  = non_active_stims

		# position circles over board. units are taken from the create_circle function
		poslist = self.config['positions']
		for a in active_stims:
			a.pos = poslist[int(a.name)]

		for a in non_active_stims:
			a.pos = poslist[int(a.name)]
			a.autoDraw = True
			
		self.stimlist = [non_active_stims, active_stims]


	def sendTrigger(self, stim):
		'''This function is called with callOnFlip which
		 "call the function just after the flip, before psychopy does a bit of housecleaning. ", 
		 according to some dude on the internets'''
		self.LSL.push_sample([self.stimlist[1][stim].name], pushthrough = True) # push marker immdiately after first bit of the sequence
	
	def wait_for_event(self, key, wait = True):
		if wait == True or self.BEGIN_EXP == [False]:
			while key not in event.getKeys(): # wait for S key to start
				pass

	def run_exp(self, stim_duration_FRAMES = 3, ISI_FRAMES = 9, repetitions =  10, waitforS=True, stimuli_number = 6):
		'Eye tracking expreiment. Stimuli duration and interstimuli interval should be supplied as number of frames.'
		cycle_ms = (stim_duration_FRAMES + ISI_FRAMES)*1000.0/self.refresh_rate
		print 'Stimuli cycle is %.2f ms' % cycle_ms
		seq = [1]*stim_duration_FRAMES + [0]*ISI_FRAMES

		aims = [int(a) -1 for a in np.genfromtxt('aims_play.txt')]

		self.wait_for_event(key = 's', wait = waitforS)

		for letter, aim in enumerate(aims):
			self.LSL.push_sample([777]) # input of new letter
			superseq = generate_superseq(numbers = self.stimuli_indices, repetitions = repetitions)

			# aim_stimuli
			self.stimlist[1][aim].autoDraw = True # indicate aim stimuli
			self.stimlist[0][aim].autoDraw = False 
			self.win.flip()
			
			core.wait(2)
			
			self.stimlist[0][aim].autoDraw = True  # fade back
			self.stimlist[1][aim].autoDraw = False 
			self.win.flip()
			core.wait(1)

			if 'escape' in event.getKeys():
				self.exit_()

			self.win.flip() # just in case

			for a in superseq:
				# first bit of sequence and marker
				self.win.callOnFlip(self.sendTrigger, stim = a)
				self.stimlist[1][a].autoDraw = True
				self.stimlist[0][a].autoDraw = False
				self.win.flip()

				# other bits of sequence
				for b in seq[1:]:
					self.stimlist[b][a].autoDraw = True
					self.stimlist[b==0][a].autoDraw = False
					self.win.flip()

			
			core.wait(1.5) # wait one second after last blink
			self.LSL.push_sample([888]) # end of the trial
			core.wait(0.5)
			print 'next letter'
		
		else:
			self.exit_()

	def exit_(self):
		''' exit and kill dependent processes'''
		self.LSL.push_sample([999])
		core.wait(1)

		# from matplotlib import pyplot as plt
		# plt.plot(self.win.frameIntervals[2:-3], 'o')
		# plt.show()

		sys.exit()

def generate_superseq(numbers =[0,1,2,3], repetitions = 10):
	''' receives IDs of stimuli, and number of repetitions, returns stimuli sequence without repeats'''
	seq = numbers*repetitions
	random.shuffle(seq) # generate random list
	dd_l =  [seq[a] for a in range(len(seq)) if seq[a] != seq[a-1]] #find duplicates
	dup_l =  [seq[a] for a in range(len(seq)) if seq[a] == seq[a-1]]
	for a in dup_l: # deduplicate
		p = [b for b in range(len(dd_l)) if dd_l[b] !=a and dd_l[b-1] !=a]
		dd_l.insert(p[1],a)
	return dd_l

def create_lsl_outlet(name = 'CycleStart', DeviceMac = '00:07:80:64:EB:46'):
	''' Create outlet for Enobio. Requires name of the stream (same as in NIC) and maybe MAC adress of the device. Returns outlet object. Use by Outlet.push_sample([MARKER_INT])'''
	info = StreamInfo(name,'Markers',1,0,'int32', DeviceMac)
	outlet =StreamOutlet(info)
	return outlet

def view():
	'''function for multiprocessing'''
	ENV = ENVIRONMENT()
	ENV.build_gui(monitor = mymon)
	ENV.run_exp(base_mseq)

if __name__ == '__main__':
	print 'done imports'
	os.chdir(os.path.dirname(__file__)) 	# VLC PATH BUG ==> submit?

	ENV = ENVIRONMENT(namespace = type('test', (object,), {})(), DEMO = True, 
						config = './letters_table_5x5.bcicfg')
	# ENV.Fullscreen = True
	# ENV.photocell = True
	ENV.refresh_rate = 60
	ENV.build_gui(monitor = mymon, screen = 0, stimuli_number = 25)
	ENV.run_exp(stim_duration_FRAMES = 2, ISI_FRAMES = 2, waitforS = True)
