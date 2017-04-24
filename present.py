#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Name:       present.py
# Purpose:    Visual environment for eye tracking experiment
# Author: Rafael Grigoryan, kriattiffer at gmail.com
# Date: December 20, 2016
# ----------------------------------------------------------------------------

import os, sys, random, ast, math, time, itertools, datetime
from psychopy import visual, core, event, monitors
from pylsl import StreamInfo, StreamOutlet
import numpy as np

mymon = monitors.Monitor('Eizo', distance=48, width = 52.5)
mymon.setSizePix([1920, 1080])		


class ENVIRONMENT():
	""" class for visual stimulation during the experiment """
	def __init__(self, namespace, DEMO = False):
		self.namespace = namespace

		self.DEMO = DEMO
		#load configuration file
		try:
			self.config =  load_config(namespace.config)
			if 'rows' in self.config.keys():
				self.ROW_COLS = True
				self.stim_group_1 = self.config['rows'] 
				self.stim_group_2 = self.config['columns'] 

			else:
				self.ROW_COLS = False
		except Exception, e:
			print 'Problem with config file:'
			print e
			self.exit_()

		self.background = self.config['background']
		self.refresh_rate = 120

		self.Fullscreen = False
		self.plot_intervals = False
		self.window_size = self.config['window_size']
		self.LEARN = True

		self.number_of_inputs = self.config['number_of_inputs']

		self.shrink_matrix = 1
		
		self.LSL = create_lsl_outlet() # create outlet for sync
		core.wait(0.1)		

	def build_gui(self, stimuli_number = False, 
					monitor = mymon, fix_size = 1, screen  = 1):
		'''
			function for creating visual enviroment. Input: various parameters of stimuli, all optional
		'''
		if not stimuli_number:
			stimuli_number = len(self.config['positions'])
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

		self.mouse = event.Mouse(win = self.win)

		# self.win.setRecordFrameIntervals(True)

		# self.refresh_rate =  math.ceil(self.win.monitorFramePeriod**-1)
		self.frame_time = self.win.monitorFramePeriod*1000

		# read image from rescources dir and crate ImageStim objects
		stimpics = os.listdir(self.config['stimuli_dir'])
		stimpics.sort(key=lambda x: int(x.split('_')[1]))
		for pic in stimpics:
			name = int(pic.split('_')[1])
			pic = os.path.join(self.config['stimuli_dir'], pic)
			if name in self.stimuli_indices:
				stim = visual.ImageStim(self.win, image=pic,
										 name = name,
										size = self.config['size']/self.shrink_matrix, units = 'pix')
				if 'non_active' not in pic:
					active_stims.append(stim)
				else:
					non_active_stims.append(stim)
			else:
				pass
		active_stims  = active_stims
		non_active_stims  = non_active_stims

		self.textarea = visual.TextStim(self.win, text = u'', rgb = 'red', pos = (0,0.97), alignVert = 'top', alignHoriz = 'center')
		self.textarea.autoDraw = True
		self.photocell = visual.Rect(self.win, width=0.1, height=0.2, fillColor = 'black', lineWidth = 0)
		self.photocell.pos = [0.95,0.9]
		self.photocell.autoDraw = True

		# position circles over board. units are taken from the create_circle function
		poslist = self.config['positions']
		for a in active_stims:
			a.pos = [b/self.shrink_matrix for b in poslist[int(a.name)]]

		for a in non_active_stims:
			a.pos = [b/self.shrink_matrix for b in poslist[int(a.name)]]

			a.autoDraw = True
			
		self.stimlist = [non_active_stims, active_stims]

	
	def sendTrigger(self, stim):
		'''
			This function is called with callOnFlip which
			"call the function just after the flip, before psychopy does a bit 
			of housecleaning. ", according to some dude on the internets
		'''

		self.LSL.push_sample([self.stim_ind.index(stim)],  pushthrough = True) # push marker immdiately after first bit of the sequence
	
	def wait_for_event(self, key = 'LMB', wait = True, timer = 1):
		''' 
			Wait for all process flags, to ansure that the experiment dosen't start too early.
			Then wait for  keypress, if specified.
		'''
		while not hasattr(self.namespace,'EYETRACK_CALIB_SUCCESS') or \
			  not hasattr(self.namespace,'EEG_RECORDING_STARTED'):
			core.wait(0.01)
		if wait == True:
			if key in 'abcdefghijklmnopqrstuvwxyz':
				while key not in event.getKeys(): # wait for S key or LMB to start
					pass
			elif key == 'LMB':
				while not self.mouse.getPressed()[0]:
					pass
			core.wait(timer)

	def run_exp(self, stim_duration_FRAMES = 3, ISI_FRAMES = 9, 
				repetitions =  10, waitforS=True, waitforLMB = False, stimuli_number = 6):
		'''
			Run BCI-eye tracking experiment. Stimuli duration and inter-stimuli interval should be supplied as number of frames.
		'''
		cycle_ms = (stim_duration_FRAMES + ISI_FRAMES)*1000.0/self.refresh_rate
		print 'Stimuli cycle is %.2f ms' % cycle_ms
		seq = [1]*stim_duration_FRAMES + [0]*ISI_FRAMES
		self.wait_for_event(key = 's', wait = waitforS)

		if self.LEARN == True:
			# aims = [int(a)-1 for a in np.genfromtxt('aims_learn.txt')]
			aims = self.config['aims_learn']
			print 'aims: {}'.format(aims)

			# shutil.copyfile('aims_learn.txt', './experimental_data/aims_learn.txt')

			print 'Ready to start learning session, press s to begin'
			while 's' not in event.getKeys(): # wait for S key to start
				pass

		elif self.LEARN == False:
			# aims = [int(a) -1 for a in np.genfromtxt('aims_play.txt')]
			aims = self.config['aims_play']
			print 'aims: {}'.format(aims)
			# shutil.copyfile('aims_play.txt', './experimental_data/aims_play.txt')

		self.wait_for_event(key = 's', wait = waitforS)
		for letter, aim in enumerate(aims):
			self.LSL.push_sample([777]) # input of new letter
			self.superseq = self.generate_superseq(numbers = self.stimuli_indices, repetitions = repetitions)

			self.wait_for_event(key = 'LMB', wait = waitforLMB)

			self.highlight_cell(aim, displaytime = 3) # indicate aim_stimulus

			if 'escape' in event.getKeys():
				self.exit_()
			
			self.win.flip() # just in case

			for a in self.superseq:
				# print a, self.stim_ind.index(a), aim, aim in a
				# first bit of sequence and marker
				self.win.callOnFlip(self.sendTrigger, stim = a)
				self.draw_screen(a,0)

				# other bits of sequence
				for b in seq[1:]:
					self.draw_screen(a,b)
						
			core.wait(2) # wait one second after last blink so the last analysis epoch is not trimmed
			self.LSL.push_sample([888]) # end of the trial
			core.wait(0.5)			
			if self.LEARN == False:
				if not self.DEMO:
					while not self.namespace.FRESH_ANSWER:
						pass
					self.textarea.text = self.namespace.ANSWER_TEXT
					self.namespace.FRESH_ANSWER = False
					self.win.flip()
					print 'next letter'
		
		if self.LEARN == True:
			core.wait(1)
			self.LSL.push_sample([888999]) # end of learningsession
			# start online session
			print stim_duration_FRAMES
			self.LEARN = False

			# wait while classifier finishes learning
			while not hasattr(self.namespace,'START_ONLINE_SESSION'):
				pass
			print  'learning session finished, press s to continue'
			while 's' not in event.getKeys(): # wait for S key to start
				pass
			print 'Online session started'
			
			self.LSL.push_sample([999888])
			self.run_exp(stim_duration_FRAMES = stim_duration_FRAMES,
							  ISI_FRAMES = ISI_FRAMES, repetitions =  repetitions,
							  waitforS= waitforS, stimuli_number = stimuli_number)
		else:
			self.exit_()

	def highlight_cell(self, cell, displaytime = 2):
			'''
				Change cell state to active for some time.
				Arguments:
				cell	int	cell ID
				displaytime int	time to keep cell active
					default: 2
			'''
			print cell#, len(self.stimlist[1])
			self.stimlist[1][cell].autoDraw = True # indicate aim stimuli
			self.stimlist[0][cell].autoDraw = False 
			self.win.flip()
			
			core.wait(displaytime)
			
			self.stimlist[0][cell].autoDraw = True  # fade back
			self.stimlist[1][cell].autoDraw = False 
			self.win.flip()
			core.wait(1)

	def draw_screen(self, a, b):
		'''
			Redraw screen according to current frame of the stimuli sequence
			Arguments:
			a	list	stimuli sequence
			b	int	current bit of stimuli sequence

		'''
		if self.ROW_COLS: # in row-col mode, stimlist is connstructed from lists, representing rows and columns
			for rc  in a:
				self.stimlist[b][rc].autoDraw = True
				self.stimlist[b==0][rc].autoDraw = False			
		else:	
			self.stimlist[b][a].autoDraw = True
			self.stimlist[b==0][a].autoDraw = False

		if self.photocell:
			if b ==1:
				self.photocell.fillColor = 'white'
			else:
				self.photocell.fillColor = 'black'
		self.win.flip()

	def exit_(self):
		'''
			Exit and kill dependent processes
		'''
		regname = os.path.basename(self.namespace.config).split('.')[0]
		with open('./experimental_data/answers_{}_{}.txt'.format(regname, timestring()), 'w') as f:
			f.write(self.namespace.ANSWER_TEXT)
		self.LSL.push_sample([999])
		print 'exiting GUI'
		core.wait(2)
		self.win.close()

		if self.plot_intervals == True:
			from matplotlib import pyplot as plt
			plt.plot(self.win.frameIntervals[2:-3], 'o')
			plt.show()

		sys.exit()

	def generate_superseq(self, numbers =[0,1,2,3], repetitions = 10):
		''' 
			Receives IDs of stimuli, and number of repetitions, returns stimuli sequence without repeats

		'''
		
		def create_deduplicated_list(numbers, repetitions):
			seq = numbers*repetitions
			random.shuffle(seq) # generate random list
			dd_l =  [seq[a] for a in range(len(seq)) if seq[a] != seq[a-1]] #find duplicates
			dup_l =  [seq[a] for a in range(len(seq)) if seq[a] == seq[a-1]]
			for a in dup_l: # deduplicate
				p = [b for b in range(len(dd_l)) if dd_l[b] !=a and dd_l[b-1] !=a]
				dd_l.insert(p[1],a)
			# print dd_l
			return dd_l
			
		def evenly_spaced(*iterables):
		    """
			    >>> evenly_spaced(range(10), list('abc'))
			    [0, 1, 'a', 2, 3, 4, 'b', 5, 6, 7, 'c', 8, 9]
			    borrowed from http://stackoverflow.com/questions/19293481
		    """
		    return [item[1] for item in
		            sorted(itertools.chain.from_iterable(
		            zip(itertools.count(start=1.0 / (len(seq) + 1), 
		                         step=1.0 / (len(seq) + 1)), seq)
		            for seq in iterables))]
		if self.ROW_COLS:
			self.stim_ind = self.stim_group_1 + self.stim_group_2

			dd_l_gr1 = create_deduplicated_list(range(len(self.stim_group_1)), repetitions)
			dd_l_gr2 = create_deduplicated_list(range(len(self.stim_group_2)), repetitions)
			dd_l_gr2 = [a + len(self.stim_group_1) for a in dd_l_gr2]

			# dd_l = [[self.stim_ind[a], self.stim_ind[b]] for a,b in zip(dd_l_gr1, dd_l_gr2)]
			dd_l = evenly_spaced(dd_l_gr1, dd_l_gr2)
			dd_l = [[self.stim_group_1+self.stim_group_2][0][a] for a in dd_l]
		else:
			self.stim_ind = numbers
			dd_l = create_deduplicated_list(numbers, repetitions)
		return dd_l

def create_lsl_outlet(name = 'CycleStart', DeviceMac = '00:07:80:64:EB:46'):
	''' 
		Create outlet for sending markers. Returns outlet object. Use by Outlet.push_sample([MARKER_INT])
	'''
	info = StreamInfo(name,'Markers',1,0,'int32', DeviceMac)
	outlet =StreamOutlet(info)
	return outlet

def load_config(config):
	'''
		Support for configs as executable .py files (needed for resier generation of
		high-dimensional matrices).
		Valid configuration file must either be eval-able Pyrthon dictionary or 
		myst be python file, containing dictionary named 'config'
		Arguments:
		config	str	path to configuration file
	'''
	if config[-3:] == '.py':
		import imp
		cfg_py = imp.load_source('test', config)
		return cfg_py.config
	else:
		return ast.literal_eval(open(config).read())

def timestring():
	'''
		Return currrent time in string format
	'''
	return datetime.datetime.fromtimestamp(time.time()).strftime('%H_%M__%d_%m')

class emptyclass():
	"""
		Fake namespace-like class for testing purposes
	"""
	EYETRACK_CALIB_SUCCESS = True
	EEG_RECORDING_STARTED = True
	config = './configs/faces.bcicfg.py'

if __name__ == '__main__':
	print 'done imports'
	os.chdir(os.path.dirname(__file__)) 	# VLC PATH BUG ==> submit?

	ENV = ENVIRONMENT(namespace =emptyclass, DEMO = True)
	# ENV = ENVIRONMENT(namespace =emptyclass, DEMO = True, config = './configs/hexospell.bcicfg')

	# ENV = ENVIRONMENT(namespace =emptyclass, DEMO = True, config = './configs/letters_table_6x6.bcicfg')

	ENV.Fullscreen = True
	ENV.refresh_rate = 60
	ENV.shrink_matrix = 1.1
	ENV.build_gui(monitor = mymon, screen = 1)#, stimuli_number = 25)

	ENV.run_exp(stim_duration_FRAMES = 9, ISI_FRAMES = 3, repetitions = 10, waitforS = False)
