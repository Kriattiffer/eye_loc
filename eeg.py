#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Name:       record.py
# Purpose:    EEG recording during eye tracking experiment
# Author: Rafael Grigoryan, kriattiffer at gmail.com
# Date: January 17, 2017
# ----------------------------------------------------------------------------

from pylsl import StreamInlet, resolve_stream
from scipy import signal
import numpy as np
import sys, os, warnings, time

class EEG_STREAM(object):
	""" 
		Class for EEG\markers streaming and recording.
	"""
	def __init__(self, namespace,  mapnames = {'eeg':'eegdata.mmap', 'markers':'markers.mmap', 'photocell':'photocell.mmap'}, device = 'NVX52',
				top_exp_length = 60,
	 			StreamEeg = True, StreamMarkers = True):
		''' 
			create objects for later use
		'''
		self.namespace = namespace
		self.device = device
		self.StreamEeg, self.StreamMarkers = StreamEeg, StreamMarkers
		self.stop = False  # set EEG_STREAM.stop to 1 to flush arrays to disc. This variable is also used to choose the exact time to stop record.
		self.learning_end = False
		self.save_separate_learn_and_play = True

		self.ie, self.im, self.ip =  self.create_streams()

		
		self.EEG_ARRAY = self.create_array(mmapname=mapnames['eeg'], top_exp_length = top_exp_length, number_of_channels = self.ie.channel_count)
		self.MARKER_ARRAY = self.create_array(mmapname=mapnames['markers'], top_exp_length = 1, number_of_channels = 1)
		self.PHOTOCELL_ARRAY = self.create_array(mmapname=mapnames['photocell'], top_exp_length = 1, number_of_channels = 1)
		
		self.namespace.DATA_ARRAYS_PREALLOCATED = True

		self.line_counter = 0
		self.line_counter_mark = 0
		self.line_counter_evt = 0


	def create_streams(self, recursion_meter = 0, max_recursion_depth = 3):
		'''
			Opens two LSL streams: one for EEG, another for markers, If error, tries to reconnect several times
		'''
		if self.device == 'Enobio':
			stream_type_eeg = 'EEG'
			stream_name_markers = 'CycleStart'
		elif self.device == 'NVX52':
			stream_type_eeg = 'Data'
			stream_name_markers = 'CycleStart'
			stream_name_events = 'NVX52_Events'
			STREAM_NVX_PHOTOCELL = True

		else:
			print 'I don\'t know device {}!s\n\n\nWill now exit!'.format(self.device)
			sys.exit()

		
		if recursion_meter == 0:
			recursion_meter +=1
		elif 0<recursion_meter <max_recursion_depth:
			print 'Trying to reconnect for the %i time \n' % (recursion_meter+1)
			recursion_meter +=1
		else:
			print 'exiting'
			sys.exit()
		inlet_eeg = []; inlet_markers = []
		
		if self.StreamEeg == True:

			print ("Connecting to %s stream..." % self.device)
			if stream_type_eeg in [stream.type() for stream in resolve_stream()]:
				streams_eeg = resolve_stream('type', stream_type_eeg)
				inlet_eeg = StreamInlet(streams_eeg[0])   
				try:
					inlet_eeg
					self.sampling_rate = streams_eeg[0].nominal_srate()
					# print inlet_eeg.info().as_xml()
					print '...done \n'
				except NameError:
					print ("Error: Cannot conect to %s stream\n" % self.device)
					sys.exit()
			else:
				print 'Error: %s stream is not available\n' % self.device
				sys.exit()
		else:
			inlet_eeg = []

		if STREAM_NVX_PHOTOCELL == True:
			print ("Connecting to %s photocell stream..." % self.device)
			if stream_name_events in [stream.name() for stream in resolve_stream()]:
				streams_eeg = resolve_stream('name', stream_name_events)
				inlet_photocell = StreamInlet(streams_eeg[0])   
				try:
					inlet_photocell
					# print inlet_eeg.info().as_xml()
					print '...done \n'
				except NameError:
					print ("Error: Cannot conect to %s photocell stream\n" % self.device)
					sys.exit()
			else:
				print 'Error: %s stream is not available\n' % self.device
				sys.exit()
		else:
			inlet_photocell = []

		if self.StreamMarkers == True:
			print ("Connecting to Psychopy stream...")
			if stream_name_markers in [stream.name() for stream in resolve_stream()]:
				sterams_markers = resolve_stream('name', stream_name_markers)
				inlet_markers = StreamInlet(sterams_markers[0])   
				try:
					inlet_markers
					print '...done \n'
				except NameError:
					print ("Error: Cannot conect to Psychopy stream\n")
					sys.exit()
			else:
				print 'Error: Psychopy stream is not available\n'
				return self.create_streams(stream_type_eeg, stream_name_markers, StreamEeg, StreamMarkers, recursion_meter)
		else:
			inlet_markers = []



		return inlet_eeg, inlet_markers, inlet_photocell
	
	def create_array(self, mmapname, dtype = 'float', top_exp_length = 60, number_of_channels=8):
		''' 
			Creates very long array of Nans, which will be filled by EEG. length is determined by maximum length of the experiment in minutes
			The array is mapped to disc for later use from classiffier process
		'''
		record_length = self.sampling_rate*60*top_exp_length*1.2
		array_shape = (int(record_length), number_of_channels+1)
		print 'Creating array with dimensions %s...' %str(array_shape) 
		print array_shape
		a = np.memmap(mmapname, dtype=dtype, mode='w+', shape=array_shape)
		# a = np.zeros(array_shape, dtype = 'float')
		a[:,0:self.ie.channel_count+1] = np.NAN
		print '... done'
		return a

	def fill_array(self, data_array, line_counter, data_chunk, timestamp_chunk):
		'''
			Recieves preallocated array of NaNs, piece of data, piece of offsets and number of line,
			inserts everything into array. Works both with EEG and with markers 
		'''
		
		if timestamp_chunk:
			data_array[line_counter:line_counter+len(timestamp_chunk), 0] = timestamp_chunk
		if data_chunk:
			data_array[line_counter:line_counter+len(data_chunk),1:] = data_chunk
	
	def save_data(self, sessiontype = '', startingpoint = False):
		print '\nsaving experiment data from %s session...\n' %sessiontype
		data_arrays = prepare_arrays(startingpoint = startingpoint,
											device = self.device, 
											PHOTOCELL_ARRAY = self.PHOTOCELL_ARRAY, 
											MARKER_ARRAY = self.MARKER_ARRAY,
											EEG_ARRAY = self.EEG_ARRAY	)
		filenames = ['_data%s.txt'%sessiontype, '_markers%s.txt'%sessiontype, '_events%s.txt'%sessiontype]
		for n, array in enumerate(data_arrays):
			np.savetxt(filenames[n], array, fmt= '%.4f')	

	def record(self):
		''' 
			Main cycle for data recording. Pulls markers and eeg from lsl inlets, 
			fills preallocated arrays with data. Records data on exit.
		''' 
		self.namespace.EEG_RECORDING_STARTED = True
		while 1: #self.stop != True:	
			# pull chunks if Steam_eeg and stream_markess are True
			try:
				EEG, timestamp_eeg = self.ie.pull_chunk()
			except:
				EEG, timestamp_eeg = [], []

			try:
				marker, timestamp_mark = self.im.pull_chunk()
			except :
				marker, timestamp_mark = [],[]

			try:
				event, timestamp_event = self.ip.pull_chunk()
			except :
				event, timestamp_event = [],[]
			
			if timestamp_event:
				self.fill_array(self.PHOTOCELL_ARRAY, self.line_counter_evt, event, timestamp_event)		
				self.line_counter_evt += len(timestamp_event)



			if timestamp_mark:			
				self.fill_array(self.MARKER_ARRAY, self.line_counter_mark, marker, timestamp_mark)	
				self.line_counter_mark += len(timestamp_mark)
				if marker == [[999]]:
					self.stop = self.line_counter + 500 # set last 
				if marker == [[888999]]:
					if self.save_separate_learn_and_play == True:
						self.save_data(sessiontype = '_learn')  #save data as usual
						print '\n...Learning data saved.\n'

						self.learning_end = timestamp_mark[0]
						print self.learning_end
						print self.learning_end
						print self.EEG_ARRAY[:,0]>self.learning_end
				# 	pass


			if timestamp_eeg:
				self.fill_array(self.EEG_ARRAY, self.line_counter, EEG, timestamp_eeg)
				self.line_counter += len(timestamp_eeg)
				if self.stop != False : # save last EEG chunk before exit
					if self.line_counter >= self.stop:
						self.save_data(sessiontype = '_play', startingpoint = self.learning_end)
						print '\n...data saved.\n Goodbye.\n'
						sys.exit()

def prepare_arrays(PHOTOCELL_ARRAY, MARKER_ARRAY, EEG_ARRAY,
 					device,
 					startingpoint = False ):

	def merge_arrays(eventdata, markerdata):
		cc = 0
		markerdata = markerdata.copy()
		for n, a in enumerate(markerdata): # vectorize?
			if a[1] not in [777, 888, 888999, 999888, 999]: # exclude technical markers that are not synchronized with win.flip()
				markerdata[n,0] = eventdata[cc,0]
				cc+=1
			else:
				# markerdata[n,0] = -1
				pass
		return markerdata

	if startingpoint:
		print np.logical_and((MARKER_ARRAY[:,0]>startingpoint), 
															(np.isnan(MARKER_ARRAY[:,1]) != True))
		with warnings.catch_warnings(): # >< operators generate warnings on arrays with NaNs, like our EEG array
			warnings.simplefilter("ignore")
			eegdata = EEG_ARRAY[np.logical_and((EEG_ARRAY[:,0]>startingpoint), 
																	(np.isnan(EEG_ARRAY[:,1]) != True)),:]  # delete all unused lines from data matrix AND use data only after learning has ended
			markerdata = MARKER_ARRAY[np.logical_and((MARKER_ARRAY[:,0]>startingpoint), 
															(np.isnan(MARKER_ARRAY[:,1]) != True)),:]
	else:
		eegdata = EEG_ARRAY[np.isnan(EEG_ARRAY[:,1]) != True,:]  # delete all unused lines from data matrix
		markerdata = MARKER_ARRAY[np.isnan(MARKER_ARRAY[:,1]) != True,:]
		
	if device == 'NVX52':
		eventdata = PHOTOCELL_ARRAY[np.isnan(PHOTOCELL_ARRAY[:,1]) != True,:]
		markerdata = merge_arrays(eventdata, markerdata)
	
		return eegdata, markerdata, eventdata
	else:
		return eegdata, markerdata

def butter_filt(data, cutoff_array, fs = 500, order=4):
    nyq = 0.5 * fs
    normal_cutoff = [a /nyq for a in cutoff_array]
    b, a = signal.butter(order, normal_cutoff, btype = 'bandpass', analog=False)
    data = signal.filtfilt(b, a, data, axis = 0)
    return data

os.chdir(os.path.dirname(__file__)) 	# VLC PATH BUG ==> submit?

if __name__ == '__main__':
	Stream = EEG_STREAM(StreamMarkers = True, 
						device = 'NVX52'
						)
	Stream.record()