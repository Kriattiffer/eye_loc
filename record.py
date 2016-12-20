# -*- coding: utf-8 -*- 

from pylsl import StreamInlet, resolve_stream
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
import sys, os, warnings

class FFT_PLOT():
	"""class for real-time plotting of FFT fot every channel"""
	def __init__(self, sample_length, max_fft_freq=140, plot_to_second_screen = True):
		''' here we createth plot object, to update it in loop. for performance we useth here powerful magic called 'blitting',  
			that helpeth us to redraw only data, while keenping backround and axes intact. How it worketh remains unknown >>> look into it later!
			max_fft_freq is cutoff frequency for fft_plot.'''

		channels = ['1','2','3','4','5','6','7','8'] # need to set from config file		
		self.sample_length = sample_length # number of samples to analyze
		T = 500.0				#sampling rate, Hz
		self.Bin_resolution = T/sample_length
		self.max_fft_freq = max_fft_freq
		###preallocate array for fourier vectors
		self.fouriers = []
		self.sft = 'list'
		### create_plot ###
		self.fig,self.axes = plt.subplots(nrows =3, ncols = 3)
		self.axes = self.axes.flatten()[:-1]
		if plot_to_second_screen == True:
			plt.get_current_fig_manager().window.wm_geometry("-1920+0") # move FFT window tio second screen. Frame redraw in pesent.py starts to suck ==> possible problem with video card
		self.fig.show()
		self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axes]
		x = np.arange(0, self.max_fft_freq, self.Bin_resolution)
		y = np.arange(0,1,1.0/self.max_fft_freq*self.Bin_resolution)
		y = y[x>=0]
		print np.shape(x), np.shape(y)
		self.lines = [ax.plot(x,y)[0] for ax in self.axes] #np.arange(0,1,1/119.0)	
		[self.axes[i].set_title(channels[i], fontweight= 'bold',) for i in range(len(self.axes))]
		self.fig.canvas.draw()

	def update_fft(self, FFT, averaging_bin=10):
		''' receives FFT vector, trimmes it to several points (whth 500 hz refresh rate it is 60 hz maximum), redraws plot.'''
		if not plt.fignum_exists(1): # dosent't try to update closed figure # careful with additional figures!
			return

		FFT = np.abs(FFT[:self.max_fft_freq/self.Bin_resolution,:])
		# print type(self.fouriers)
		if self.sft == 'list':
			if len(self.fouriers) < averaging_bin:
				self.fouriers.append(FFT)
			else:
				self.fouriers = np.array(self.fouriers)
				self.sft = type(self.fouriers) 
		else:
			# print np.shape(self.fouriers[0:-1])
			self.fouriers[0:-1] = self.fouriers[1:]
			self.fouriers[-1] = FFT
			FFT = np.average(self.fouriers, axis = 0)

		for line, ax, background, channel  in zip(self.lines, self.axes, self.backgrounds, range(len(self.axes))):
			self.fig.canvas.restore_region(background)			
			line.set_ydata(FFT[:, channel])
			# print np.argmax(FFT[:, channel])
			ax.draw_artist(ax.patch)
			ax.draw_artist(line)
			self.fig.canvas.blit(ax.bbox)
			# self.fig.canvas.update()
			# self.fig.canvas.flush_events()
		self.fig.canvas.start_event_loop(0.001) #0.1 ms seems enough


class EEG_STREAM(object):
	""" class for EEG\markers streaming, plotting and recording. """
	def __init__(self,  mapnames = {'eeg':'eegdata.mmap', 'markers':'markers.mmap'}, 
				sample_length = 1000, top_exp_length = 60, number_of_channels = 8,
	 			StreamEeg = True, StreamMarkers = True, plot_fft = True, plot_to_second_screen = True):
		''' create objects for later use'''
		self.StreamEeg, self.StreamMarkers = StreamEeg, StreamMarkers
		self.plot_fft = plot_fft
		self.stop = False  # set EEG_STREAM.stop to 1 to flush arrays to disc. This variable is also used to choose the exact time to stop record.
		self.learning_end = False
		self.save_separate_learn_and_play = True

		self.ie, self.im =  self.create_streams()
		
		self.EEG_ARRAY = self.create_array(mmapname=mapnames['eeg'], top_exp_length = top_exp_length, number_of_channels = number_of_channels+1)
		self.MARKER_ARRAY = self.create_array(mmapname=mapnames['markers'], top_exp_length = 1, number_of_channels = 2)
		self.line_counter = 0
		self.line_counter_mark = 0
		self.sample_length = sample_length
		if self.plot_fft == True:
			self.plot = FFT_PLOT(max_fft_freq=60, sample_length = self.sample_length, plot_to_second_screen = plot_to_second_screen)

	def create_streams(self, stream_type_eeg = 'EEG', stream_name_markers = 'CycleStart', recursion_meter = 0, max_recursion_depth = 3):
		''' Opens two LSL streams: one for EEG, another for markers, If error, tries to reconnect several times'''
		if recursion_meter == 0:
			recursion_meter +=1
		elif 0<recursion_meter <max_recursion_depth:
			print 'Trying to reconnect for the %i time \n' % (recursion_meter+1)
			recursion_meter +=1
		else:
			print 'exiting'
			plt.close()
			sys.exit()
		inlet_eeg = []; inlet_markers = []
		
		if self.StreamEeg == True:
			print ("Connecting to NIC stream...")
			if stream_type_eeg in [stream.type() for stream in resolve_stream()]:
				streams_eeg = resolve_stream('type', 'EEG')
				inlet_eeg = StreamInlet(streams_eeg[0])   
				try:
					inlet_eeg
					# print inlet_eeg.info().as_xml()
					print '...done \n'
				except NameError:
					print ("Error: Cannot conect to NIC stream\n")
					plt.close()
					sys.exit()
			else:
				print 'Error: NIC stream is not available\n'
				plt.close()
				sys.exit()
		else:
			inlet_eeg = []

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
					plt.close()
					sys.exit()
			else:
				print 'Error: Psychopy stream is not available\n'
				return self.create_streams(stream_type_eeg, stream_name_markers, StreamEeg, StreamMarkers, recursion_meter)
		else:
			inlet_markers = []
		return inlet_eeg, inlet_markers
	
	def create_array(self, mmapname, top_exp_length = 60, number_of_channels  = 9):
		'''Creates very long array of Nans, which will be filled by EEG. length is determined by maximum length of the experiment in minutes
			The array is mapped to disc for later use from classiffier process'''
		record_length = 500*60*top_exp_length*1.2
		array_shape = (record_length, number_of_channels)
		print 'Creating array with dimensions %s...' %str(array_shape) 
		a = np.memmap(mmapname, dtype='float', mode='w+', shape=(array_shape))
		# a = np.zeros(array_shape, dtype = 'float')
		a[:,0:number_of_channels] = np.NAN
		print '... done'
		return a

	def fill_array(self, data_array, line_counter, data_chunk, timestamp_chunk):
		'''Recieves preallocated array of NaNs, piece of data, piece of offsets and number of line, inserts everything into array. Works both with EEG and with markers '''
		length = len(timestamp_chunk)
		data_array[line_counter:line_counter+length, 0] = timestamp_chunk
		data_array[line_counter:line_counter+length,1:] = data_chunk
	
	def save_data(self, sessiontype = '', startingpoint = False):

		print '\nsaving experiment data from %s session...\n' %sessiontype

		if startingpoint:
			print np.logical_and((self.MARKER_ARRAY[:,0]>startingpoint), 
																(np.isnan(self.MARKER_ARRAY[:,1]) != True))
			with warnings.catch_warnings(): # >< operators generate warnings on arrays with NaNs, like our EEG array
				warnings.simplefilter("ignore")
				eegdata = self.EEG_ARRAY[np.logical_and((self.EEG_ARRAY[:,0]>startingpoint), 
																		(np.isnan(self.EEG_ARRAY[:,1]) != True)),:]  # delete all unused lines from data matrix AND use data only after learning has ended
				markerdata = self.MARKER_ARRAY[np.logical_and((self.MARKER_ARRAY[:,0]>startingpoint), 
																(np.isnan(self.MARKER_ARRAY[:,1]) != True)),:]
		else: #LMAO
			eegdata = self.EEG_ARRAY[np.isnan(self.EEG_ARRAY[:,1]) != True,:]  # delete all unused lines from data matrix
			markerdata = self.MARKER_ARRAY[np.isnan(self.MARKER_ARRAY[:,1]) != True,:]
		
		pass # join EEG and data steams
		# add column with markers
		# add colum with 1-0


		np.savetxt('_data%s.txt'%sessiontype, eegdata, fmt= '%.4f')
		np.savetxt('_markers%s.txt'%sessiontype, markerdata, fmt= '%.4f')


	def plot_and_record(self):
		''' Main cycle for recording and plotting FFT. Pulls markers and eeg from lsl inlets, 
		fills preallocated arrays with data. After certain offset calculates FFT and updates plots. Records data on exit.'''
	
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
			
			if timestamp_mark:					
				self.line_counter_mark += len(timestamp_mark)
				self.fill_array(self.MARKER_ARRAY, self.line_counter_mark, marker, timestamp_mark)				
				if marker == [[999]]:
					self.stop = timestamp_mark[0] # set last 
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
				if self.plot_fft == True and self.line_counter>self.sample_length and self.line_counter % 10 == 0:
					FFT = compute_fft(self.EEG_ARRAY, self.line_counter, sample_length = self.sample_length)
					self.plot.update_fft(FFT)
				# print timestamp_eeg
				if self.stop != False : # save last EEG chunk before exit
					if timestamp_eeg[-1] >= self.stop:
						plt.close() # oherwise get Fatal Python error: PyEval_RestoreThread: NULL tstate
						self.save_data(sessiontype = '_play', startingpoint = self.learning_end)
						print '\n...data saved.\n Goodbye.\n'
						sys.exit()



def butter_filt(data, cutoff_array, fs = 500, order=4):
    nyq = 0.5 * fs
    normal_cutoff = [a /nyq for a in cutoff_array]
    b, a = signal.butter(order, normal_cutoff, btype = 'bandpass', analog=False)
    data = signal.filtfilt(b, a, data, axis = 0)
    return data

def compute_fft(EEG_ARRAY,offset, sample_length = 1000):
	''' computes fourier transform from slice of EEG_ARRAY. slice is determined by current position and length of the sample to analyze.
	FT should be somehow normalized to fit into graph window - how?'''
	ARRAY_SLICE =  EEG_ARRAY[offset-sample_length:offset,1:]
	ARRAY_SLICE =butter_filt(ARRAY_SLICE, [3,40])
	fft = np.fft.rfft(ARRAY_SLICE, axis = 0)
	fft[0] = 0
	# print np.shape(fft)
	#normalize to one:	
	# fft = fft/np.sum(fft)
	# fft = fft/np.max(fft)
	fft = fft/10000
	return fft


os.chdir(os.path.dirname(__file__)) 	# VLC PATH BUG ==> submit?

if __name__ == '__main__':
	Stream = EEG_STREAM( plot_fft = True, StreamMarkers = False, sample_length = 2000, 
						plot_to_second_screen = False)
	Stream.plot_and_record()