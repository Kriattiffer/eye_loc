import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import os

def butter_filt(data, cutoff_array, fs = 1000, order=4):
    nyq = 0.5 * fs
    normal_cutoff = [a /nyq for a in cutoff_array]
    b, a = signal.butter(order, normal_cutoff, btype = 'bandpass', analog=False)
    data = signal.filtfilt(b, a, data, axis = 0)
    return data

def slice_eeg(offsets,eeg, channel, sample_length = 1000):
	slices = [] 
	for offset in offsets: 
		# print offsets
		ind = np.argmax(eeg[:,0] > offset) -100#+8
		slice = eeg[ind:ind+sample_length, channel]
		# slice = slice - np.average(slice, axis = 0) #?
		# slice = slice - slice[0,:] #?
		
		if np.shape(slice)[0]<sample_length:
			pass
		else:
			slices.append(slice)
	return np.array(slices)


def from_LSL(channel, mend = 888, mstart = 777):

	markers = np.genfromtxt(markersfile)
	eeg = np.genfromtxt(datafile)#[:,(0,channel)]

	print np.shape(eeg)

	print np.shape(markers)
	# eeg[:,1:] = butter_filt(eeg[:,1:], (0.1,49))

	plt.plot(markers[0:-10,0]-markers[1:-9,0])
	plt.show()
	# mmm = np.logical_and(markers[:,1]!=mstart, markers [:,-1] !=mend)
	mmm = [markers[:,1]==0] # photodiode in above stimulus 0

	markers =  markers[mmm]
	letter_slices = [[] for a in range(np.shape(markers)[0])]
	offsts = markers[:,0]
	deltaof = offsts[1:] - offsts[:-1]
	deltaof = deltaof[deltaof<1]

	# plt.plot(eeg[:,0], eeg[:,channel])
	# for m in markers:
		# plt.axvline(m[0], color = 'red')
	# plt.show()

	# NN = []
	# for a in markers[:,0]:
	# 	nn = find_nearest(max_inds, a)
	# 	NN.append(nn)
	# NN = np.array(NN)
	
	# deltas = NN - markers[:,0]

	# deltaof = np.round(deltaof, 3)
	# # plt.hist(deltaof, bins=15)
	# # plt.show()
	# # plt.clf()
	# print 'delta t markers'
	# plt.plot(deltaof*1000, 'o-')
	# plt.show()

	# print 'delta t markers from 0'
	# deltamarkers =( markers[:,0] - markers[0,0])
	# plt.plot(deltamarkers, 'o')
	# plt.show()

	# print 'delta t between markers and maximums'
	# plt.plot((deltas - deltas[0])*1000, 'o')
	# plt.show()
	# # plt.clf()

	sleeg = slice_eeg(offsts, eeg,channel)
	print np.shape(sleeg)

	return sleeg



if __name__ == '__main__':
	
	datafile = "../_data_play.txt"
	markersfile = "../_markers_play.txt"

	slices = from_LSL(1)


	print np.shape(slices)
	plt.plot(slices.T)
	slices = np.average(slices, axis = 0)
# 
	plt.plot(slices, linewidth = 6)

	plt.show()
