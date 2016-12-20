import sys
sys.path.append('..')
import numpy as np
from record import butter_filt
from matplotlib import pyplot as plt
from scipy import signal
import os

def slice_eeg(offsets,eeg, channel, sample_length = 70):
	slices = [] 
	for offset in offsets: 
		# print offsets
		ind = np.argmax(eeg[:,0] > offset) #+8
		slice = eeg[ind:ind+sample_length, channel]
		# slice = slice - np.average(slice, axis = 0) #?
		# slice = slice - slice[0,:] #?
		
		if np.shape(slice)[0]<sample_length:
			pass
		else:
			slices.append(slice)
	return np.array(slices)

def get_maximums_from_eeg(eeg, window_length = 140):
	eeg = eeg[:,0:2]

	maxlist = []
	amold =0 
	for a in  range(0, np.shape(eeg)[0]-window_length):
		slice = eeg[a:a+window_length,:]
		am = np.argmax(slice[:,1])
		if slice[:,1][am] < 14900:
			pass
		else:
			mx = slice[:,0][am]
			if mx not in maxlist:
				try:
					delta = (mx-maxlist[-1])*1000 
					if abs(delta)<window_length:
						pass	
						# print mx
						# print maxlist[-1]
						# print a
						# print (mx-maxlist[-1])*1000
						# plt.plot(slice[:,1])
						# plt.plot(eeg[a-delta:a+window_length-delta,1])
						# plt.plot(amold, 'o')

						# plt.show()
					else:
						pass
						maxlist.append(mx)

				except:
					pass
				
				if len(maxlist) == 0:
				# if 1:
					maxlist.append(mx)

	maxinds =  np.array(maxlist)
	minddelta = (maxinds[1:] - maxinds[:-1])*1000

	# plt.hist(minddelta, bins = 200)
	# plt.show()
	print 'delta t maximums'
	plt.plot(minddelta,"o")
	plt.show()
	return maxinds


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def from_LSL(channel, mend = 999, mstart = 777):
	'''1 for photodiode, 0 for neurotest'''
	markers = np.genfromtxt(markersfile)
	eeg = np.genfromtxt(datafile)
	print np.shape(eeg)

	print np.shape(markers)
	# eeg[:,1:] = butter_filt(eeg[:,1:], (0.1,49))
	max_inds  = get_maximums_from_eeg(eeg)


	mmm = np.logical_and(markers[:,1]!=mstart, markers [:,-1] !=mend)

	markers =  markers[mmm]
	letter_slices = [[] for a in range(np.shape(markers)[0])]
	offsts = markers[:,0]
	deltaof = offsts[1:] - offsts[:-1]
	deltaof = deltaof[deltaof<1]



	NN = []
	for a in markers[:,0]:
		nn = find_nearest(max_inds, a)
		NN.append(nn)
	NN = np.array(NN)
	
	deltas = NN - markers[:,0]

	deltaof = np.round(deltaof, 3)
	# plt.hist(deltaof, bins=15)
	# plt.show()
	# plt.clf()
	print 'delta t markers'
	plt.plot(deltaof*1000, 'o-')
	plt.show()

	print 'delta t markers from 0'
	deltamarkers =( markers[:,0] - markers[0,0])
	plt.plot(deltamarkers, 'o')
	plt.show()

	print 'delta t between markers and maximums'
	plt.plot((deltas - deltas[0])*1000, 'o')
	plt.show()
	# plt.clf()

	sleeg = slice_eeg(offsts, eeg,channel)
	print np.shape(sleeg)

	return sleeg

def from_plain_eeg(channel):
	'''1 for photodiode, 0 for neurotest'''

	eeg = np.genfromtxt(datafile)#[20000:-20000,:]

	offs = np.array( range(0, np.shape(eeg)[0]*2, 250))/1000.0 + eeg[0,0]
	sleeg = slice_eeg(offs, eeg,channel)

	# max_inds  = get_maximums_from_eeg(eeg)
	# NN = []
	# for a in offs:
	# 	nn = find_nearest(max_inds, a)
	# 	# print a, nn, a-nn
	# 	NN.append(nn)
	# NN = np.array(NN)
	# deltas = (NN - offs)*1000

	# eegtimings = eeg[1:,0] - eeg[0:-1,0]
	# eegtimings = np.round(eegtimings*1000, 8)
	# print 'delta t between eeg samples'

	# # plt.plot(eegtimings, 'o')
	# # plt.show()

	# print 'delta t between markers and maximums'
	# plt.plot((deltas - deltas[0])*1000, 'o')
	# plt.show()

	print np.shape(sleeg)
	return sleeg




if __name__ == '__main__':
	
	datafile = "./_data_play.txt"
	markersfile = "./_markers_play.txt"

	slices = from_LSL(1)


	print np.shape(slices)
	plt.plot(slices.T)
	slices = np.average(slices, axis = 0)
	plt.plot(slices, linewidth = 6)

	plt.show()
