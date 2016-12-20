# -*- coding: utf-8 -*- 

import numpy as np
from record import butter_filt
from matplotlib import pyplot as plt
from scipy import signal
import os

def slice_eeg(offsets,eeg, sample_length = 300):
		slices = [] 
		for offset in offsets:
			ind = np.argmax(eeg[:,0] > offset) #+8
			slice = eeg[ind:ind+sample_length]

			# slice = slice - np.average(slice, axis = 0) #?
			# slice = slice - slice[0,:] #?
			
			if np.shape(slice)[0]<sample_length:
				pass
			else:
				slices.append(slice)
		slices = np.average(slices, axis = 0)
		return np.array(slices)


def from_LSL(mstart = 777, mend = 888):

	markers = np.genfromtxt(r'_data\3108_eeg\_markers.txt')
	eeg = np.genfromtxt(r'_data\3108_eeg\_data.txt')
	print markers
	print np.shape(eeg)

	eeg[:,1:] = butter_filt(eeg[:,1:], (0.1,40))

	aim_list = [111,222,333,444,555,666]*10
	aim_list  = aim_list[0:47]


	mmm = markers[:,1]==mstart
	mmm[-1] = True
	letter_slices = [[] for a in range(len(aim_list)+1)]
	cc = -1

	for a , mrk in zip(mmm[:-1], markers):
		print cc

		if a:
			cc +=1
		else:		
			letter_slices[cc].append(mrk)

	slices =  np.array(letter_slices)
	aim_eeg = []
	non_aim_eeg = []

	for aim, letter  in zip(aim_list, letter_slices):
		letter = np.array(letter)
		if len(letter) !=0:
			aim_offsets =  letter[letter[:,1] == aim][:,0]
			non_aim_offsets = letter[letter[:,1] != aim][:,0]
			
			aim_slices = slice_eeg(aim_offsets, eeg)
			non_aim_slices = slice_eeg(non_aim_offsets, eeg)

			aim_eeg.append(aim_slices)
			non_aim_eeg.append(non_aim_slices)
		else:
			pass

	aim_eeg = np.array(aim_eeg)
	non_aim_eeg = np.array(non_aim_eeg)

	aim_eeg = np.average(aim_eeg, axis = 0)
	non_aim_eeg = np.average(non_aim_eeg, axis = 0)
	# plt.plot(non_aim_eeg[:,1:])
	# plt.title('LSL')

	# plt.show()
	# plt.clf()
	return [aim_eeg[:,1:], non_aim_eeg[:,1:]]


lsleeg = from_LSL()

fig,axs = plt.subplots(nrows =3, ncols = 3)
channels = ['cpz', 'p3', 'p4', 'po3', 'poz', 'po4', 'o1', 'o2']
for a in range(8):
	# ax[a]
	delta = lsleeg[0][:,a] - lsleeg[1][:,a]
	axs.flatten()[a].plot(range(0, len(delta)*2, 2), delta)
	axs.flatten()[a].plot(range(0, len(delta)*2, 2), np.zeros(np.shape(delta)))

	axs.flatten()[a].set_title(channels[a])
	# delta = lsleeg[1][:,a]
	# axs.flatten()[a].plot(range(0, len(delta)*2, 2), delta)
	# axs.flatten()[a].plot(range(0, len(delta)*2, 2), np.zeros(np.shape(delta)))

	# plt.plot(lsleeg[1][:,a])
plt.show()
	# plt.clf()
# plt.plot(tmpl[:,1:])


# # plt.plot(efeeg[0]/1000) 
# print np.logical_or(a==1, a==2)