 # -*- coding: utf-8 -*- 

import numpy as np
import itertools

r,c = 6,6
step_vert = 150
step_horiz = 150

names = [a for a in u'abcdefghijklmonpqrstuvwxyz_1234567890!@#$%^&*()+=-~[]{};:\"\|?.,/<>½¾¿±®©§£¥¢÷µ¬']
aim_word = 'brain'
rows = [list(a) for a in np.arange(r*c).reshape((c,r)).T]
columns = [list(a) for a in np.arange(r*c).reshape((c,r))] 
posr = [75 - step_horiz* (len(rows)/2- a) for a in range(r)]
posc = [75 - step_vert* (len(columns)/2- a) for a in range(len(columns))]
pos = [(r, c) for c in posc[::-1] for r in posr ]
config = {
	'stimuli_dir':'.\\rescources\\stimuli\\letters_grey_black',
	'background':'black',
	'rows':rows,
	'columns':columns,
	'positions':pos,
	'names':names,
	'size':100,
	'window_size':(1680, 1050),
	'number_of_inputs':12,
	'aims_learn': [0,5,36,41,21],#[29,35],#,9,13,17,21,25,30,34,38],
	'aims_play': [names.index(a) for a in aim_word],#[0:2]
	'shrink_matrix' : 1,
	'textsize' : 0.07
			}
# print config['aims_play']
#print rows + columns
print pos
