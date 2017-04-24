# -*- coding: utf-8 -*- 

import numpy as np
import itertools

r,c = 6,7
step_vert = 125
step_horiz = 200

names = [a for a in u'abcdefghijklmonpqrstuvwxyz_1234567890!@#$%^&*()+=-~[]{};:\"\|?.,/<>½¾¿±®©§£¥¢÷µ¬']
# print [names.index(a) for a in 'neuroscience!']
rows = [list(a) for a in np.arange(r*c).reshape((c,r)).T]
columns = [list(a) for a in np.arange(r*c).reshape((c,r))] 
posr = [100 - step_horiz* (len(rows)/2- a) for a in range(r)]
posc = [0 - step_vert* (len(columns)/2- a) for a in range(len(columns))]
pos = [(r, c) for c in posc[::-1] for r in posr ]
config = {
	'stimuli_dir':'.\\rescources\\stimuli\\letters_table_black',
	'background':'black',
	'rows':rows,
	'columns':columns,
	'positions':pos,
	'names':names,
	'size':100,
	'window_size':(1680, 1050),
	'number_of_inputs':12,
	'aims_learn': [1,5,9,13,17,21,25,30,34,38],
	'aims_play': [names.index(a) for a in 'neuroscience']
			}
print config['aims_play']
# print config