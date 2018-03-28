# -*- coding: utf-8 -*- 

import numpy as np
import itertools

r,c = 9,5
step_vert = 160
step_horiz = 160

names = [a for a in u'abcdefghijklmonpqrstuvwxyz_1234567890!@#$%^&*()+=-~[]{};:\"\|?.,/<>½¾¿±®©§£¥¢÷µ¬']
aim_word = '@neuroscience!'
rows = [list(a) for a in np.arange(r*c).reshape((c,r)).T]
columns = [list(a) for a in np.arange(r*c).reshape((c,r))] 
posr = [200 - step_horiz* (len(rows)/2- a) for a in range(r)]
posr [0] = -780
posc = [0 - step_vert* (len(columns)/2- a) for a in range(len(columns))]
pos = [(r, c) for c in posc[::-1] for r in posr ]
config = {
	'stimuli_dir':'.\\rescources\\stimuli\\nchat_gb',
	'background':'black',
	'rows':rows,
	'columns':columns,
	'positions':pos,
	'names':names,
	'size':100,
	'window_size':(1680, 1050),
	'number_of_inputs':12,
	'aims_learn': [0,5,36,41,21],#[29,35],#,9,13,17,21,25,30,34,38],
	'aims_play': [names.index(a) for a in aim_word]#[0:2]
			}
# print config['aims_play']
# print rows + columns

# print config['positions']


# a
# j
# s
# 1
# 0