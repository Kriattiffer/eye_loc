import numpy as np
import itertools

r,c = 8,9
step = 125
names = [str(a) for a in range(72)]

rows = [list(a) for a in np.arange(r*c).reshape((c,r)).T]
columns = [list(a) for a in np.arange(r*c).reshape((c,r))] 
posr = [0 - step* (len(rows)/2- a) for a in range(r)]
posc = [0 - step* (len(columns)/2- a) for a in range(len(columns))]
pos = [(r, c) for c in posc[::-1] for r in posr ]
config = {

	'stimuli_dir': '.\\rescources\\stimuli\\letters_table',

	'rows': rows,
	'columns': columns,
	'positions': pos,
	'names': names,
	'size':100
}