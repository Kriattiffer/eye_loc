import os, sys
import shutil

# copyfile(src, dst)
def move(basename, dirname, filename):
	path = os.path.join(basename, dirname, filename)
	shutil.move(os.path.join(basename, filename), path)

a = [a for a in os.walk(r'D:\Rafael\eye_loc\experimental_data\ann\BCI_EYE_anna_p\valid_eeg_anal')]

for fn in a:

	if len(fn[-1]) ==0:
		continue

	print fn[0], [a for a in fn[-1] if 'play' in a]

	for f in fn[-1]:
		f = f.upper()

	# [os.mkdir(os.path.join(fn[0], dr)) for dr in ['SS', 'LL', 'SL', 'LS', 'M']]

		if  'SMALL_STIMS_LARGE_MATRIX' in f:
			move(fn[0], 'SL', f)
			
		elif  'LARGE_STIMS_LARGE_MATRIX' in f:
			move(fn[0], 'LL', f)

		elif  'SMALL_STIMS' in f:
			move(fn[0], 'SS', f)

		elif  'LARGE_STIMS' in f:
			move(fn[0], 'LS', f)
			
		elif  'MEDIUM' in f:
			move(fn[0], 'M', f)