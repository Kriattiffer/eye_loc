# -*- coding: utf-8 -*- 

from PIL import Image, ImageTk, ImageDraw, ImageFont
# background = '#593315'
background = 'black'
kelly_colors_hex = ['#FFB300', '#803E75', '#FF6800', '#A6BDD7', '#C10020', '#CEA262',
					'#817066', '#007D34', '#F6768E', '#00538A', '#FF7A5C', '#53377A', 
					'#FF8E00', '#B32851', '#F4C800', '#7F180D', '#93AA00', '#593315', 
					'#F13A13', '#232C16']
# color names
'''Vivid Yellow
Strong Purple
Vivid Orange
Very Light Blue
Vivid Red
Grayish Yellow
Medium Gray
Vivid Green
Strong Purplish Pink
Strong Blue
Strong Yellowish Pink
Strong Violet
Vivid Orange Yellow
Strong Purplish Red
Vivid Greenish Yellow
Strong Reddish Brown
Vivid Yellowish Green
Deep Yellowish Brown
Vivid Reddish Orange
Dark Olive Green'''

stimlist = [str(a) for a in range(32)]
stimlist = [a for a in u'abcdefghijklmonpqrstuvwxyz_1234567890!@#$%^&*()+=-~[]{};:\"\|?.,/<>½¾¿±®©§£¥¢÷µ¬']
# stim_colors = ['red']*len(stimlist)
stim_colors = ['#505050']*len(stimlist)


# stimlist = [str(a) for a in kelly_colors_hex]
# stim_colors = kelly_colors_hex


imgsize = (200,200)
textpos = 50, -30
# fnt = ImageFont.truetype("arial.ttf", 200)
fnt = ImageFont.truetype("Inconsolata-Regular.ttf", 200)


################################################################

def draw_text(stimlist):
	stims_a, stims_na = [],[]
	for n, stim in enumerate(stimlist):
		b  = Image.new('RGBA', imgsize, background)
		draw = ImageDraw.Draw(b)
		draw.text(textpos, stim, stim_colors[n], font  =fnt)
		stims_na.append(b)

		b1  = Image.new('RGBA', imgsize, 'black')# '#CEA262')
		draw = ImageDraw.Draw(b1)
		draw.text(textpos, stim, "white", font  =fnt)
		draw = ImageDraw.Draw(b1)
		stims_a.append(b1)

	return {'active':stims_a, 'non_active':stims_na}


def save_stims(picdict):
	pass
	for a in picdict.keys():
		for n, pic in enumerate(picdict[a]):
			name = 'stim_%i_%s.png' %(n, a)
			pic.save(name)


def draw_circles():
	stims_a, stims_na = [],[]
	for n, stimcol in enumerate(stim_colors):
		b  = Image.new('RGB', imgsize, background)
		draw = ImageDraw.Draw(b)
		draw.ellipse([0,0,imgsize[-1], imgsize[-1]], outline = None, fill = stimcol)
		stims_a.append(b)

		b  = Image.new('RGB', imgsize, background)
		draw = ImageDraw.Draw(b)
		stims_na.append(b)
	return {'active':stims_a, 'non_active':stims_na}



# pics = draw_circles()
pics = draw_text(stimlist)
save_stims(pics)