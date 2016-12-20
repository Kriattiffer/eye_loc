# -*- coding: utf-8 -*- 

# Runs on SMI server laptop

from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_stream

def create_stream(stream_name_markers = 'CycleStart', recursion_meter = 0, max_recursion_depth = 3):
		''' Opens LSL stream for markers, If error, tries to reconnect several times'''
		if recursion_meter == 0:
			recursion_meter +=1
		elif 0<recursion_meter <max_recursion_depth:
			print 'Trying to reconnect for the %i time \n' % (recursion_meter+1)
			recursion_meter +=1
		else:
			print 'exiting'
			sys.exit()
			inlet_markers = []
			
		print ("Eyetracker connecting to markers stream...")
		# inlet for markers
		if stream_name_markers in [stream.name() for stream in resolve_stream()]:
			sterams_markers = resolve_stream('name', stream_name_markers)
			inlet_markers = StreamInlet(sterams_markers[0])   
			try:
				inlet_markers
				print '...done \n'
			except NameError:
				print ("Error: Eyetracker cannot conect to markers stream\n")
				sys.exit()
		else:
			print 'Error: markers stream is not available\n'
			return create_stream(stream_name_markers,recursion_meter)
		return inlet_markers

def send_marker_to_iViewX(marker):
	pass

def main():
	im = create_stream()
	while 1:
		marker, timestamp_mark = im.pull_sample()
		IDF_marker =  str([marker, timestamp_mark])
		send_marker_to_iViewX(IDF_marker)


if __name__ == '__main__':
	main()