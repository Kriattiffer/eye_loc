from pylsl import StreamInfo, StreamInlet, resolve_stream
import pickle

streams = resolve_stream('name', 'CycleStart')
print streams
inlet_pic = StreamInlet(streams[0])   

print 'connected'
while  1:
	ss, ss_time = inlet_pic.pull_sample()
	if ss !=[]:
		print ss
		s = pickle.loads(ss[0])
		s.show()