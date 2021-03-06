#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# ----------------------------------------------------------------------------
# Name:       eyetracker.py
# Purpose:    Communication with SMI RED 500 device
# Author: Rafael Grigoryan, kriattiffer at gmail.com
# Date: December 20, 2016
# ----------------------------------------------------------------------------

from iViewXAPI import  *  #iViewX library
from ctypes import *
import time, sys, ast, os
import numpy as np
import present
from pylsl import StreamInlet, resolve_stream


class CCalibrationPointStruct(Structure):
    _fields_=[('number', c_int),('positionX', c_int),('positionY', c_int)]
calibrationPoint=CCalibrationPointStruct(0,0,0)


def create_stream(stream_name_markers = 'CycleStart', recursion_meter = 0, max_recursion_depth = 3):
        ''' Opens LSL stream for markers, If error, tries to reconnect several times'''
        if recursion_meter == 0:
            recursion_meter +=1
        elif 0<recursion_meter <max_recursion_depth:
            print 'Trying to reconnect for the %i time \n' % (recursion_meter+1)
            recursion_meter +=1
        else:
            print ("Error: Eyetracker cannot conect to markers stream\n")
            return None
            
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
                return None
        else:
            print 'Error: markers stream is not available\n'
            return create_stream(stream_name_markers,recursion_meter)
        return inlet_markers


class Eyetracker():
    """ Class for interaction with iViewXAPI and experiment in present.py """
    def __init__(self, namespace, debug = False, number_of_points = 9, screen = 0):
        self.namespace = namespace
        self.number_of_points = number_of_points
        self.screen = screen

        self.im  = create_stream()
        self.host_ip = '192.168.0.2'
        self.server_ip = '192.168.0.3'
        if not self.im:
            if debug != True:
                self.exit_()

    def get_calibration_point(self, number):
        'updates calibrationPoint data structure'
        self.res=iViewXAPI.iV_GetCalibrationPoint(number, byref(calibrationPoint))

    def get_accuracy(self):
        'updates AccuracyStruct data structure'
        self.res=iViewXAPI.iV_GetAccuracy(byref(accuracyData), c_int(0))
        dlx=accuracyData.deviationLX
        dly=accuracyData.deviationLY
        drx=accuracyData.deviationRX
        dry=accuracyData.deviationRY
        return np.mean([dlx,drx]), np.mean([dly, dry])

    def  calibrate(self):
        '''configure and start calibration'''

        numberofPoints = self.number_of_points # can be 2, 5 and 9
        displayDevice = self.screen # 0 - primary, 1- secondary (?)
        pointBrightness = 250
        backgroundBrightnress = 0
        targetFile = b""
        calibrationSpeed = 0 # slow
        autoAccept  = 1 # 0 = manual, 1 = semi-auto, 2 = auto 
        targetShape = 1 # 0 = image, 1 = circle1, 2 = circle2, 3 = cross
        targetSize = 20
        WTF = 1 #do not touch -- preset?

        calibrationData = CCalibration(numberofPoints, WTF, displayDevice, 
                                        calibrationSpeed, autoAccept, pointBrightness,
                                        backgroundBrightnress, targetShape, targetSize, targetFile)

        self.res = iViewXAPI.iV_SetupCalibration(byref(calibrationData))
        print "iV_SetupCalibration " + str(self.res)

        new_positions= [(841,526),(196,137), (1374, 148),(204,933), (1382,920), (210, 532), (832,143), (1392, 517), (861, 920)]
        #new_positions= [(840, 525),(280,60), (280, 990), (1400, 60), (1400,990), (280, 525), (840, 525), (840, 60), (840, 990)]
        #for i in range(1,10,1):
            #self.get_calibration_point(i)
            #print calibrationPoint.positionX, calibrationPoint.positionY
        for i in range(len(new_positions)):
            self.res=iViewXAPI.iV_ChangeCalibrationPoint(i+1,new_positions[i][0], new_positions[i][1]) # new coordinates for calibration points

        self.res = iViewXAPI.iV_Calibrate()
        print   "iV_Calibrate " + str(self.res)

    
    def validate(self):
        ''' Present 4 points to validate last calibration.
            Results are displayed in iViewX'''
        self.res = iViewXAPI.iV_Validate()
        print "iV_Validate " + str(self.res)
        #self.res = iViewXAPI.iV_ShowAccuracyMonitor()
        #self.res = iViewXAPI.iV_ShowEyeImageMonitor()
        #raw_input('press any key to continue')

    def connect_to_iView(self):
        ''' Connect to iViewX using predeficed host and server IPs'''
        self.res = iViewXAPI.iV_Connect(c_char_p(self.host_ip), c_int(4444), 
                                        c_char_p(self.server_ip), c_int(5555))
        print "iV_sysinfo " + str(self.res)

    def send_marker_to_iViewX(self, marker):
        ''' Sends marker to the eyetracker. Marker becomes iViewX event. '''
        marker = ast.literal_eval(marker)
        if marker == [888]:
            print marker
            res = iViewXAPI.iV_SendImageMessage(str(marker[0]) + '.jpg')
        else:
            res = iViewXAPI.iV_SendImageMessage(str(marker[0]))
        # if str(self.res) !='1':
            # print "iV_SendImageMessage " + str(self.res)
    
    def experiment_loop(self):
        self.res = iViewXAPI.iV_StartRecording ()
        print "iV_record " + str(self.res)
        if not self.im:
            print 'LSL socket is Nonetype, exiting'
            self.exit_()
        print 'server running...'
        while 1:
            marker, timestamp_mark = self.im.pull_sample()
            IDF_marker =  str([marker, timestamp_mark])
            self.send_marker_to_iViewX(IDF_marker)
            if marker == [999]:
                self.exit_()

    def main(self):
        accuracy_x=[]
        accuracy_y=[]
        self.connect_to_iView()
        self.calibrate()
        #self.validate()
        for i in range(3):
            self.validate() # validate 3 times, then show mean deviations
            x,y =self.get_accuracy()
            accuracy_x.append(x)
            accuracy_y.append(y)  
        print 'Mean deviation x (deg): ', np.mean(accuracy_x)
        print 'Mean deviation y (deg): ', np.mean(accuracy_y)

        self.namespace.EYETRACK_CALIB_SUCCESS = True
        self.experiment_loop()


    def exit_(self):
        ''' Close all streams, save data and exit.'''
        self.im.close_stream()
        time.sleep(1)
        self.res = iViewXAPI.iV_StopRecording()

        user = '1'
        regname = os.path.basename(self.namespace.config).split('.')[0] + '_'
        filename = r'C:\Users\iView X\Documents\SMI_BCI_Experiments/' + regname + present.timestring()
        self.res = iViewXAPI.iV_SaveData(filename, 'description', user, 1) # filename, description, user, owerwrite
        if self.res == 1:
            print 'Eyatracking data saved fo %s.idf' % filename
        else:
            print "iV_SaveData " + str(self.res)
        self.res = iViewXAPI.iV_Disconnect()
        
        sys.exit()



if __name__ == '__main__':
            
    RED = Eyetracker(namespace = type('test', (object,), {})(), debug = True)
    RED.main()


