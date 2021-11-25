from multiprocessing import Process, Queue
import cv2
import numpy as np
import Jetson.GPIO as GPIO
#import serial
import time
from threading import Thread
import tensorflow as tf
#importing the necesarry package

class VideoStream:
    '''
    A class for camera
    the image acquisition caried out in a separate thread from the main python thread
    '''
    def __init__(self, src=0):
        '''initialize the camera'''
        self.stream = cv2.VideoCapture(src,cv2.CAP_V4L2) #in jetson nano the backend in V4L2
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,320)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
        self.stream.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        (self.grabbed, self.frame) = self.stream.read()
        # initialize stop variable
        print(self.stream.get(cv2.CAP_PROP_FPS))
        

    def start(self):
        '''start the thread to read frames from the camera'''
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        # the read the frame from the camera
        while True:
            # stop reading if the stop variable is True
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
    def read(self):
        # return the frame most recently read
        return self.frame
    def stop(self):
        # stop the thread and release the camera.
        self.stopped = True
        self.stream.release()
#declare the GPIO pin for the stepper motor direction and pulse pin. 
dir_pin = 23
pul_pin = 21
def move(p,d):
    '''funtion for sending pulses to move the stepper motor'''
    GPIO.output(dir_pin,d)
    for i in range(p):
        GPIO.output(pul_pin,GPIO.HIGH) #high signal
        time.sleep(0.001)
        GPIO.output(pul_pin,GPIO.LOW) #low signal
        time.sleep(0.001)

def clear_queue(queue):
    ''' function for clearing the queue'''
    try:
        while True:
            queue.get_nowait()
    except:
        pass
    queue.close()
    queue.join_thread()


def eject(qu,dum):
    '''
    function for performing the divertion of the plastic contaminant.
    this function executed in a different process from the main process, so the detection and the ejection has different process.
    this process get the signal from the detection process by getting the boolean signal. 

    arguments: qu: queue where the signal will be given
               dum: dummy argument because it returns error if the function only has 1 argument.
    '''
    print('ejector_On')
    plank = False # variable to track the status of the plank, whether it is opened or closed.
    st = time.time()
    while (True):
        et = time.time()
        if(not qu.empty()): # get the signal when the queue is not empty.
            if(qu.get()):
                st = time.time() # start the timer when there is True signal
                print('open')
                if plank == False:
                    move(130,1) # open the gap to the contaminant dump by rotating 117 degree
                    plank = True
            else:
                print('pass')
        if (et - st >= 0.5 and plank): #close the plank only when the timer ends and the plank is in opened position
            move(120,0) # close the gap by rotating 108 degree, it is smaller than the opening angle, for compensating the plank overshoot.
            time.sleep(0.3)
            plank = False
    GPIO.cleanup()


if __name__ == '__main__':
    q = Queue() # start the queue for sending signal to the eject process

    GPIO.setmode(GPIO.BOARD) #set the GPIO numbering mode according to the board pin number
    # 1 ccw, 0 cw axis facing up
    GPIO.setup(dir_pin, GPIO.OUT, initial = GPIO.LOW) #assign the GPIO pin for stepper motor direction
    GPIO.setup(pul_pin, GPIO.OUT, initial = GPIO.LOW) #assign the GPIO pin for stepper motor pulse
    dummy = 0 # dummy arguments
    p = Process(target=eject, args=(q,dummy),daemon=True) #start the process for eject function
    p.start()

    cam = VideoStream(src=0).start() #start the camera thread
    f = cam.read() #try acquiring frame from the camera
    f = np.array(f)
    print(f.shape) #checking the resolution of the image
    f_count = 0 #fps counter variable
    #the text settings
    font_sty = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (200,200,0)
    font_color2 = (50,250,0)
    font_thick = 1
    txt_pos1 = (20,f.shape[0]-40)
    txt_pos2 = (20,f.shape[0]-10)
    msg = 'la' #variable to contain the text to be displayed
    s = time.time()
    saved_model_loaded = tf.saved_model.load('mymodel_horgreyfat3_best_TFTRT_fp16') # load the neural network model
    infer = saved_model_loaded.signatures['serving_default']
    print('que empty: ',q.empty())
    dcounter = 0 # initialize the detection counter

#main program loop
while(True):
    frame = cam.read() # read the frame from the stream
    f_count = f_count + 1 # counting frame per second
    f = frame 
    f = cv2.cvtColor(f,cv2.COLOR_BGR2RGB) # reorder the BGR channel to RGB channel
    f = f/255 # rescale the range of the pixel value, so that it became 0 to 1.
    f = np.expand_dims(f,axis=0) # expand the dimension of the image, because the neural network accept 4 dimension [batch,height,width,channel]
    f = tf.constant(f,dtype = tf.float32) # convert it to Tensor
    ans = infer(f)['global_max_pooling2d'].numpy() #feed the image to the NN model.
    if(ans[0][0]>0.5): #if the prediciton value > 0.5 increase the detection counter
        dcounter = dcounter + 1
        pls = False
        if (dcounter>2): #send True signal to the eject process if the detection counter > 2
            msg = 'detection: there is foreign object'
            pls = True
        
    else: # if the prediction value < 0.5, reset the detection counter to 0
        msg = 'detection: none'
        pls = False
        dcounter=0

    if (q.empty()):# send the signal to the eject process through the queue
        q.put_nowait(pls)

    e = time.time()
    #display the fps and the detection result.
    if ((e-s) >= 1.0): #every 1 sec, count the frames that has been read
        msg2 = 'FPS: '+str(f_count)
        s = time.time()
        f_count = 0
    cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
    frame = cv2.putText(frame,msg,txt_pos1,font_sty,font_scale,font_color,font_thick,cv2.LINE_AA)
    frame = cv2.putText(frame,msg2,txt_pos2,font_sty,font_scale,font_color2,font_thick,cv2.LINE_AA)
    cv2.imshow('preview',frame) #showing the acquired image

    if cv2.waitKey(1) == 27: #if esc is pressed, terminate the program
        break
cam.stop()
cv2.destroyAllWindows()
clear_queue(q)
p.terminate()
GPIO.cleanup()
