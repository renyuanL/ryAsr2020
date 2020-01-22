'''

ryRealTimeAsr02.py
ryRealTimeAsr03.py

'''

import sounddevice as sd
import numpy as np

__all__= ['ryQ', 'ryRecogQGet', 'ryStream']


nChannel= 1
nSamplePerSec= sampleRate= 16000
nSamplePerFrame= 1000
nFramePerSec= nSamplePerSec//nSamplePerFrame # 16 frame/sec

indexFrame= 0
bufferTime= 10 #sec
nFramePerBuffer= nFramePerSec* bufferTime #160 # frames == 10 sec
BufferSize= nFramePerBuffer


ryBuffer= (1e-10)*np.random.random((BufferSize, nSamplePerFrame, nChannel))

def ryClearBuffer():
    global ryBuffer, indexFrame
    indexFrame=0
    ryBuffer= (1e-10)*np.random.random((BufferSize, nSamplePerFrame, nChannel))

def ryCallback(indata, outdata, frames, time, status):
    global indexFrame, ryBuffer

    if status:
        print(status)
        
    ## for sound playback
    outdata[:] = indata * 0.1  # *2

    ryBuffer[indexFrame%BufferSize]= indata       
    indexFrame += 1


import time

import pylab as pl

#import ryLab01_1 as ry
import ryRecog06 as ry


def ryOpenStream(aCallback):
    
    aStream= sd.Stream(callback=    aCallback, 
               channels=   nChannel,       # 1 for mono, 2 for stereo
               samplerate= nSamplePerSec,  # sample/sec
               blocksize=  nSamplePerFrame #1000   # frame_size_in_sample, sample/frame
               )
    return aStream
    

def ryGet1secSpeech(withPrintInfo= False):
    global ryBuffer, BufferSize, indexFrame
    
    x= ryBuffer
    t1= (indexFrame%BufferSize)
    x= np.vstack((x[t1:], x[0:t1]))
    x= x.flatten()    
    x= x.astype(np.float32) 
    x= x[-16000:]    
    spEng= x.std()
    
    if withPrintInfo== True:
        print('[{:.1f}]({:.4f}), '.format(t, spEng), end='', flush=True)
    else:
        print('.', end='', flush=True)
    
    return x

def ryGet1secSpeechAndRecogItWithProb():
    
    #global yp
    
    x= ryGet1secSpeech()
    
    #y, prob= ry.recWav(x, featureOut= False, withProb= True)
    yp= ry.recWav(x, probOut= True)
    
    
    y=    yp[0,0]
    prob= yp[1,0].astype('float32')
    
    return y, prob

# In[]


from queue import Queue
from threading import Thread
import numpy as np
import matplotlib.pyplot as pl
import time
import sounddevice as sd


ryQ= Queue(100)
ryRecog_stop= False

def ryRecog(q): 
    
    print('ryRecog start ....') 
    while ryRecog_stop == False:
 
        x= ryGet1secSpeech()
        
        yp= ry.recWav(x, probOut= True)
    
        y=    yp[0,0]
        prob= yp[1,0].astype('float32')
    
        q.put((y,prob))
        
    print('ryRecog ended ....')
       
def ryRecogQGet(q):
    N=  q.qsize()
    xL= [q.get() for i in range(N)]
    return xL

#

#th_ryRecog.start()






# In[]

ryClearBuffer()


def ryAsrStream():
    return ryOpenStream(ryCallback)

ryStream= ryAsrStream()
ryStream.start()


th_ryRecog= Thread(target= ryRecog,  
                   args=(ryQ,), 
                   daemon= True)
th_ryRecog.start()


# %%
ryTestRecogFor_T_sec_stop= False
def ryTestRecogFor_T_sec(T= 100, dt= .1):
    #T= 100
    #dt= .1
    t= 0
    while t<T and ryTestRecogFor_T_sec_stop==False:
        
        # 擷取辨認結果
        yL= ryRecogQGet(ryQ)
        
        # 處理辨認結果
        for y, prob in yL:
            if y =='_unknown_' or prob<0.5: 
                continue
            else:
                print(f'{y} @t={t:.1f}')
        
        time.sleep(dt)
        t +=dt
    print('---- end ryTestRecogFor_n_sec(n= 100) ----')

# 測試 ryRecog for 100 秒
#ryTestRecogFor_T_sec(T= 100)

if __name__=='__main__':
    T= 1000 # sec
    th_ryTest= Thread(target= ryTestRecogFor_T_sec,  
                       args=(T,), 
                       daemon= True)
    th_ryTest.start()
    
    # 在這裡 可做很多事，但目前僅是睡覺....
    time.sleep(T)
    
    # 最後的清道夫
    #'''
    
    ryRecog_stop= True
    ryTestRecogFor_T_sec_stop= True
    ryStream.stop()
    ryStream.close()
    #'''


# %%
if __name__=='__main__xxxx':
    
    spDuration= 1000 * bufferTime  # bufferTime= 10 seconds
    recProbToConfirm= 0.2
    
    
    #from tqdm import tqdm
    
    #with ryOpenStream(ryCallback) as ryStream:
    with ryAsrStream() as asrStream:
        
        t0= time.time()
        
        t=   0
        dt= .1 # sec  time_interval to do recognition
        loopNumber= 1000
        loopTime= loopNumber * dt
        
        nStop= 0
        nReallyStop= 10
        
        #while t<spDuration:
        #for i in range(loopNumber):
        print('\n.... Listen forever ....\n')
        while True:
            
            t00= time.time()
            
            y, prob= ryGet1secSpeechAndRecogItWithProb()
            
            if prob > recProbToConfirm: #0.8:
                
                if y == '_unknown_': y=''
                
                info= f'【{y}】    @p= {prob:.2f}  @time=({t:.1f})'
                print(info, end='\n', flush=True)
            
                if y== 'stop':
                    nStop+= 1
                    
                    info= f'【{y}】({nStop}, Really？)'
                    print(info, end='\n', flush=True)
                    
                    if nStop >= nReallyStop:
                        
                        info= f'【{y}】(OK, I will STOP！！！)'
                        print(info, end='\n', flush=True)
                        break
                
                if y== 'go':
                    nStop= 0
                    
                    info= f'【{y}】(OK, Reset nStop= {nStop}, and then GO)！！！'
                    print(info, end='\n', flush=True)
            
            dt00= time.time()-t00
            if dt-dt00>0:
                time.sleep(dt-dt00)
            
            t+=dt
            
        dtt= time.time() - t0
        print('dtt(sec)= {:.3f}'.format(dtt))
    
    print('ry: 明けましておめでとう Happy New Year, 2020 !!!')
