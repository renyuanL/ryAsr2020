# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 23:44:17 2019
@author: renyu

functionalKeras005_spchCmdNmelSpec.py
ryLab00.py

ryLab01.py   ... 
ryLab01_1.py ...

ryRecog03.py ...

第一次用 CNN 做出 能辨識 35 個 英文詞 的 語音辨識系統 ...

"""
# In[]


# In[]
#
import time

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import load_model

import sounddevice as sd

import librosa

import matplotlib.pyplot as pl



# In[]
# 完整的詞彙列表，1 + 35 = 36 個英文詞
ryGscList= [ 
 '_silence_',
 
 'one',  'two', 'three', 'four', 'five',
 'six', 'seven', 'eight', 'nine', 'zero',
 
 'yes', 'no',
 'go', 'stop',
 'on', 'off',
 'up', 'down',
 'left', 'right',
 'forward', 'backward',
 
 'marvin', 'sheila',
 'dog', 'cat',
 'bird', 'bed',
 'happy', 'house',
 'learn', 'follow',
 'tree', 'visual',
 'wow',
 
 '_unknown_'
 ]

ryGscList[23]= '_unknown_' # rename "marvin" as "_unknown_"

#LabelDic= ryGscList
# In[11]:


## 重要演算法 開始囉....

def ryFeature(x, 
           sample_rate= 16000, 
           
           frame_length=  512, #1024,
           frame_step=    160, #128,  # frame_length//2
           
           num_mel_bins=     512//8, #32, #100,   #128,
           lower_edge_hertz=   0,     # 0
           upper_edge_hertz= 16000/2, # sample_rate/2   
           
           mfcc_dim= 16 #13
           ):
    
    stfts= tf.signal.stft(x, 
                          frame_length, #=  256, #1024, 
                          frame_step, #=    128,
                          #fft_length= 1024
                          pad_end=True
                          )
    
    spectrograms=     tf.abs(stfts)
    log_spectrograms= tf.math.log(spectrograms)# + 1e-10)
    
    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins= stfts.shape[-1]  #.value
    
    
    linear_to_mel_weight_matrix= tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins, 
          num_spectrogram_bins, 
          sample_rate, 
          lower_edge_hertz,
          upper_edge_hertz)
    
    '''
    mel_spectrograms= tf.tensordot(
          spectrograms, 
          linear_to_mel_weight_matrix, 1)
    
    '''
    
    W= linear_to_mel_weight_matrix
      
    W1= W/ tf.math.reduce_sum(W, axis=0)  # 我把 那些 三角形濾波器 sum to one 了。
    
    mel_spectrograms= spectrograms @ W1  # 這行取代 tensordot()
    
    # 以下這行似乎無作用
    '''
    mel_spectrograms.set_shape(
          spectrograms.shape[:-1].concatenate(
              linear_to_mel_weight_matrix.shape[-1:]))
    '''
    
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms= tf.math.log(mel_spectrograms + 1e-10) # 加上 1e-10 避免 log(0) 的出現
    
    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs= tf.signal.mfccs_from_log_mel_spectrograms(
          log_mel_spectrograms)[..., :mfcc_dim]
    
    feature= {'mfcc':               mfccs, 
              'log_mel_spectrogram':log_mel_spectrograms, 
              
              'log_spectrogram':    log_spectrograms,  # 以下 2個 不太有用，留下僅供參考。
              'spectrogram':        spectrograms}
    
    return  feature




# In[14]:


# 要對 X_* 們作 正規化

def normalize(x, axis= None):   
    if axis== None:
        #x= (x-x.mean())/x.std()
        x= (x - tf.math.reduce_mean(x)
            )/tf.math.reduce_std(x)
    else:
        #x= (x-x.mean(axis= axis))/x.std(axis= axis)
        x= (x - tf.math.reduce_mean(x, axis)
            )/tf.math.reduce_std(x, axis)
    
    return x

# In[17]:




# 啟動 tf.keras
tf.keras.backend.clear_session()  

# 準備好模型的檔名以便訓練時儲存。
fnModel= 'ryAsr2020_ryTrainModel.hdf5'
print(f"... fnModel= {fnModel}")





labels= np.array(ryGscList)


basePath= '../ryData/'
modelPath= basePath + fnModel
 
NNmodel= model= load_model(modelPath)

def recWav(x, probOut= False): #, indexMapping= None):
    
    x= x.flatten()    

    X= ryFeature(x)['mfcc']
        
    X= normalize(X)  # normalized for only one utterence x

    #y= predict(X, probOut= probOut, indexMapping= indexMapping)
    
    X= tf.reshape(X, (1, X.shape[0], X.shape[1], 1))
    
    prob=    model.predict(X)[0]
    index=   prob.argsort()[-1::-1]#np.argmax(prob)
    maxProb= prob[index] #np.max(prob)
    
    #if indexMapping != None:
    #    index= indexMapping[index]
    #####index= CmdList[index]    
    
    y= labels[index]
    
    if probOut==True:
        y= np.vstack((y, maxProb))
    return y
    



# In[]

def rec_long_wav(x= None, T=1, dt=.5, fs=16000, pauseByKey= False, fn= None):
    

    x= sd.rec(int(T*fs), 
            samplerate= fs, 
            channels=   1, 
            dtype=      'float32')
        
    sd.wait()  # Wait until recording is finished
        
    
    T= x.size/fs
    if T==1:
        y= recWav(x, probOut= True)
        y= y[:,0]
    elif T>1:
        # 若輸入語音的長度 T > 1 (sec)，
        # 則移動音框 dt 切成一個一個 1 sec 的語音片段
        # 保持 T/dt 個輸出結果 (邊界之處仍有bug...)
        x= x.flatten()
        xx= librosa.util.frame(x, frame_length=int(1*fs), hop_length=int(dt*fs), axis=0)
        yL= []
        for x in xx:
            y= recWav(x, probOut= True)
            yL += [y[:,0]]
        
        y= np.vstack(yL)
    else:
        y= np.empty()
        pass
    
    
    #print('y=【{}】'.format(y))
    
    return y




if __name__=='__main__':
    
    timeDuration= 10 #sec
   
    xyL= []
    while True:
        aKey= input('press "q" to quit, or any other to record a {} sec wav...'.format(
                timeDuration))
        if aKey == 'q': break
        
        y= rec_long_wav(T=timeDuration, dt= .1)
        
        print(y)
        
        xyL += [y]
    
    
    print('ry: Good Luck, Bye...')    


