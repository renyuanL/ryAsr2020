# ryAsr2020
Restarting Asr Project @ 2020

全新 語音辨識控制引擎 ryAsr2020

https://github.com/renyuanL/ryAsr2020

雖然還沒有很完整的說明，但已有個初胚，就在過年前先釋放出來..... 
過年期間沒啥事的話，務必去玩玩，沒GPU也OK....

1. 進去 ryWork 資料夾， 
2. 執行 
    python ryRealTimeAsr06_spaceInvader.py 
3. 若發現缺少啥 模組， 就 pip install 啥模組，直到能跑

PS: 麥可風要先接好....

-----------------------

簡要說明
========

1. 從Internet 撈 Data，並整理成方便使用的格式：
   ( 這個過程目前很耗時，但還好做一次就好，希望能改善，並嘗試增加本地的語言的語音資料。)

    "Internet" ==> ryPrepareDataset03.py  ==> "gscV2_data.npz" (a big speech data file)
    
2. Train a model
    ( 目前使用22個英文關鍵詞，辨識率大約 80%，改善空間仍然很大) 

    "gscV2_data.npz" ==> ryAsr2020_ryTrainModel06.py ==> "ryAsr2020_ryTrainModel.hdf5" (a CNN model with mel-cepstrum feature)

3. do Automatic Speech Recognition (ASR)
    
    "ryAsr2020_ryTrainModel.hdf5" ==> ryRecog06.py  ==> "Results of ASR "
    
4. play with ASR in Real-time
    ( 可以嘗試一些遊戲，善用這22個 英文詞彙。)

    "ryAsr2020_ryTrainModel.hdf5" 
    ==> 
    01. ryRealTimeAsr06.py
    02. ryRealTimeAsr06_spaceInvader.py
    03. ryRealTimeAsr06_threading.py


