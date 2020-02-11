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

1. 從Internet 撈 Data，並整理成方便使用的格式：

    "Internet" ==> ryPrepareDataset03.py  ==> "gscV2_data.npz" (a big speech data file)

2. Train a model

    "gscV2_data.npz" ==> ryAsr2020_ryTrainModel06.py ==> "ryAsr2020_ryTrainModel.hdf5" (a CNN model with mel-cepstrum feature)

3. do Automatic Speech Recognition (ASR)

    "ryAsr2020_ryTrainModel.hdf5" ==> ryRealTimeAsr06.py ==> "Results of ASR "
