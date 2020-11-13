# AMPA-Net
This is the official TensorFlow implementation of AMPA-Net
[AMPA-Net：Optimization-Inspired Attention Neural Network for Deep Compressed Sensing，In the IEEE 20th International Conference on Communication Technology (ICCT) oral]
## Compatibility
* The code is tested using Tensorflow 1.0 under Ubuntu 16.04 with Python 2.7.

* Recommend Environment: Anaconda

## Preparing Training Datasets-Image-91
   Download  Training_Data_Img91.mat from https://pan.baidu.com/s/1c34-DWDhFBsNPX23bs4H1A code is 7dV2 and cp it into this folder
## Preparing Testing Datasets:Urban-100 and BSDS100
   Download  Urban-100 from https://pan.baidu.com/s/1M9yoJaQ1DqCrpXO-DqSo4Q code is 2i5o and cp it into this folder
   Download  BSDS-100 from https://pan.baidu.com/s/12Y4FOzmOcNjQTjvtRq0mpQ code is OBbE and cp it into this folder
## Running training and testing
   Python trainn.py 

## Results

Our model achieves the following performance on Set11,BSD68,BSDS100,Urban100
|  Model name  |     cs=50%   |    cs=40%    |    cs=25%    |   cs=10%     |     cs=4%    |    cs=1%    |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |------------ |
|   BM3D-AMP   |     34.11    |     32.79    | 71.4 (± 1.8) | 74.7 (± 3.3) | 56.7 (± 5.2) |56.7 (± 5.2) |
|   LD-AMP     |     35.92    | 76.9 (± 4.6) | 73.0 (± 2.0) | 74.1 (± 2.8) | 57.4 (± 4.9) |56.7 (± 5.2) |
| Ad-Recon-Net |     34.21    | 77.4 (± 4.9) | 73.9 (± 1.3) | 74.3 (± 3.3) | 57.6 (± 6.3) |56.7 (± 5.2) |
|    FGMN      |              | 77.4 (± 4.9) | 73.9 (± 1.3) | 74.3 (± 3.3) | 57.6 (± 6.3) |56.7 (± 5.2) |
|   Full-Conv  |              | 77.4 (± 4.9) | 73.9 (± 1.3) | 74.3 (± 3.3) | 57.6 (± 6.3) |56.7 (± 5.2) |
|    DR2-Net   |     32.40    | 77.4 (± 4.9) | 73.9 (± 1.3) | 74.3 (± 3.3) | 57.6 (± 6.3) |56.7 (± 5.2) |
|   ISTA-Net   |     37.43    | 77.4 (± 4.9) | 73.9 (± 1.3) | 74.3 (± 3.3) | 57.6 (± 6.3) |56.7 (± 5.2) |
| ISTA-NetPlus |     38.07    | 77.4 (± 4.9) | 73.9 (± 1.3) | 74.3 (± 3.3) | 57.6 (± 6.3) |56.7 (± 5.2) |
|   AMP-Net    |     39.52    | 77.4 (± 4.9) | 73.9 (± 1.3) | 74.3 (± 3.3) | 57.6 (± 6.3) |56.7 (± 5.2) |
|   AMPA-Net   |     40.32    | 77.4 (± 4.9) | 73.9 (± 1.3) | 74.3 (± 3.3) | 57.6 (± 6.3) |56.7 (± 5.2) |


## Contact
Email: nanyuli1994@gmail.com


## License
MIT
