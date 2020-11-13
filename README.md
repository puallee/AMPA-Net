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
|   BM3D-AMP   |     34.11    |     32.79    |  27.87| 22.12 | 17.32 |4.91|
|   LD-AMP     |     35.92    |33.56  | 28.46        | 22.64 | 18.40 |5.21 |
| Ad-Recon-Net |     34.21    | 32.72 | 30.80        | 27.53 | 23.22 |20.33 |
|    FGMN      |              |  |                   |  | 23.87 |21.27 |
|   Full-Conv  |              | | 32.69              | 28.30 |  |21.27 |
|    DR2-Net   |     32.40    | 31.20 | 28.66        | 24.71 | 20.08 |17.44|
|   ISTA-Net   |     37.43    | 35.36 | 31.53        | 25.80 | 21.23 |17.30 |
| ISTA-NetPlus |     38.07    | 36.06 | 32.57        | 26.64 | 21.31 |17.34|
|   AMP-Net    |     39.52    | 37.13 | 33.60        | 28.47 | 24.21 |20.48 |
|   AMPA-Net   |     40.32    | 38.27 | 34.61        | 29.30 | 24.95 |21.59 |

*  performance on Set11
 |  Model name  |     cs=50%   |    cs=40%    |    cs=25%    |   cs=10%     |     cs=4%    |    cs=1%    |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |------------ |
|   BM3D-AMP   |     34.11    |     32.79    |  27.87| 22.12 | 17.32 |4.91|
|   LD-AMP     |     35.92    |33.56  | 28.46        | 22.64 | 18.40 |5.21 |
| Ad-Recon-Net |     34.21    | 32.72 | 30.80        | 27.53 | 23.22 |20.33 |
|    FGMN      |              |  |                   |  | 23.87 |21.27 |
|   Full-Conv  |              | | 32.69              | 28.30 |  |21.27 |
|    DR2-Net   |     32.40    | 31.20 | 28.66        | 24.71 | 20.08 |17.44|
|   ISTA-Net   |     37.43    | 35.36 | 31.53        | 25.80 | 21.23 |17.30 |
| ISTA-NetPlus |     38.07    | 36.06 | 32.57        | 26.64 | 21.31 |17.34|
|   AMP-Net    |     39.52    | 37.13 | 33.60        | 28.47 | 24.21 |20.48 |
|   AMPA-Net   |     40.32    | 38.27 | 34.61        | 29.30 | 24.95 |21.59 |
 * performance on Set68


## Contact
Email: nanyuli1994@gmail.com


## License
MIT
