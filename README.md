# AMPA-Net
This is the official TensorFlow implementation of AMPA-Net
[Nanyu Li, Charles C. Zhou, AMPA-Net： Optimization-Inspired Attention Neural Network for Deep Compressed Sensing， In the IEEE 20th International Conference on Communication Technology (ICCT) oral]
which can be downloaded from https://arxiv.org/abs/2010.06907
## Abstract
Compressed sensing (CS) is a challenging problem in image processing due to reconstructing an almost complete image from a limited measurement. To achieve fast and accurate CS reconstruction, we synthesize the advantages of two well-known methods (neural network and optimization algorithm) to propose a novel optimization-inspired neural network which dubbed AMP-Net. AMP-Net realizes the fusion of the Approximate Message Passing (AMP) algorithm and neural network. All of its parameters are learned automatically. Furthermore, we propose an AMPA-Net which uses three attention networks to improve the representation ability of AMP-Net. Finally, We demonstrate the effectiveness of AMP-Net and AMPA-Net on four standard CS reconstruction benchmark data sets.

## Compatibility
* The code is tested using Tensorflow 1.0 under Ubuntu 16.04 with Python 2.7.

* Recommend Environment: Anaconda

## Preparing Training Datasets-Image-91
   Download  Training_Data_Img91.mat from：https://pan.baidu.com/s/1UgRuDbIXCNZOEuedVlK8bA?pwd=cq1b code is cq1b and cp it into this folder
## Preparing Testing Datasets:Urban-100 and BSDS100
   Download  Urban-100 from：https://pan.baidu.com/s/1AUNWfVx8Jy12D12yJClWYg?pwd=rr3q  code is rr3q  and cp it into this folder
   
   Download  BSDS-100 from：https://pan.baidu.com/s/1KwpRYlp_7SjphA_Vc2LMiA?pwd=kbyw code is kbyw and cp it into this folder
## Running training and testing
   Python trainn.py 

## Results

Our model achieves the following performance on Set11, BSD68, BSDS100, Urban100

## performance on Set11
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


## performance on Set68                                              

 |  Model name  |     cs=50%   |    cs=40%    |    cs=25%    |   cs=10%     |     cs=4%    |    cs=1%    |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |------------ |
|   LD-AMP     |     31.22    |30.12  | 22.79        | 20.11 | 12.02 |3.50 |
| Ad-Recon-Net |     29.94    | 28.82 | 25.02        | 25.45 | 22.28 |19.68 |
| ISTA-NetPlus |     34.01    | 32.21 | 29.21        | 25.33 | 22.17 |19.50|
|   AMP-Net    |     35.02    | 33.10 | 30.25        | 26.92 | 23.77 |20.85 |
|   AMPA-Net   |     36.33    | 34.41 | 31.38        | 27.58 | 24.90 |21.99 |

## performance on BSDS100                                              

 |  Model name  |     cs=50%   |    cs=40%    |    cs=25%    |   cs=10%     |     cs=4%    |    cs=1%    |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |------------ |
|   LD-AMP     |     30.81    |29.57 | 22.46        | 19.64 | 10.40 |3.21 |
| Ad-Recon-Net |     29.21    | 28.12 | 24.80        | 25.13 | 22.22 |19.35 |
| ISTA-NetPlus |     33.64    | 31.83 | 29.00        | 25.08 | 22.10 |19.17|
|   AMP-Net    |     34.21    | 32.13 | 29.59        | 26.87 | 23.21 |19.48|
|   AMPA-Net   |     35.95    | 34.03 | 31.01        | 27.29 | 24.75 |21.62 |

## performance on Urban-100                                              

 |  Model name  |     cs=50%   |    cs=40%    |    cs=25%    |   cs=10%     |     cs=4%    |    cs=1%    |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |------------ |
|   LD-AMP     |     30.41    |29.12 | 22.02       | 17.14 | 8.42 |1.31 |
| Ad-Recon-Net |     29.15    | 27.90 | 24.20        | 23.13 | 19.22 |16.82 |
| ISTA-NetPlus |     33.94    | 31.96 | 28.32        | 23.44 | 19.41 |16.47|
|   AMP-Net    |     34.08    | 31.95 | 29.02        | 26.20 | 20.01 |16.88|
|   AMPA-Net   |     35.86    | 33.92 | 30.49        | 25.76 | 22.40 |18.86 |



## Contact
Email: nanyuli1994@gmail.com


## License
MIT
