
# View-Adaptive-E(2)-Equivariant-Steerable-CNNs-for-Skeleton-based-Human-Action-Recognition

**[Documentation](https://quva-lab.github.io/e2cnn/)** | **[Paper](https://arxiv.org/abs/1911.08251)** | **[Thesis](https://gabri95.github.io/Thesis/thesis.pdf)**

## Introduction
For skeleton-based human action recognition, one of the key challenges is the large view variations when capturing data. In order to alleviate the effects of view variations, we introduce a novel view adaptation scheme, which automatically determines the virtual observation viewpoints in a learning based data driven manner.  

This repository holds the codes and methods for the following paper:

[**View Adaptive Neural Networks for High Performance Skeleton-based Human Action Recognition**](https://arxiv.org/pdf/1804.07453.pdf). TPAMI, 2019.


## Flowchart

![image](https://github.com/hsyntemiz/VA-ES-CNNs-for-Skeleton-based-Human-Action-Recognition/blob/master/image/544proj-va-cnn.png)


Figure 1: Flowchat of the end-to-end view adaptive neural network. It consists of a main classification network and a view adaptation subnetwork. The view adaptation subnetwork automatically determines the virtual observation viewpoints and transforms the skeleton input to representations under the new viewpoints for classification by the main classification network. The entire network is end-toend trained to optimize the classification performance.



## Prerequisites
The code is built with following libraries:
- Python 3.7
- [Anaconda](https://www.anaconda.com/)
- [PyTorch](https://pytorch.org/) 1.7.1
- [e2cnn](https://github.com/QUVA-Lab/e2cnn) 0.1.5 



## Data Preparation

We use the NTU60 RGB+D dataset as an example for description. We need to first dowload the [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) dataset

- Extract the dataset to ./data/ntu/nturgb+d_skeletons/
- Process the data
```bash
 cd ./data/ntu
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```


## Training

```bash
# For CNN-based model with view adaptation module
python  va-cnn.py --model VA --aug 1 --train 1 

# For steer-CNN-based model with view adaptation module
python  va-cnn.py --model VA --aug 1 --train 1 --steer

# For CNN-based model without view adaptation module
python  va-cnn.py --model baseline --aug 1 --train 1




## Testing

```bash
# For CNN-based model with view adaptation module
python  va-cnn.py --model VA --aug 1 --train 0

# For steer-CNN-based model with view adaptation module
python  va-cnn.py --model VA --aug 1 --train 0 --steer

# For CNN-based model without view adaptation module
python  va-cnn.py --model baseline --aug 1 --train 0



## Reference
If you find the papers and repo useful, you can cite the paper: 

View Adaptive Neural Networks for High Performance Skeleton-based Human Action Recognition. TPAMI, 2019.

```


```
Microsoft Open Source Code of Conduct: https://opensource.microsoft.com/codeofconduct

