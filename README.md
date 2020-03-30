# LTHP

**Learnable Triangulation of Human Pose(ICCV 2019)** - [Paper](https://arxiv.org/pdf/1905.05754v1.pdf) | [official code](https://github.com/karfly/learnable-triangulation-pytorch)

This repository was created for implementation tests (**Naver Clova**). (20.03.20)

Pytorch 1.0.1, Python 3.7.5

## Data

### Dataset : [Human3.6M](http://vision.imar.ro/human3.6m/description.php)

I haven't downloaded the dataset because they haven't given me a license yet after signing up.

(On the website, a message called 'Account needs manual verification from the administration. Please be patient.')

### Preprocessing

If the dataset is downloaded, the data preprocessing assumes that it follows [this repository](https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/datasets/human36m_preprocessing/README.md).

## Network
- 2D backbone Network
- Algebraic Triangulation Network
- Volumetric Triangulation Network

## Train
Since data cannot be downloaded, it is implemented except for dataset and dataloader part.

In the [test.ipynb](https://github.com/jaywican/LTHP/blob/master/test.ipynb) file, you can view the output shape of the network through the random tensor.
