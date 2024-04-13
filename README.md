<h1 align="center">CS231n: Assignment Solutions</h1>
<p align="center"><b>Convolutional Neural Networks for Visual Recognition</b></p>
<p align="center"><i>Stanford - Spring 2023</i></p>

## About
### Overview
These are my solutions for the **CS231n** course assignments offered by Stanford University (Spring 2023). Inline questions are explained, the code is brief and commented (see examples below).

### Main sources (official)
* [**Course page**](http://cs231n.stanford.edu/index.html)
* [**Assignments**](http://cs231n.stanford.edu/assignments.html)
* [**Lecture notes**](https://cs231n.github.io/)
* **Lecture videos**  
[English (2017)](https://www.youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk) [Korean (DSBA)](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) 

<br>

## Solutions
### Assignment 1
* [Q1](assignment1/knn.ipynb): k-Nearest Neighbor classifier
* [Q2](assignment1/svm.ipynb): Training a Support Vector Machine
* [Q3](assignment1/softmax.ipynb): Implement a Softmax classifier
* [Q4](assignment1/two_layer_net.ipynb): Two-Layer Neural Network
* [Q5](assignment1/features.ipynb): Higher Level Representations: Image Features
### Assignment 2
* [Q1](assignment2/FullyConnectedNets.ipynb): Fully-connected Neural Network
* [Q2](assignment2/BatchNormalization.ipynb): Batch Normalization
* [Q3](assignment2/Dropout.ipynb): Dropout
* [Q4](assignment2/ConvolutionalNetworks.ipynb): Convolutional Networks
* [Q5](assignment2/PyTorch.ipynb) _option 1_: PyTorch on CIFAR-10
* [Q5](assignment2/TensorFlow.ipynb) _option 2_: TensorFlow on CIFAR-10

### Assignment 3
* [Q1](assignment3/RNN_Captioning.ipynb): Image Captioning with Vanilla RNNs
* [Q2](assignment3/Transformer_Captioning.ipynb): Image Captioning with Transformers
* [Q3](assignment3/Network_Visualization.ipynb): Network Visualization: Saliency Maps, Class Visualization, and Fooling Images
* [Q4](assignment3/Generative_Adversarial_Networks.ipynb): Generative Adversarial Networks
* [Q5](assignment3/Self_Supervised_Learning.ipynb): Self-Supervised Learning for Image Classification
* [Q6](assignment3/LSTM_Captioning.ipynb): Image Captioning with LSTMs

<br>

## Running Locally

It is advised to run in [Colab](https://colab.research.google.com/), however, you can also run locally. To do so, first, set up your environment - either through [conda](https://docs.conda.io/en/latest/) or [venv](https://docs.python.org/3/library/venv.html). It is advised to install [PyTorch](https://pytorch.org/get-started/locally/) in advance with GPU acceleration. Then, follow the steps:
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Change every first code cell in `.ipynb` files to:
   ```bash
   %cd cs231n/datasets/
   !bash get_datasets.sh
   %cd ../../
   ```
3. Change the first code cell in section **Fast Layers** in [ConvolutionalNetworks.ipynb](assignment2/ConvolutionalNetworks.ipynb) to:
   ```bash
   %cd cs231n
   !python setup.py build_ext --inplace
   %cd ..
   ```

I've gathered all the requirements for all 3 assignments into one file [requirements.txt](requirements.txt) so there is no need to additionally install the requirements specified under each assignment folder. If you plan to complete [TensorFlow.ipynb](assignment2/TensorFlow.ipynb), then you also need to additionally install [Tensorflow](https://www.tensorflow.org/install).


> **Note**: to use MPS acceleration via Apple M1, see the comment in [#4](https://github.com/mantasu/cs231n/issues/4#issuecomment-1492202538).


## Difference in assignments code (.ipynb)
In basic, assignments work in colab. But I worked in local environment.  

### Example (assignemt1)
**Original code (colab)**
```python
# This mounts your Google Drive to the Colab VM.
from google.colab import drive
drive.mount('/content/drive')

# TODO: Enter the foldername in your Drive where you have saved the unzipped
# assignment folder, e.g. 'cs231n/assignments/assignment2/'
FOLDERNAME = None
assert FOLDERNAME is not None, "[!] Enter the foldername."

# Now that we've mounted your Drive, this ensures that
# the Python interpreter of the Colab VM can load
# python files from within it.
import sys
sys.path.append('/content/drive/My Drive/{}'.format(FOLDERNAME))

# This downloads the CIFAR-10 dataset to your Drive
# if it doesn't already exist.
%cd /content/drive/My\ Drive/$FOLDERNAME/cs231n/datasets/
!bash get_datasets.sh
%cd /content/drive/My\ Drive/$FOLDERNAME
```

**My code (local)**
```python
# This mounts your Google Drive to the Colab VM.
# from google.colab import drive
# drive.mount('/content/drive')

# TODO: Enter the foldername in your Drive where you have saved the unzipped
# assignment folder, e.g. 'cs231n/assignments/assignment1/'
FOLDERNAME = '/cs231n/assignment1_colab/assignment1'
assert FOLDERNAME is not None, "[!] Enter the foldername."

# Now that we've mounted your Drive, this ensures that
# the Python interpreter of the Colab VM can load
# python files from within it.
import sys
sys.path.append('/home/USER/PATH{}'.format(FOLDERNAME))

# This downloads the CIFAR-10 dataset to your Drive
# if it doesn't already exist.
%cd /home/USER/PATH$FOLDERNAME/cs231n/datasets/
!bash get_datasets.sh
%cd /home/USER/PATH$FOLDERNAME
```