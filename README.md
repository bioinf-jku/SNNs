# Self-Normalizing Networks
Tutorials and implementations for "Self-normalizing networks"(SNNs) as suggested by Klambauer et al. ([arXiv pre-print](https://arxiv.org/pdf/1706.02515.pdf)). 

## Versions
- see [environment](environment.yml) file for full list of prerequisites. Tutorial implementations use Tensorflow > 2.0 (Keras) or Pytorch, but versions for Tensorflow 1.x 
  users based on the deprecated tf.contrib module (with separate [environment](TF_1_x/environment.yml) file) are also available.

#### Note for Tensorflow >= 1.4 users
Tensorflow >= 1.4 already has the function [tf.nn.selu](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/selu) and [tf.contrib.nn.alpha_dropout](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/contrib/nn/alpha_dropout) that implement the SELU activation function and the suggested dropout version. 
#### Note for Tensorflow >= 2.0 users
Tensorflow 2.3 already has selu activation function when using high level framework keras, [tf.keras.activations.selu](https://www.tensorflow.org/api_docs/python/tf/keras/activations/selu). 
Must be combined with [tf.keras.initializers.LecunNormal](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/LecunNormal), corresponding dropout version is [tf.keras.layers.AlphaDropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/AlphaDropout).
#### Note for Pytorch users
Pytorch versions >= 0.2 feature [torch.nn.SELU](https://pytorch.org/docs/stable/generated/torch.nn.SELU.html#torch.nn.SELU) and [torch.nn.AlphaDropout](https://pytorch.org/docs/stable/generated/torch.nn.AlphaDropout.html#torch.nn.AlphaDropout), they must be combined with the correct initializer, namely [torch.nn.init.kaiming_normal_](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_) (parameter, mode='fan_in', nonlinearity='linear') 
as this is identical to lecun initialisation (mode='fan_in') with a gain of 1 (nonlinearity='linear'). 


## Tutorials

### Tensorflow 1.x 
- Multilayer Perceptron on MNIST ([notebook](TF_1_x/SelfNormalizingNetworks_MLP_MNIST.ipynb))
- Convolutional Neural Network on MNIST ([notebook](TF_1_x/SelfNormalizingNetworks_CNN_MNIST.ipynb))
- Convolutional Neural Network on CIFAR10 ([notebook](TF_1_x/SelfNormalizingNetworks_CNN_CIFAR10.ipynb))

### Tensorflow 2.x (Keras)
- Multilayer Perceptron on MNIST ([python script](TF_2_x/MNIST-MLP-SELU.py))
- Convolutional Neural Network on MNIST ([python script](TF_2_x/MNIST-Conv-SELU.py))
- Convolutional Neural Network on CIFAR10 ([python script](TF_2_x/CIFAR10-Conv-SELU.py))

### Pytorch

- Multilayer Perceptron on MNIST ([notebook](Pytorch/SelfNormalizingNetworks_MLP_MNIST.ipynb))
- Convolutional Neural Network on MNIST ([notebook](Pytorch/SelfNormalizingNetworks_CNN_MNIST.ipynb))
- Convolutional Neural Network on CIFAR10 ([notebook](Pytorch/SelfNormalizingNetworks_CNN_CIFAR10.ipynb))

## Further material

### Design novel SELU functions (Tensorflow 1.x)
- How to obtain the SELU parameters alpha and lambda for arbitrary fixed points ([notebook](TF_1_x/getSELUparameters.ipynb))

### Basic python functions to implement SNNs (Tensorflow 1.x)
are provided as code chunks here: [selu.py](TF_1_x/selu.py)

### Notebooks and code to produce Figure 1 (Tensorflow 1.x)
are provided here: [Figure1](figure1/), builds on top of the [biutils](https://github.com/untom/biutils) package.

### Calculations and numeric checks of the theorems (Mathematica)
are provided as mathematica notebooks here:

- [Mathematica notebook](Calculations/SELU_calculations.nb)
- [Mathematica PDF](Calculations/SELU_calculations.pdf)

### UCI, Tox21 and HTRU2 data sets

- [UCI](http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz)
- [Tox21](http://bioinf.jku.at/research/DeepTox/tox21.zip)
- [HTRU2](https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip)
