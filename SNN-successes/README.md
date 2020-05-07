# Models and architectures built on Self-Normalizing Networks

## GANs
- [THINKING  LIKE  A  MACHINE - GENERATING  VISUAL RATIONALES WITH WASSERSTEIN GANS](https://pdfs.semanticscholar.org/dd4c/23a21b1199f34e5003e26d2171d02ba12d45.pdf): Both discriminator and generator trained without batch normalization.
- [Deformable Deep Convolutional Generative Adversarial Network in Microwave Based Hand Gesture Recognition System](https://arxiv.org/abs/1711.01968):
 The  rate  between  SELU  and  SELU+BN proves  that  SELU  itself  has  the  convergence  quality  of  BN.

## Convolutional neural networks
- [Effectiveness of Self Normalizing Neural Networks for Text Classification](https://arxiv.org/abs/1905.01338): Applied properties of SNN to CNN to give rise to new arechitecture SCNN (Self normalizaing CNN) which performed better than CNN in text classification.
- [Solving internal covariate shift in deep learning with linked neurons](https://arxiv.org/abs/1712.02609): Show that ultra-deep CNNs without batch normalization can only be trained SELUs (except with the suggested method described by the authors).
- [DCASE 2017 ACOUSTIC SCENE CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORK IN TIME SERIES](http://www.cs.tut.fi/sgn/arg/dcase2017/documents/challenge_technical_reports/DCASE2017_Biho_116.pdf): Deep CNN trained without batch normalization.
- [Point-wise Convolutional Neural Network](https://arxiv.org/abs/1712.05245):  Training with SELU converges faster than training with ReLU; improved accuracy with SELU.
- [Over the Air Deep Learning Based Radio Signal Classification](https://arxiv.org/abs/1712.04578): Slight performance improvement over ReLU.
- [Convolutional neural networks for structured omics: OmicsCNN and the OmicsConv layer](https://arxiv.org/abs/1710.05918): Deep CNN trained without batch normalization.
- [Searching for Activation Functions](https://arxiv.org/abs/1710.05941): ResNet architectures trained with SELUs probably together with batch normalization.
- [EddyNet: A Deep Neural Network For Pixel-Wise Classification of Oceanic Eddies](https://arxiv.org/abs/1711.03954): Fast CNN training with SELUs. ReLU with BN better at final performance but skip connections not handled appropriately.
- [SMILES2Vec: An Interpretable General-Purpose Deep Neural Network for Predicting Chemical Properties](https://arxiv.org/abs/1712.02034): 20-layer ResNet trained with SELUs.
- [Sentiment Analysis of Tweets in Malayalam Using Long Short-Term Memory Units and Convolutional Neural Nets](https://link.springer.com/chapter/10.1007/978-3-319-71928-3_31)
- [RETUYT in TASS 2017: Sentiment Analysis for Spanish Tweets using SVM and CNN](https://arxiv.org/abs/1710.06393)


## FNNs are finally deep
- [Predicting Adolescent Suicide Attempts with Neural Networks](https://arxiv.org/abs/1711.10057): The use of the SELU activation renders batch normalization
unnecessary.
- [Improving Palliative Care with Deep Learning](https://arxiv.org/abs/1711.06402): An 18-layer neural network with SELUs performed best.
- [An Iterative Closest Points Approach to Neural Generative Models](https://arxiv.org/abs/1711.06562)
- [Retrieval of Surface Ozone from UV-MFRSR Irradiances using Deep Learning](http://uvb.nrel.colostate.edu/UVB/publications/AGU-Retrieval-Surface-Ozone-Deep-Learning.pdf): 6-10 layer networks perform best. 


## Reinforcement Learning
- [Automated Cloud Provisioning on AWS using Deep Reinforcement Learning](https://arxiv.org/abs/1709.04305): Deep CNN architecture trained with SELUs.
- [Learning to Run with Actor-Critic Ensemble](https://arxiv.org/abs/1712.08987): Second best method (actor-critic ensemble) at the NIPS2017 "Learning to Run" competition. They have
tried several activation functions and found that the activation function of Scaled Exponential Linear Units (SELU) are superior to ReLU, Leaky ReLU, Tanh and Sigmoid.

## Autoencoders
- [Replacement AutoEncoder: A Privacy-Preserving Algorithm for Sensory Data Analysis](https://arxiv.org/abs/1710.06564): Deep autoencoder trained with SELUs.
- [Application of generative autoencoder in de novo molecular design](https://arxiv.org/abs/1711.07839): Faster convergence with SELUs.

## Recurrent Neural Networks
- [Sentiment extraction from Consumer-generated noisy short texts](http://sentic.net/sentire2017meisheri.pdf): SNNs used in FC layers.


