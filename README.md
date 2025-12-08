<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>


# The Great Comeback of Self-Normalizing Networks in 2025
## *Günter Klambauer*

It has been a wild year in AI and especially for self-normalizing networks and SELU activations!

- [Normalization-Free Transformers](https://arxiv.org/abs/2503.10622) have re-discovered controlled signal propagation.
- SELU is the default in [Conditional Flow Matching works](https://arxiv.org/abs/2302.00482) -- **the 3×64 baseline phenomenon**.
- Time-Series Foundation Models adopt SELU in their architectures ([FIM/FIM-ℓ](https://openreview.net/forum?id=NPSZ7V1CCY), [Flowstate](https://arxiv.org/abs/2508.05287)).
- The [SELU-transformer](https://ieeexplore.ieee.org/abstract/document/10957007?casa_token=TgFEKMg4iUkAAAAA:utZNYo98h6-_FVKzJwdPlLUEiK-reLFE802g8X1IuRuAbDkl0JEB8-2hTNC6ZisxVpWYWQ7HP1zt) made a **resurgence** in specialized NLP and tabular domains, as [TabTranSELU](https://doi.org/10.54254/2755-2721/51/20241174) **kicking out that darn SwiGLU module**.
- RL systems are using SELU for stability in [PPO](https://arxiv.org/abs/2506.01880), e.g. for code optimization.
- Graph Convolutional Networks (GCN) have been using SELU activations since [DMoN](https://arxiv.org/abs/2006.16904). In 2025, methods like [GyralNet](https://arxiv.org/abs/2503.19823), use the design as standard.
- [AI systems in drug discovery still dominated by SNNs](https://arxiv.org/abs/2511.14744) -- AI is hitting a wall in drug discovery: No progress at toxicity/activity prediction for molecules; **self-normalizing networks** still in the [lead](https://huggingface.co/spaces/ml-jku/tox21_leaderboard) aka leaderboard :)


### Normalization-free Transformers. Will we get LLMs without normalization layers?
In March 2025, I saw ["Transformers without normalization"](https://arxiv.org/abs/2503.10622) by Yann LeCun and colleagues drop on arxiv. I thought "now they finally have it", because Yann has been thinking in similar directions as I did already back in his "Efficient Backprop" tutorial. After all, self-normalizing networks require the initialization named after him ("LeCun's initialization"). Strangely it's just a scaled tanh-activation that does the job.. ok!

### The 3x64 baseline phenomenon in conditional flow matching and Schrödinger bridges
Then we got all this nice work on **conditional flow matching**: here 2 or 3 layer SELU-networks with a width of 64 have quasi become standard since Alex Tong's work and [implementation in torch CMF](https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/models/models.py). Here the SELU-network represents the derivative of another function -- this is where the smoothness of SELU networks, i.e. smooth derivative of the other function, is clearly the improvement over ReLU networks.

### Time-series foundation models rely on SELU
2025 was clearly the year of **time-series foundation models** and I am very happy that we had a part in this. Clearly our [TiRex](https://arxiv.org/abs/2505.23719) taking the lead in the [GIFT Eval leaderboard](https://huggingface.co/spaces/Salesforce/GIFT-Eval) (ahead of Amazon's Chronos) was one of my favourite moments in 2025. However, the other foundation models, like FIM/FIM-ℓ and [Flowstate](https://arxiv.org/abs/2508.05287), they all use SELU activations.

### RL systems use SELU for stability
One of the quiet but undeniable trends of 2025 is the **return of SELU in reinforcement learning**.
Across several independent lines of work, researchers rediscovered what we already saw in 2017 during the **Learning-to-Run** challenge: 7
actor–critic methods become meaningfully **more stable** when the policy/value heads use SELU instead of ReLU.
This year, the evidence became impossible to ignore: In **PPO-based code optimization**, the *Pearl* system (2025) [uses SELU inside its actor–critic MLPs](https://arxiv.org/abs/2506.01880) and reports substantially smoother training dynamics during policy updates. HPC scheduling frameworks such as [InEPS](https://link.springer.com/article/10.1007/s11227-024-06907-y) apply SELU in their PPO actor–critic networks to tame exploding/vanishing activations caused by heterogeneous inputs and reward signals. Multi-objective RL, e.g., [latent-conditioned policy gradient methods](https://link.springer.com/chapter/10.1007/978-3-031-44223-0_6) increasingly defaults to **SELU for all hidden layers**, because it simply behaves more predictably under policy-gradient noise. I think the pattern is the following: Whenever RL systems avoid batch normalization (which they usually want or must), SELU becomes one of the most stable activations for deep value functions and stochastic policies. 


### Graph convolutional networks consistently replace ReLU with SELU for better convergence and robustness
A growing line of **graph clustering** ([DMoN](https://arxiv.org/abs/2006.16904), [DGCLUSTER](https://arxiv.org/abs/2312.12697), [MetaGC](https://doi.org/10.1145/3583780.3615038), [Potts-GNN](https://arxiv.org/abs/2308.09644)) and **privacy-preserving GNN** work ([LPGNN](https://arxiv.org/abs/2006.05535), GAP, UPGNET) consistently replaces ReLU with SELU and reports better convergence or robustness. 
Since [DMoN](https://arxiv.org/abs/2006.16904), GCN use the following forward propagation:

$$
H^{(l+1)} = \mathrm{SELU} \left( \tilde{A} H^{(l)} W^{(l)}  + H^{(l)} W_{\text{skip}}^{(l)} \right)
$$

where \\(\tilde{A}\\) is the normalized adjacency matrix. Classic [GCN](https://arxiv.org/abs/1609.02907) used \\(H^{(l+1)} = \sigma \left( \tilde{A}  H^{(l)} W^{(l)} \right)\\) with sigmoid or ReLU activation. While the full SNN theory doesn’t directly apply to message-passing, a shallow GNN layer is still “linear aggregation + nonlinearity,” and SELU’s self-normalizing behavior seems to provide more stable training in normalization-free, noisy, or shallow GNN settings. 


### AI is hitting a wall in drug discovery, a wall built of SELUs
We've re-evaluated machine-learning and deep learning methods from the last 25 years at the Tox21 Data Challenge dataset. Ok, LLMs can do this at least a bit -- but far off any reasonable performance. Recent methods like GNNs are a bit behing state-of-the-art, but we were actually extremely suprised that the SELU-networks from 2017 still perform best on this [Tox21 leaderboard](https://huggingface.co/spaces/ml-jku/tox21_leaderboard). People were wondering why AI hasn't found a new drug yet, nor has improved drug discovery a lot.. yeah, this might be a hint. Deep Learning methods are good at DESIGNING molecules, and are brilliant at MAKING them (in the sense of predicting chemical synthesis routes), but AI systems are obviously BAD AT TESTING those molecules. By TESTING, i mean virtually testing them by predicting their biological properties, such as toxic effects. Well, suprisingly we have to improve the TEST in the DESIGN-MAKE-TEST-ANALYSE cycle.


# Papers, models and architectures built on Self-Normalizing Networks (SELU / SNN)

## Foundational papers
- [Self-Normalizing Neural Networks (NeurIPS 2017)](https://arxiv.org/abs/1706.02515):
  Introduces SELU, AlphaDropout, and the self-normalizing theory enabling deep multi-layer perceptrons (MLPs)
  without explicit normalization.
- [Bidirectionally self-normalizing neural networks. Neural Networks, 167, 283-291](https://www.sciencedirect.com/science/article/abs/pii/S0893608023004367)
  Extends SNNs to both forward and backward pass by introducing shift parameters in activation function

## Generative Models

### Diffusion & Score-based models
- [SE(3)-Equivariant Diffusion Graph Nets](https://openreview.net/pdf/37fd86fe050a82aa6d9d9a308685520622b3fcc4.pdf):
  Uses SELU activations in MLPs inside the diffusion-graph pipeline for fluid flow fields.
- [Radar emitter denoising with DDPM variants (2024)](https://www.mdpi.com/2072-4292/16/17/3215):
  Employs SELU in 1D conv blocks within a DDPM-style architecture.
- [DIDiffGes: Decoupled Semi-Implicit Diffusion Models for Gesture Generation (2025)](https://arxiv.org/abs/2503.17059):
  Uses MLP blocks with SELU in a diffusion+GAN hybrid for fast sampling.
- [Learning of Population Dynamics: Inverse Optimization Meets JKO Scheme (2025)](https://arxiv.org/html/2506.01502v1):  
  Their OT-map MLP concatenates time and uses SELU activations.


### Flow Matching / Schrödinger Bridges
- [Improving and generalizing flow-based generative models with minibatch optimal transport](https://arxiv.org/abs/2302.00482) Schrödinger bridge model is a self-normalizing MLP with SELU.
- [Simulation-Free Schrödinger Bridges via Score and Flow Matching (AISTATS 2024)](https://proceedings.mlr.press/v238/tong24a/tong24a.pdf):
  Uses 3-layer MLPs with SELU activations in vector-field/score networks.
- [Meta Flow Matching (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/ebdb990471f653dffb425eff03c7c980-Paper-Conference.pdf):
  Uses multi-layer MLPs with SELU activations for synthetic and biological experiments.
- [Source-Guided Flow Matching (2025)](https://arxiv.org/abs/2508.14807):
  Uses SELU-MLP vector fields with smoothing priors.
- [Explicit Flow Matching: On the theory of ... (2024)](https://openreview.net/pdf/fea78f8c3ee7a53805dea330c18827a6d754ad33.pdf):  
  Evaluation uses a 3-layer MLP with SeLU activations and hidden dim 64.
- [Flows Don’t Cross in High Dimension (2025)](https://openreview.net/pdf?id=nK9TmlJu8F):  
  Neural vector-field experiments use a 3-layer MLP, hidden dim 64, with SELU.
- [OAT-FM: Optimal Acceleration Transport for Improved Flow Matching (2025)](https://arxiv.org/html/2509.24936v1):  
  Low-dimensional OT/CFM benchmarks parameterize the vector field with a 3-hidden-layer MLP (width 64) + SELU.
- [ParetoFlow: Guided Flows in Multi-Objective Optimization (2024)](https://arxiv.org/html/2412.03718v1):  
  Uses a multi-layer MLP with SeLU activations following flow-matching training protocols.
- [TorchCFM library](https://github.com/atong01/conditional-flow-matching):  
  Example configs reference 3×64 MLPs with SeLU** in action/CFM demos.

### Multi-marginal / irregular-time dynamics
- [Multi-Marginal Stochastic Flow Matching for High-Dimensional Snapshot Data at Irregular Time Points (ICML 2025)](https://arxiv.org/html/2508.04351v1):
  Training setup uses **MLPs with two hidden layers of width 64 and SELU activations** for most non-image experiments.
- [Multi-Marginal Flow Matching with Adversarially Learnt Interpolants (2025)](https://arxiv.org/html/2510.01159v1):
  **CFM nets** in the cell-tracking experiment are 3-hidden-layer MLPs with 256 units per layer and SELU; also notes 3-layer SELU MLPs used for vector fields in single-cell setups.
- [Dynamic Conditional Optimal Transport through Simulation-Free Flows (2024)](https://www.proceedings.com/content/079/079017-2968open.pdf):
  For FM and their COT-FM variant, the model architecture is an MLP with SeLU activations.


### Normalizing Flows
- [Contextual Movement Models Based on Normalizing Flows (2021)](https://ieeexplore.ieee.org/document/9441650):
  Uses SELU in MLP components for movement modeling.
- [Individual Survival Curves with Conditional Normalizing Flows (2021)](https://arxiv.org/abs/2107.12825):
  Uses SELU as activation across datasets in a CNF-based survival modeling setting.

### Autoencoders & VAEs
- [Training Deep AutoEncoders for Collaborative Filtering (2017/2018)](https://arxiv.org/abs/1708.01715):
  Practical deep AE results with SELU among competitive activations.
- [Application of generative autoencoder in de novo molecular design (2017)](https://arxiv.org/abs/1711.07839):
  Reports faster convergence with SELU in molecular generation pipelines.

### GANs
- [A Practical Approach for Training Deep Convolutional GANs with SELU Activation (2019)](https://dl.acm.org/doi/10.1145/3313181.3313187):
  Empirical study suggesting SELU can help stabilize or accelerate GAN training in specific setups.
- [THINKING LIKE A MACHINE - Generating Visual Rationales with Wasserstein GANs (2017)](https://pdfs.semanticscholar.org/dd4c/23a21b1199f34e5003e26d2171d02ba12d45.pdf):
  Uses SELU-style setups to reduce reliance on batch normalization.


## MLPs / Tabular / Scientific MLPs
- [Improving Palliative Care with Deep Learning (2017)](https://arxiv.org/abs/1711.06402):
  Deep FNN with SELUs performed best in their comparisons.
- [Training Deep AutoEncoders for Collaborative Filtering (2017/2018)](https://arxiv.org/abs/1708.01715):
  Observes deep autoencoders benefit from activation choices including SELU.


## Transformers & LLM-adjacent models
- [TMRN-GLU: Transformer-Based Automatic Classification Recognition Network (2022)](https://www.mdpi.com/2079-9292/11/10/1554):
  Chooses SELU where activations are needed, citing stability across SNR conditions.
- [SELU-transformer](https://ieeexplore.ieee.org/abstract/document/10957007?casa_token=TgFEKMg4iUkAAAAA:utZNYo98h6-_FVKzJwdPlLUEiK-reLFE802g8X1IuRuAbDkl0JEB8-2hTNC6ZisxVpWYWQ7HP1zt):
  Reports strong text classification performance with SELU-Transformer variants.
- [TabTranSELU: A transformer adaptation for solving tabular data (2024)](https://pdfs.semanticscholar.org/885c/822b8d71d5edadb874dafdcb67f36ce49d11.pdf):
  Replaces a normalization+ReLU pattern with SELU for tabular transformer stability.
- [BERTSurv (clinical outcomes)](https://cse.cs.ucsb.edu/sites/default/files/publications/bertsurv-bert_based_survival_models_for_predicting_outcomes_for_trauma_patients.pdf):
  Uses ReLU or SELU in downstream components alongside BERT.

## Graph Neural Networks (GNNs)
- [Graph Clustering with Graph Neural Networks (DMoN, JMLR 2023)](https://jmlr.org/papers/volume24/20-998/20-998.pdf):
  Uses SELU inside a modified GCN and explicitly replaces ReLU with SELU for better convergence.
- [Locally Private Graph Neural Networks (CCS 2021)](https://www.idiap.ch/~gatica/publications/SajadmaneshGatica-ccs21.pdf):
  Uses SELU in GCN/GAT/GraphSAGE backbones (two graph conv layers with SELU + dropout).
- [Reducing Oversmoothing in Graph Neural Networks by Activation Design (ICLR 2023 submission)](https://openreview.net/pdf?id=8CDeu0f4i2):
  Compares SELU-enhanced GCN/GAT variants and discusses depth/oversmoothing behavior.
- [Deep Probabilistic Dual Graph Convolution Network (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2b6d0f44e1c1f9f1b3c4b1b4d4d66c0-Abstract-Conference.html):
  Uses SELU in its GCN stack.

## Time series & Foundation(-style) models
- [Zero-shot Imputation with Foundation Inference Models for Dynamical Systems (ICLR 2025)](https://arxiv.org/abs/2402.07594):
  Uses SELU as activation for all feed-forward networks in FIM/FIM-ℓ modules.
- [FlowState: Sampling Rate Invariant Time Series Forecasting](https://arxiv.org/abs/2508.05287): main layer uses SELU.

## Convolutional Neural Networks
- [Solving internal covariate shift in deep learning with linked neurons (2017)](https://arxiv.org/abs/1712.02609):
  Shows ultra-deep CNN training issues without BN and highlights SELU-based solutions.
- [Point-wise Convolutional Neural Network (2017)](https://arxiv.org/abs/1712.05245):
  Reports faster convergence with SELU vs ReLU in point-wise CNNs.
- [Effectiveness of Self Normalizing Neural Networks for Text Classification (2019)](https://arxiv.org/abs/1905.01338):
  Applies SELU/SNN ideas to CNN-based text classification.

## Recurrent / Sequence models (non-Transformer)
- [Learning to Run with Actor-Critic Ensemble (NIPS 2017 challenge report)](https://arxiv.org/abs/1712.08987):
  Reports SELU outperforming several activations in their RL ensemble.
- [Sentiment extraction from Consumer-generated noisy short texts (2017)](http://sentic.net/sentire2017meisheri.pdf):
  Uses SELU in feed-forward components.

## Reinforcement Learning: improved stability of RL systems
- [Automated Cloud Provisioning on AWS using Deep Reinforcement Learning (2017)](https://arxiv.org/abs/1709.04305):
  Deep CNN + DQN-style setup using SELU in the network architecture.
- [Learning to Run with Actor-Critic Ensemble (2017)](https://arxiv.org/abs/1712.08987):
  Reports testing multiple activations and finding SELU superior; uses SELU in actor/critic FC layers.
- [Multi-Agent Trust Region Policy Optimization (MATRPO) (2020)](https://arxiv.org/abs/2010.07916): 
  Policies/critics use two hidden layers of 128 SeLU units in their experiments.
- [Application of Deep Q-Network in Portfolio Management (2020)](https://arxiv.org/abs/2003.06365):
  Uses SELU in conv layers (argues negative-valued signals matter for this input type).
- [Latent-Conditioned Policy Gradient for Multi-Objective RL (2023)](https://arxiv.org/abs/2303.08909):
  Uses SELU for most non-output activations in policy/value networks.
- [Intelligent Energy Pairing Scheduler (InEPS) for Heterogeneous HPC Clusters (2025)](https://link.springer.com/article/10.1007/s11227-024-06907-y):
  Uses SELU between actor/critic layers in a PPO-style scheduling system.
- [Quantum compiling by deep reinforcement learning (2021)](https://arxiv.org/abs/2105.15048):
  Cross-domain RL application that may use SELU in parts of the network
- [Pearl: Automatic Code Optimization Using Deep Reinforcement Learning (2025)](https://arxiv.org/abs/2506.01880):
  Uses PPO with a GNN backbone and SELU in the MLP before the policy/value heads.




# Tutorials and implementations for "Self-normalizing networks"(SNNs) as suggested by Klambauer et al. ([arXiv pre-print](https://arxiv.org/pdf/1706.02515.pdf)). 

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
