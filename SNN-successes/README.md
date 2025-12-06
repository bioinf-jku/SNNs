# Models and architectures built on Self-Normalizing Networks (SELU / SNN)


## Foundational papers
- [Self-Normalizing Neural Networks (NeurIPS 2017)](https://arxiv.org/abs/1706.02515):
  Introduces SELU, AlphaDropout, and the self-normalizing theory enabling deep FNNs
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
  Their OT-map MLP concatenates time and **uses selu activations**.


### Flow Matching / Schrödinger Bridges
- [Improving and generalizing flow-based generative models with minibatch optimal transport](https://arxiv.org/abs/2302.00482) Schrödinger bridge model is a self-normalizing MLP with SELU.
- [Simulation-Free Schrödinger Bridges via Score and Flow Matching (AISTATS 2024)](https://proceedings.mlr.press/v238/tong24a/tong24a.pdf):
  Uses 3-layer MLPs with SELU activations in vector-field/score networks.
- [Meta Flow Matching (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/ebdb990471f653dffb425eff03c7c980-Paper-Conference.pdf):
  Uses multi-layer MLPs with SELU activations for synthetic and biological experiments.
- [Source-Guided Flow Matching (2025)](https://arxiv.org/abs/2508.14807):
  Uses SELU-MLP vector fields with smoothing priors.
- [Explicit Flow Matching: On the theory of ... (2024)](https://openreview.net/pdf/fea78f8c3ee7a53805dea330c18827a6d754ad33.pdf):  
  Evaluation uses a **3-layer MLP with SeLU activations** and hidden dim 64.
- [Flows Don’t Cross in High Dimension (2025)](https://openreview.net/pdf?id=nK9TmlJu8F):  
  Neural vector-field experiments use a **3-layer MLP, hidden dim 64, with SELU**.
- [OAT-FM: Optimal Acceleration Transport for Improved Flow Matching (2025)](https://arxiv.org/html/2509.24936v1):  
  Low-dimensional OT/CFM benchmarks parameterize the vector field with a **3-hidden-layer MLP (width 64) + SELU**.
- [ParetoFlow: Guided Flows in Multi-Objective Optimization (2024)](https://arxiv.org/html/2412.03718v1):  
  Uses a **multi-layer MLP with SeLU activations** following flow-matching training protocols.
- [TorchCFM library](https://github.com/atong01/conditional-flow-matching):  
  Example configs reference **3×64 MLPs with SeLU** in action/CFM demos.

### Multi-marginal / irregular-time dynamics
- [Multi-Marginal Stochastic Flow Matching for High-Dimensional Snapshot Data at Irregular Time Points (ICML 2025)](https://arxiv.org/html/2508.04351v1):  
  Training setup uses **MLPs with two hidden layers of width 64 and SELU activations** for most non-image experiments.
- [Multi-Marginal Flow Matching with Adversarially Learnt Interpolants (2025)](https://arxiv.org/html/2510.01159v1):  
  **CFM nets** in the cell-tracking experiment are **3-hidden-layer MLPs with 256 units per layer and SELU**; also notes **3-layer SELU MLPs** used for vector fields in single-cell setups.
- [Dynamic Conditional Optimal Transport through Simulation-Free Flows (2024)](https://www.proceedings.com/content/079/079017-2968open.pdf):  
  For FM and their COT-FM variant, the model architecture is an **MLP with SeLU activations**.


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
- [SELU-transformer for hoax/news classification (2020–2024 line of work)](https://www.semanticscholar.org/paper/BET%3A-A-Backtranslation-Approach-for-Easy-Data-in-Corbeil-Ghadivel/84d2d7f7fa2f656db458a5c369d0aa35ddb60f5e):
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
- [Foundation Inference Models for Stochastic Differential Equations (2025)](https://arxiv.org/abs/2502.19049):
  Follow-up foundation-style work in the same line (check architecture details for continued SELU use).
- [Towards Foundation Inference Models that Learn ODEs In-Context (2025)](https://arxiv.org/abs/2510.12650):
  Extends the FIM line toward in-context ODE learning.

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
  Deep CNN + DQN-style setup using **SELU** in the network architecture.
- [Learning to Run with Actor-Critic Ensemble (2017)](https://arxiv.org/abs/1712.08987):  
  Reports testing multiple activations and finding **SELU** superior; uses **SELU** in actor/critic FC layers.
- [Multi-Agent Trust Region Policy Optimization (MATRPO) (2020)](https://arxiv.org/abs/2010.07916):  
  Policies/critics use **two hidden layers of 128 SeLU units** in their experiments.
- [Application of Deep Q-Network in Portfolio Management (2020)](https://arxiv.org/abs/2003.06365):  
  Uses **SELU** in conv layers (argues negative-valued signals matter for this input type).
- [Latent-Conditioned Policy Gradient for Multi-Objective RL (2023)](https://arxiv.org/abs/2303.08909):  
  Uses **SELU** for most non-output activations in policy/value networks.
- [Intelligent Energy Pairing Scheduler (InEPS) for Heterogeneous HPC Clusters (2025)](https://link.springer.com/article/10.1007/s11227-024-06907-y):  
  Uses **SELU** between actor/critic layers in a PPO-style scheduling system.
- [Quantum compiling by deep reinforcement learning (2021)](https://arxiv.org/abs/2105.15048):  
  Cross-domain RL application that may use **SELU** in parts of the network
- [Pearl: Automatic Code Optimization Using Deep Reinforcement Learning (2025)](https://arxiv.org/abs/2506.01880): Uses **PPO** with a GNN backbone and **SELU** in the MLP before the policy/value heads.


