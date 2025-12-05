# Models and architectures built on Self-Normalizing Networks (SELU / SNN)
A curated list of models (2017–2025) that explicitly use **SELU** and/or the
**Self-Normalizing Neural Networks** design principles (SELU + AlphaDropout +
LeCun normal init, avoiding BN where appropriate).

## Foundational papers
- [Self-Normalizing Neural Networks (NeurIPS 2017)](https://arxiv.org/abs/1706.02515):
  Introduces SELU, AlphaDropout, and the self-normalizing theory enabling deep FNNs
  without explicit normalization.

## Convolutional Neural Networks
- [Solving internal covariate shift in deep learning with linked neurons (2017)](https://arxiv.org/abs/1712.02609):
  Shows ultra-deep CNN training issues without BN and highlights SELU-based solutions.
- [Point-wise Convolutional Neural Network (2017)](https://arxiv.org/abs/1712.05245):
  Reports faster convergence with SELU vs ReLU in point-wise CNNs.
- [Effectiveness of Self Normalizing Neural Networks for Text Classification (2019)](https://arxiv.org/abs/1905.01338):
  Applies SELU/SNN ideas to CNN-based text classification.

## MLPs / Tabular / Scientific MLPs
- [Improving Palliative Care with Deep Learning (2017)](https://arxiv.org/abs/1711.06402):
  Deep FNN with SELUs performed best in their comparisons.
- [Training Deep AutoEncoders for Collaborative Filtering (2017/2018)](https://arxiv.org/abs/1708.01715):
  Observes deep autoencoders benefit from activation choices including SELU.
- [Training deep autoencoders for collaborative filtering (GTC 2018 talk)](https://on-demand.gputechconf.com/gtc/2018/presentation/s8212-training-deep-autoencoders-for-collaborative-filtering.pdf):
  Practical large-scale recommender success with deep AEs, highlighting SELU-family
  activations enabling depth.

## Recurrent / Sequence models (non-Transformer)
- [Learning to Run with Actor-Critic Ensemble (NIPS 2017 challenge report)](https://arxiv.org/abs/1712.08987):
  Reports SELU outperforming several activations in their RL ensemble.
- [Sentiment extraction from Consumer-generated noisy short texts (2017)](http://sentic.net/sentire2017meisheri.pdf):
  Uses SELU in feed-forward components.

## Graph Neural Networks (GNNs)
- [Graph Clustering with Graph Neural Networks (DMoN, JMLR 2023)](https://jmlr.org/papers/volume24/20-998/20-998.pdf):
  Uses SeLU inside a modified GCN and explicitly replaces ReLU with SeLU for better convergence.
- [Locally Private Graph Neural Networks (CCS 2021)](https://www.idiap.ch/~gatica/publications/SajadmaneshGatica-ccs21.pdf):
  Uses SeLU in GCN/GAT/GraphSAGE backbones (two graph conv layers with SeLU + dropout).
- [Reducing Oversmoothing in Graph Neural Networks by Activation Design (ICLR 2023 submission)](https://openreview.net/pdf?id=8CDeu0f4i2):
  Compares SeLU-enhanced GCN/GAT variants and discusses depth/oversmoothing behavior.
- [Deep Probabilistic Dual Graph Convolution Network (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2b6d0f44e1c1f9f1b3c4b1b4d4d66c0-Abstract-Conference.html):
  Uses SeLU in its GCN stack.

## Transformers & LLM-adjacent models

- [TMRN-GLU: Transformer-Based Automatic Classification Recognition Network (2022)](https://www.mdpi.com/2079-9292/11/10/1554):
  Chooses SELU where activations are needed, citing stability across SNR conditions.
- [SeLU-transformer for hoax/news classification (2020–2024 line of work)](https://www.semanticscholar.org/paper/BET%3A-A-Backtranslation-Approach-for-Easy-Data-in-Corbeil-Ghadivel/84d2d7f7fa2f656db458a5c369d0aa35ddb60f5e):
  Reports strong text classification performance with SeLU-Transformer variants.
- [TabTranSELU: A transformer adaptation for solving tabular data (2024)](https://pdfs.semanticscholar.org/885c/822b8d71d5edadb874dafdcb67f36ce49d11.pdf):
  Replaces a normalization+ReLU pattern with SELU for tabular transformer stability.
- [BERTSurv (clinical outcomes)](https://cse.cs.ucsb.edu/sites/default/files/publications/bertsurv-bert_based_survival_models_for_predicting_outcomes_for_trauma_patients.pdf):
  Uses ReLU or SELU in downstream components alongside BERT.

## Time series & Foundation(-style) models
- [Zero-shot Imputation with Foundation Inference Models for Dynamical Systems (ICLR 2025)](https://arxiv.org/abs/2402.07594):
  Uses SeLU as activation for all feed-forward networks in FIM/FIM-ℓ modules.
- [Foundation Inference Models for Stochastic Differential Equations (2025)](https://arxiv.org/abs/2502.19049):
  Follow-up foundation-style work in the same line (check architecture details for continued SeLU use).
- [Towards Foundation Inference Models that Learn ODEs In-Context (2025)](https://arxiv.org/abs/2510.12650):
  Extends the FIM line toward in-context ODE learning.

## Generative Models
### Autoencoders & VAEs
- [Self-Normalizing Neural Networks (NeurIPS 2017)](https://arxiv.org/abs/1706.02515):
  Foundational theory that motivates deep encoder/decoder stacks with SELU.
- [Training Deep AutoEncoders for Collaborative Filtering (2017/2018)](https://arxiv.org/abs/1708.01715):
  Practical deep AE results with SELU among competitive activations.
- [Application of generative autoencoder in de novo molecular design (2017)](https://arxiv.org/abs/1711.07839):
  Reports faster convergence with SELU in molecular generation pipelines.

### GANs
- [A Practical Approach for Training Deep Convolutional GANs with SELU Activation (2019)](https://dl.acm.org/doi/10.1145/3313181.3313187):
  Empirical study suggesting SELU can help stabilize or accelerate GAN training in specific setups.
- [THINKING LIKE A MACHINE - Generating Visual Rationales with Wasserstein GANs (2017)](https://pdfs.semanticscholar.org/dd4c/23a21b1199f34e5003e26d2171d02ba12d45.pdf):
  Uses SELU-style setups to reduce reliance on batch normalization.

### Normalizing Flows
- [Contextual Movement Models Based on Normalizing Flows (2021)](https://ieeexplore.ieee.org/document/9441650):
  Uses SELU in MLP components for movement modeling.
- [Individual Survival Curves with Conditional Normalizing Flows (2021)](https://arxiv.org/abs/2107.12825):
  Uses SELU as activation across datasets in a CNF-based survival modeling setting.

### Diffusion & Score-based models
- [SE(3)-Equivariant Diffusion Graph Nets (workshop 2024)](https://openreview.net/pdf/37fd86fe050a82aa6d9d9a308685520622b3fcc4.pdf):
  Uses SELU activations in MLPs inside the diffusion-graph pipeline for fluid flow fields.
- [Radar emitter denoising with DDPM variants (2024)](https://www.mdpi.com/2072-4292/16/17/3215):
  Employs SELU in 1D conv blocks within a DDPM-style architecture.
- [DIDiffGes: Decoupled Semi-Implicit Diffusion Models for Gesture Generation (2025)](https://arxiv.org/abs/2503.17059):
  Uses MLP blocks with SELU in a diffusion+GAN hybrid for fast sampling.

### Flow Matching / Schrödinger Bridges
- [Simulation-Free Schrödinger Bridges via Score and Flow Matching (AISTATS 2024)](https://proceedings.mlr.press/v238/tong24a/tong24a.pdf):
  Uses 3-layer MLPs with SeLU activations in vector-field/score networks.
- [Meta Flow Matching (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/ebdb990471f653dffb425eff03c7c980-Paper-Conference.pdf):
  Uses multi-layer MLPs with SELU activations for synthetic and biological experiments.
- [Source-Guided Flow Matching (2025)](https://arxiv.org/abs/2508.14807):
  Uses SELU-MLP vector fields with smoothing priors.

## Reinforcement Learning
- [Automated Cloud Provisioning on AWS using Deep Reinforcement Learning (2017)](https://arxiv.org/abs/1709.04305):
  Deep CNN with SELUs in RL settings.
- [Learning to Run with Actor-Critic Ensemble (2017)](https://arxiv.org/abs/1712.08987):
  Reports SELU among top-performing activation choices.
