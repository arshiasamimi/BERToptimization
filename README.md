# DistilBERT Hyperparameter Optimization: A Comparative Study of PSO, Bayesian Optimization, and Genetic Algorithms

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Results Summary](#results-summary)
- [Hyperparameter Search Space](#hyperparameter-search-space)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Optimization Algorithms](#optimization-algorithms)
- [Evaluation Metrics](#evaluation-metrics)
- [Key Insights](#key-insights)
- [Technical Details](#technical-details)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

## Overview

This repository presents a comprehensive implementation and comparative analysis of three state-of-the-art hyperparameter optimization (HPO) algorithms for fine-tuning DistilBERT on the IMDB sentiment analysis dataset. The project demonstrates how different optimization strategies navigate the complex, non-convex hyperparameter landscape of transformer-based models.

### Why This Matters

Hyperparameter optimization for deep learning models is a challenging black-box optimization problem characterized by:
- **Expensive evaluations**: Each hyperparameter configuration requires training a full DistilBERT model
- **Non-convex landscape**: Multiple local optima with no closed-form gradient information
- **Mixed variable types**: Combination of continuous (learning rate, dropout) and discrete (epochs) parameters
- **Noisy objective**: Stochasticity in training leads to varying validation metrics

This project implements three fundamentally different approaches to tackle these challenges and provides empirical evidence for their comparative performance.

## Key Features

- **End-to-end HPO pipeline**: Complete workflow from data preprocessing to model evaluation
- **Three optimization algorithms**: 
  - Particle Swarm Optimization (swarm intelligence)
  - Bayesian Optimization (probabilistic surrogate modeling)
  - Genetic Algorithm (evolutionary computation)
- **Comprehensive evaluation**: Loss, accuracy, F1 score, computational efficiency, and convergence analysis
- **Rich visualizations**: Side-by-side comparisons, convergence plots, performance matrices
- **Mixed parameter handling**: Seamless optimization of both continuous and discrete hyperparameters
- **GPU-accelerated training**: Optimized training loops with mixed precision support
- **Reproducible results**: Fixed random seeds and deterministic settings

## Results Summary

The Genetic Algorithm emerged as the overall champion across multiple performance metrics:

### Test Set Performance

| Metric | PSO | Bayesian Optimization | **Genetic Algorithm** |
|--------|-----|----------------------|----------------------|
| Test Loss | 0.3752 | 0.3735 | **0.3698** |
| Test Accuracy | 84.20% | 84.15% | **84.40%** |
| Test F1 Score | 0.8418 | 0.8415 | **0.8438** |

### Computational Efficiency

| Metric | PSO | Bayesian Optimization | **Genetic Algorithm** |
|--------|-----|----------------------|----------------------|
| Total Time (seconds) | 2551.40 | 3107.22 | **2184.13** |
| Eval Steps to Best | 19 | 13 | **11** |

### Key Achievements
- ✅ **Lowest test loss**: 0.3698 (Genetic Algorithm)
- ✅ **Highest accuracy**: 84.40% (Genetic Algorithm)
- ✅ **Fastest optimization**: 2184 seconds (Genetic Algorithm)
- ✅ **Most sample-efficient**: Found optimum in just 11 evaluations (Genetic Algorithm)

## Hyperparameter Search Space

The optimization explores a 5-dimensional hyperparameter space with carefully chosen bounds and scaling:

| Parameter | Lower Bound | Upper Bound | Scale | Description |
|-----------|-------------|-------------|-------|-------------|
| learning_rate | 5×10⁻⁶ | 5×10⁻⁵ | log | AdamW optimizer learning rate |
| epochs | 3 | 7 | integer | Number of training epochs |
| weight_decay | 1×10⁻⁵ | 5×10⁻² | log | L2 regularization strength |
| warmup_ratio | 0.0 | 0.5 | linear | Learning rate warmup ratio |
| dropout_rate | 0.05 | 0.30 | linear | Dropout probability |

### Why These Ranges?
- **Learning rate**: Log scale captures the sensitivity of transformer training to small changes in learning rate
- **Epochs**: Balanced between underfitting (3) and overfitting (7)
- **Weight decay**: Log scale for regularization strength
- **Warmup ratio**: Linear scale as warmup effectiveness varies linearly
- **Dropout**: Linear scale for probability values

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 12GB+ RAM
- 5GB+ free storage

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/distilbert-hpo.git
cd distilbert-hpo

# Install required packages
pip install torch torchvision transformers datasets scikit-learn
pip install pyswarms scikit-optimize matplotlib pandas tqdm

# Optional: Install for GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
