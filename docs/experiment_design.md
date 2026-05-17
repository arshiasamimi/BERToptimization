# Project Design: Non-Linear Optimization Strategies for DistilBERT

## Core Idea

A comparative study of **non-linear optimization** strategies applied to a **DistilBERT** model for sentiment analysis on the IMDB dataset. The project uses DistilBERT (`distilbert-base-uncased`, 66M parameters) for its 60% faster training speed compared to standard BERT, making the full experiment feasible within Kaggle's 12-hour GPU session limit (estimated total runtime: 7 hours).

---

## Model & Freezing Strategy

**Architecture:** DistilBERT with a sequence classification head.

**Freezing:** All 6 encoder layers are frozen. Only the classification head is trainable — consisting of a pre-classifier (linear layer), an activation function (ReLU/GELU), and the final classifier (linear layer). This limits optimization to the final layers, keeping the project feasible on Kaggle's free tier.

**Preprocessing:** Tokenization with `DistilBertTokenizer`, padding and truncation to a fixed `max_length` (128 or 256). DistilBERT does not use `token_type_ids`.

---

## Phase 1: First-Order Optimization Comparison

**Goal:** Compare how different gradient-based update rules navigate the non-convex loss landscape.

**Data:** 10,000 IMDB reviews (20% of the dataset)

**Optimizers:**
| Optimizer | Key Property |
|-----------|--------------|
| AdamW | Adaptive learning rates with first/second moments; baseline for transformers |
| RMSProp | Moving average of squared gradients; prevents vanishing rates |
| SGD with Momentum | Gradient velocity; helps roll through local minima |
| Adagrad | Sum of squared historical gradients; downscales for sparse features |

**Metrics & Outputs:**
- Convergence curves (Training/Validation Loss vs. Epoch)
- Stability analysis (validation loss monotonic increase as overfitting red flag)
- Performance table (Accuracy, F1-score, AUC)

---

## Phase 2: Global Optimization Comparison (Hyperparameter Tuning)

**Goal:** Treat training configuration as an optimization problem — find best settings for learning rate, batch size, dropout, etc.

**Data:** 2,500 reviews (allows 10–20 iterations per method)

**Target Model:** Only the best-performing first-order optimizer from Phase 1 (expected: AdamW)

**HPO Methods:**
| Method | Strategy |
|--------|----------|
| Genetic Algorithm (GA) | Selection, crossover, mutation on population of hyperparameter sets |
| Particle Swarm Optimization (PSO) | Particles share global best solution through search space |
| Bayesian Optimization | Gaussian processes model objective function to select next points |

**Metrics & Outputs:**
- Fitness evolution plot (best accuracy vs. iteration number for each method)
- Parameter distribution map (scatter plot of learning rates vs. batch sizes explored)

---

## Phase 3: Final Evaluation

**Training:** Best configuration (optimizer + hyperparameters) on the full 10,000-review slice

**Testing:** Independent 10,000-review test set

**Deliverable:** Comparative analysis of convergence speed, stability, and final loss across local (first-order) vs. global (meta-heuristic) optimization methods.

---

## Expected Outputs

1. Convergence curves for all four first-order optimizers
2. Performance comparison table (Accuracy, F1, AUC)
3. Fitness evolution plots for GA, PSO, Bayesian Optimization
4. Hyperparameter search space visualizations
5. Final test set evaluation of best model
6. Written comparison: local gradient-based vs. global meta-heuristic optimization