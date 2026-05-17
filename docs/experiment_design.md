# Experiment Design: Non-Linear Optimization Strategies for DistilBERT

## Project Overview

This project is structured as a comparative study of **non-linear optimization** strategies applied to a **DistilBERT** model for sentiment analysis. By using **DistilBERT** (`distilbert-base-uncased`), you reduce the search space to **66 million parameters** and achieve training speeds roughly **60% faster** than standard BERT.

The following structure is designed to run within **Kaggle's 12-hour GPU session limit**, utilizing a total estimated runtime of **7 hours**.

---

## Phase 0: Environment & Model Preparation

- **Dataset:** Use the **IMDB Dataset** (50,000 polar reviews).
- **Architecture:** Use **DistilBERT** with a sequence classification head.
- **Freezing Strategy:** **Freeze all 6 encoder layers** of the DistilBERT base model. You will only **unfreeze the classification head**, which typically consists of a **pre-classifier** (linear), an activation function (ReLU/Gelu), and the final **classifier** (linear). This limits the optimization variables to the final layers, ensuring the project is "smoothly doable" on Kaggle's free tier.
- **Preprocessing:** Tokenize reviews using the **DistilBertTokenizer**, applying padding and truncation to a fixed `max_length` (e.g., 128 or 256). **Note:** DistilBERT does not require `token_type_ids`.

---

## Phase 1: First-Order Optimization Comparison (Local Search)

In this phase, you compare how different gradient-based update rules navigate the non-convex loss landscape.

- **Data Slice:** **10,000 reviews** (20% of the dataset).
- **Techniques to Compare:**
  - **AdamW:** The baseline for Transformers; it uses **adaptive learning rates** based on first and second moments of the gradients.
  - **RMSProp:** Uses a **moving average of squared gradients** to adjust learning rates, preventing them from vanishing too quickly.
  - **SGD with Momentum:** Uses the "**velocity**" of previous gradients to help the optimizer "roll" through local minima.
  - **Adagrad:** Adapts the learning rate by **downscaling parameters** based on the sum of squared historical gradients.

- **Metrics & Visualizations:**
  - **Convergence Curves:** Plot **Training/Validation Loss vs. Epoch** to show which optimizer reaches the minimum fastest.
  - **Stability Analysis:** Monitor for **monotonically increasing validation loss**, a clear red flag for overfitting.
  - **Performance Table:** Report final **Accuracy, F1-score, and AUC** for each optimizer.

---

## Phase 2: Global Optimization Comparison (Hyperparameter Tuning)

This phase treats the training configuration itself as an optimization problem to find the best settings (Learning Rate, Batch Size, Dropout, etc.).

- **Data Slice:** **2,500 reviews** (to allow for 10–20 iterations per method).
- **Target Model:** Only perform this phase using the **best-performing first-order optimizer** from Phase 1 (likely AdamW).

- **Techniques to Compare:**
  - **Genetic Algorithm (GA):** Evolves a "population" of hyperparameter sets via **selection, crossover, and mutation**.
  - **Particle Swarm Optimization (PSO):** Treats hyperparameter sets as **particles** flying through a search space toward the "global best".
  - **Bayesian Optimization:** Uses **Gaussian processes** to build a statistical model of the objective function to pick the next most promising points.

- **Metrics & Visualizations:**
  - **Fitness Evolution Plot:** Graph the **highest accuracy achieved** over the number of search iterations.
  - **Parameter Distribution Map:** A scatter plot showing which regions of the hyperparameter space (e.g., $1 \times 10^{-5}$ vs $5 \times 10^{-5}$ learning rate) each method explored.

---

## Phase 3: Final Analysis & Reporting

- **Final Evaluation:** Train your best model (Best Optimizer + Best Hyperparameters) on the **full 10,000-review slice** and evaluate on the independent **10,000-review test set**.
- **Optimization Insights:** Compare the **convergence speed, stability, and final loss** across all methods to determine which combination is superior for your non-linear optimization context.
- **Kaggle Stability:** Use the **"Save & Run All"** feature for the longer Phase 2 runs to avoid session timeouts.