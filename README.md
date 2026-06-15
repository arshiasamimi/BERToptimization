You're right - the content got cut off. Here's the complete README in a single markdown code block that you can copy directly:

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

## Project Structure

```
distilbert-hpo/
│
├── BERToptimization.ipynb    # Main Jupyter notebook with complete implementation
├── requirements.txt           # Python dependencies
├── README.md                  # Documentation
│
├── data/                      # (Generated) Dataset cache
│   └── imdb/                  # IMDB dataset storage
│
├── models/                    # (Generated) Model checkpoints
│   └── best_model.pth         # Best performing model
│
├── results/                   # (Generated) Results and metrics
│   ├── plots/                 # Visualization outputs
│   ├── metrics/               # CSV files with metrics
│   └── logs/                  # Training logs
│
└── configs/                   # Configuration files
    └── hpo_config.yaml        # Hyperparameter configurations
```

## Usage

### Running the Complete Pipeline

1. **Launch Jupyter notebook**:
   ```bash
   jupyter notebook BERToptimization.ipynb
   ```

2. **Run cells sequentially** from top to bottom:

   ### Section 1: Setup & Configuration
   ```python
   # Configure experiment parameters
   TRAIN_SIZE = 10000
   VAL_RATIO = 0.1
   TEST_SIZE = 2000
   MAX_LENGTH = 128
   BATCH_SIZE = 32
   ```

   ### Section 2: Data Loading
   ```python
   # Load and preprocess IMDB dataset
   dataset = load_dataset("stanfordnlp/imdb")
   train_dataset = preprocess_dataset(dataset["train"])
   ```

   ### Section 3: Baseline Training
   ```python
   # Train baseline model with default hyperparameters
   baseline_results = train_model(
       epochs=3,
       lr=2e-5,
       weight_decay=0.01,
       warmup_ratio=0.1,
       dropout_rate=0.1
   )
   ```

   ### Section 4: Run Optimizations
   ```python
   # PSO Optimization (20 evaluations)
   pso_optimizer = GlobalBestPSO(n_particles=5, dimensions=5)
   best_cost, best_pos = pso_optimizer.optimize(pso_objective, iters=4)
   
   # Bayesian Optimization (20 evaluations)
   result = gp_minimize(bayesian_objective, dimensions, n_calls=20)
   
   # Genetic Algorithm (20 evaluations)
   ga_optimizer = GeneticAlgorithm(population_size=5, n_generations=4)
   best_solution = ga_optimizer.optimize(ga_objective)
   ```

### Customizing the Optimization

You can modify the search space, evaluation budget, or algorithm parameters:

```python
# Modify search space
search_space = {
    "learning_rate": {"low": 1e-6, "high": 1e-4, "scale": "log"},
    "epochs": {"low": 2, "high": 10, "scale": "int"},
    # ... other parameters
}

# Change evaluation budget
N_EVALUATIONS = 30  # Increase for better results (slower)

# Adjust PSO parameters
options = {"c1": 1.5, "c2": 1.5, "w": 0.8}  # Cognitive, social, inertia weights
```

## Optimization Algorithms

### 1. Particle Swarm Optimization (PSO)

PSO simulates social behavior of bird flocks or fish schools to find optimal solutions.

**Core Mechanics**:
- Each particle represents a hyperparameter configuration
- Particles share information about their personal best and global best
- Velocity update balances exploration (inertia) and exploitation (cognitive + social)

**Update Equations**:
```
v_i(t+1) = w·v_i(t) + c₁·r₁·(p_best - x_i(t)) + c₂·r₂·(g_best - x_i(t))
x_i(t+1) = x_i(t) + v_i(t+1)
```

**Algorithm Parameters**:
- Population: 5 particles
- Iterations: 4 (20 total evaluations)
- Inertia weight (w): 0.9 (emphasizes exploration)
- Cognitive parameter (c₁): 2.0
- Social parameter (c₂): 2.0

**Strengths**:
- Excellent exploration capability
- Fast convergence on unimodal problems
- Simple implementation
- Memory-efficient

### 2. Bayesian Optimization (BO)

BO builds a probabilistic surrogate model (Gaussian Process) to guide the search.

**Core Components**:
- **Surrogate Model**: Gaussian Process providing mean (μ) and uncertainty (σ) predictions
- **Acquisition Function**: Expected Improvement (EI) balancing exploration vs exploitation

**Acquisition Function**:
```
EI(x) = E[max(f_min - f(x), 0)]
```

**Algorithm Parameters**:
- Total evaluations: 20
- Initial random points: 7 (burn-in phase)
- Acquisition: Expected Improvement
- Surrogate: Gaussian Process with Matérn kernel

**Strengths**:
- Most sample-efficient (fewer evaluations needed)
- Handles noisy objectives well
- Provides uncertainty estimates
- Excellent for expensive evaluations

### 3. Genetic Algorithm (GA)

GA mimics natural selection through evolution operators.

**Core Operators**:
- **Selection**: Tournament selection (size=2) chooses parents based on fitness
- **Crossover**: Single-point crossover with 80% probability for exploitation
- **Mutation**: Gaussian noise for continuous, integer adjustments for discrete parameters (20% probability)
- **Elitism**: Best individual automatically preserved

**Algorithm Parameters**:
- Population size: 5
- Generations: 4 (20 total evaluations)
- Crossover rate: 0.8
- Mutation rate: 0.2
- Tournament size: 2

**Strengths**:
- Handles mixed variable types naturally
- Maintains population diversity
- Effective for complex, non-convex landscapes
- Parallelizable population evaluations

## Evaluation Metrics

### Primary Metrics

1. **Validation Loss** (Optimization Objective)
   - Cross-entropy loss on validation set
   - Lower is better
   - Range: [0, ∞)

2. **Test Accuracy**
   - Percentage of correct predictions
   - Higher is better
   - Range: [0, 100%]

3. **Weighted F1 Score**
   - Harmonic mean of precision and recall
   - Higher is better
   - Range: [0, 1]

### Computational Metrics

4. **Total Optimization Time**
   - Wall-clock time for complete optimization
   - Includes all model training and evaluation

5. **Evaluation Steps to Best**
   - Number of function evaluations to discover optimal configuration
   - Lower indicates better sample efficiency

## Key Insights

### Algorithm Performance Analysis

1. **Sample Efficiency**
   - **Genetic Algorithm** found optimum in 11 evaluations (best)
   - **Bayesian Optimization** required 13 evaluations
   - **PSO** took 19 evaluations (most)
   
   *Insight*: GA's population-based approach with crossover allows effective exploitation of promising regions quickly.

2. **Computational Efficiency**
   - **Genetic Algorithm**: 2184s (fastest)
   - **PSO**: 2551s
   - **Bayesian Optimization**: 3107s (slowest)
   
   *Insight*: Despite similar evaluation counts, GA's implementation and convergence characteristics made it fastest.

3. **Solution Quality**
   - **Genetic Algorithm** achieved lowest test loss (0.3698)
   - **Bayesian Optimization** closely followed (0.3735)
   - **PSO** performed well but slightly worse (0.3752)
   
   *Insight*: GA's ability to balance exploration (mutation) and exploitation (crossover) proved superior for this problem.

4. **Convergence Patterns**
   - **PSO**: Initial exploration followed by rapid convergence
   - **Bayesian**: Steady improvement throughout, efficient use of evaluations
   - **GA**: Progressive improvement with occasional jumps (mutations)

### Practical Recommendations

Based on empirical testing, the **Genetic Algorithm** is recommended as the core HPO engine when:
- Evaluation budget is limited (20-50 evaluations)
- Mixed continuous/discrete variables are present
- Fast convergence is desired
- Implementation complexity is not a constraint

Bayesian Optimization is a strong alternative for extremely expensive evaluations where sample efficiency is paramount, despite longer per-iteration overhead.

## Technical Details

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 4GB VRAM | 8GB+ VRAM (Tesla T4) |
| RAM | 8GB | 12GB+ |
| Storage | 2GB | 5GB+ |
| Network | - | Broadband (for dataset download) |

### Software Stack

```
Core Libraries:
├── PyTorch 1.13+            # Deep learning framework
├── Transformers 4.25+       # Hugging Face models
├── Datasets 2.8+            # Dataset handling
│
Optimization Libraries:
├── PySwarms 1.3+            # PSO implementation
├── Scikit-Optimize 0.10+    # Bayesian Optimization
│
Utilities:
├── NumPy 1.21+              # Numerical computing
├── Pandas 1.5+              # Data manipulation
├── Matplotlib 3.6+          # Visualization
├── Scikit-learn 1.1+        # Metrics
```

### Performance Optimizations

The code implements several optimizations for efficient training:

1. **Mixed Precision Training (FP16)**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   with autocast():
       outputs = model(input_ids, attention_mask=attention_mask)
   ```

2. **Gradient Clipping**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **Efficient Data Loading**
   ```python
   DataLoader(
       dataset,
       batch_size=32,
       num_workers=2,
       pin_memory=True,
       prefetch_factor=3,
       persistent_workers=True
   )
   ```

4. **Memory Management**
   ```python
   # Clear cache between evaluations
   torch.cuda.empty_cache()
   gc.collect()
   ```

## Future Work

### Planned Enhancements

1. **Additional Algorithms**
   - Hyperband for adaptive resource allocation
   - BOHB (Bayesian Optimization + Hyperband)
   - CMA-ES for continuous optimization
   - Random Search baseline

2. **Extended Capabilities**
   - Multi-objective optimization (loss + inference time)
   - Transfer learning across datasets
   - Automated result versioning and tracking
   - Distributed/parallel evaluation support

3. **Analysis Tools**
   - Hyperparameter importance analysis
   - Response surface visualization
   - Ablation studies
   - Statistical significance testing

4. **Model Support**
   - Additional transformer architectures (RoBERTa, ALBERT, etc.)
   - Multi-task learning scenarios
   - Different NLP tasks (NER, QA, etc.)

### Research Questions

- How do these algorithms scale to higher-dimensional spaces (10+ parameters)?
- What's the impact of different acquisition functions in Bayesian Optimization?
- Can hybrid approaches (e.g., GA + local search) improve results?
- How transferable are optimal hyperparameters across different datasets?

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Areas for Contribution

- Bug fixes and performance improvements
- Additional optimization algorithms
- Documentation enhancements
- Test coverage expansion
- New evaluation metrics
- Visualization improvements

## References

### Foundational Papers

1. **BERT & DistilBERT**
   - Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
   - Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"

2. **Optimization Algorithms**
   - Kennedy, J., & Eberhart, R. (1995). "Particle Swarm Optimization"
   - Snoek, J., et al. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms"
   - Goldberg, D. E. (1989). "Genetic Algorithms in Search, Optimization, and Machine Learning"

3. **Hyperparameter Optimization**
   - Bergstra, J., & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization"
   - Feurer, M., & Hutter, F. (2019). "Hyperparameter Optimization"

### Libraries Used

- **PyTorch**: https://pytorch.org/
- **Transformers**: https://huggingface.co/docs/transformers
- **PySwarms**: https://pyswarms.readthedocs.io/
- **Scikit-Optimize**: https://scikit-optimize.github.io/
- **Datasets**: https://huggingface.co/docs/datasets

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{distilbert-hpo-2026,
  author = {[Your Name]},
  title = {DistilBERT Hyperparameter Optimization: A Comparative Study of PSO, Bayesian Optimization, and Genetic Algorithms},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/distilbert-hpo},
  note = {Includes implementations of three HPO algorithms with comprehensive evaluation}
}
```

## Acknowledgments

- **Hugging Face** for the transformers library and datasets
- **PyTorch team** for the deep learning framework
- **OpenAI** for the IMDB dataset
- Contributors to PySwarms and Scikit-Optimize libraries

## Contact

For questions, suggestions, or collaboration opportunities:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**⭐ Star this repository if you find it useful!**

**Note**: This project was developed and tested in Google Colab with Tesla T4 GPU acceleration. Runtime may vary based on hardware configuration.
```

This is the complete README file. You can copy everything from the opening triple backticks to the closing triple backticks and paste it directly into your GitHub repository's README.md file.
