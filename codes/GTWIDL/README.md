# Generalized Time Warping Invariant Dictionary Learning (GTWIDL)

PyTorch implementation of the algorithm from the paper:
**"Generalized Time Warping Invariant Dictionary Learning for Time Series Classification and Clustering"**
([arXiv:2306.17690](https://arxiv.org/abs/2306.17690))

## Features

- **Complete GTWIDL Algorithm**: Full implementation with block coordinate descent optimization
- **Multiple Basis Functions**: Support for polynomial, exponential, logarithmic, hyperbolic tangent, and I-spline basis functions
- **Flexible Time Warping**: Continuous time warping operator for better temporal alignment
- **Classification**: Multiple classifiers including SVM, k-NN, Random Forest, and nearest dictionary methods
- **Clustering**: Standard clustering, adaptive dictionary clustering, and hierarchical methods
- **GPU Support**: Full PyTorch implementation with CUDA support
- **Visualization Tools**: Utilities for visualizing dictionaries, reconstructions, and results

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0

## Quick Start

### Simple Example

```python
from gtwidl import GTWIDL
from utils import load_synthetic_dataset, normalize_time_series
import torch

# Load data
X, y = load_synthetic_dataset(n_samples=100, length=100, n_classes=3)
X = normalize_time_series(X, method='zscore')
X_torch = torch.tensor(X, dtype=torch.float32)

# Train GTWIDL
model = GTWIDL(
    n_atoms=10,           # Number of dictionary atoms
    atom_length=100,      # Length of atoms
    n_basis=5,            # Number of basis functions
    basis_type='polynomial',
    lambda_sparse=0.1,    # Sparsity parameter
    max_iter=50
)

dictionary, alphas, betas = model.fit(X_torch)

# Transform new data
alphas_new, betas_new = model.transform(X_torch[:5])
```

Run the simple example:
```bash
python example_simple.py
```

### Classification Example

```bash
# Standard classification with SVM
python experiment_classification.py \
    --n_samples 200 \
    --n_atoms 10 \
    --classifier_type standard \
    --base_classifier svm \
    --visualize

# Nearest dictionary classification
python experiment_classification.py \
    --classifier_type nearest_dictionary \
    --visualize
```

### Clustering Example

```bash
# Standard clustering with k-means
python experiment_clustering.py \
    --n_samples 150 \
    --n_clusters 3 \
    --clustering_method standard \
    --base_clustering kmeans \
    --visualize

# Adaptive dictionary clustering
python experiment_clustering.py \
    --clustering_method adaptive \
    --visualize

# Hierarchical clustering
python experiment_clustering.py \
    --clustering_method hierarchical \
    --linkage average \
    --visualize
```

## Project Structure

```
RTHAR/
├── gtwidl.py                    # Core GTWIDL implementation
├── classification.py            # Classification methods
├── clustering.py                # Clustering methods
├── utils.py                     # Utilities and visualization
├── experiment_classification.py # Classification experiments
├── experiment_clustering.py     # Clustering experiments
├── example_simple.py            # Simple usage example
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Core Components

### 1. Basis Functions

The implementation supports five types of continuous monotonic basis functions:

- **Polynomial**: `t^1, t^2, ..., t^degree`
- **Exponential**: `exp(k*t)` for different k values
- **Logarithmic**: `log(1 + k*t)`
- **Hyperbolic Tangent**: `tanh(k*(t-c))`
- **I-Spline**: Integrated B-splines (monotonic)

### 2. Generalized Time Warping

The warping path is represented as a linear combination of basis functions:
```
p = Q @ β
```
where Q is the basis matrix and β are non-negative coefficients.

### 3. Dictionary Learning

Uses block coordinate descent to alternately optimize:
- Sparse coefficients (α)
- Warping path coefficients (β)
- Dictionary atoms (d)

### 4. Classification Methods

- **GTWIDLClassifier**: Uses sparse coefficients as features with standard classifiers (SVM, k-NN, Random Forest)
- **NearestDictionaryClassifier**: Learns class-specific dictionaries
- **DistanceMetricClassifier**: k-NN with GTWIDL-based distance

### 5. Clustering Methods

- **GTWIDLClustering**: Standard clustering on GTWIDL features
- **AdaptiveDictionaryClustering**: Learns cluster-specific dictionaries
- **HierarchicalGTWIDLClustering**: Hierarchical clustering with GTWIDL distance

## Parameters

### GTWIDL Parameters

- `n_atoms`: Number of dictionary atoms
- `atom_length`: Length of each dictionary atom
- `n_basis`: Number of basis functions for time warping
- `basis_type`: Type of basis functions ('polynomial', 'exponential', 'logarithmic', 'tanh', 'ispline')
- `lambda_sparse`: Sparsity regularization parameter
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance
- `device`: Device to run on ('cpu' or 'cuda')

### Recommended Settings

- For smooth signals: `basis_type='polynomial'` or `basis_type='exponential'`
- For complex patterns: `basis_type='ispline'`
- Typical `lambda_sparse` range: 0.01 to 1.0
- Start with `n_atoms = 10-20` and adjust based on data complexity

## Algorithm Details

### Block Coordinate Descent Optimization

1. **Initialize** dictionary, sparse coefficients, and warping coefficients
2. **Iterate** until convergence:
   - Update sparse coefficients α using FISTA/coordinate descent
   - Update warping coefficients β using gradient descent
   - Update dictionary atoms using least squares
3. **Return** learned dictionary and coefficients

### Time Complexity

- Dictionary learning: O(N * K * T * I) where:
  - N = number of samples
  - K = number of atoms
  - T = time series length
  - I = number of iterations

## Visualization

The implementation includes visualization tools for:

- Dictionary atoms
- Time series reconstruction
- Warping paths
- Clustering results
- Classification boundaries

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{xu2023generalized,
  title={Generalized Time Warping Invariant Dictionary Learning for Time Series Classification and Clustering},
  author={Xu, Ruiyu and others},
  journal={arXiv preprint arXiv:2306.17690},
  year={2023}
}
```

## License

This implementation is for research and educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or issues, please open an issue on the repository.