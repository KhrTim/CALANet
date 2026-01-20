"""
Utility functions for GTWIDL experiments
Data loading, preprocessing, visualization, and evaluation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_ucr_dataset(dataset_name: str, data_dir: str = './data') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load UCR time series dataset

    Args:
        dataset_name: Name of the UCR dataset
        data_dir: Directory containing UCR datasets

    Returns:
        X_train, y_train, X_test, y_test
    """
    import os

    train_file = os.path.join(data_dir, dataset_name, f"{dataset_name}_TRAIN.tsv")
    test_file = os.path.join(data_dir, dataset_name, f"{dataset_name}_TEST.tsv")

    # Load training data
    train_data = np.loadtxt(train_file, delimiter='\t')
    y_train = train_data[:, 0].astype(int)
    X_train = train_data[:, 1:]

    # Load test data
    test_data = np.loadtxt(test_file, delimiter='\t')
    y_test = test_data[:, 0].astype(int)
    X_test = test_data[:, 1:]

    return X_train, y_train, X_test, y_test


def load_synthetic_dataset(n_samples: int = 200,
                          length: int = 100,
                          n_classes: int = 3,
                          noise_level: float = 0.1,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic time series dataset with different patterns

    Args:
        n_samples: Number of samples
        length: Length of time series
        n_classes: Number of classes
        noise_level: Amount of noise to add
        random_state: Random seed

    Returns:
        X, y: Time series data and labels
    """
    np.random.seed(random_state)

    X = []
    y = []

    t = np.linspace(0, 4 * np.pi, length)

    for i in range(n_samples):
        class_label = i % n_classes

        if class_label == 0:
            # Sine wave
            signal = np.sin(t)
        elif class_label == 1:
            # Square wave
            signal = np.sign(np.sin(t))
        elif class_label == 2:
            # Sawtooth wave
            signal = 2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))
        else:
            # Random other patterns
            freq = np.random.uniform(0.5, 2.0)
            signal = np.sin(freq * t) + 0.3 * np.sin(3 * freq * t)

        # Add noise
        signal += np.random.normal(0, noise_level, length)

        # Add random time warping
        warp = np.cumsum(np.random.uniform(0.5, 1.5, length))
        warp = warp / warp[-1] * (length - 1)
        signal = np.interp(np.arange(length), warp, signal)

        X.append(signal)
        y.append(class_label)

    return np.array(X), np.array(y)


def normalize_time_series(X: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize time series data

    Args:
        X: Time series data (n_samples, length) or (n_samples, length, n_dims)
        method: Normalization method ('zscore', 'minmax', 'none')

    Returns:
        X_normalized: Normalized time series
    """
    if method == 'none':
        return X

    X_norm = X.copy()

    if method == 'zscore':
        # Z-score normalization per sample
        for i in range(len(X)):
            mean = np.mean(X[i])
            std = np.std(X[i])
            if std > 1e-8:
                X_norm[i] = (X[i] - mean) / std
            else:
                X_norm[i] = X[i] - mean

    elif method == 'minmax':
        # Min-max normalization per sample
        for i in range(len(X)):
            min_val = np.min(X[i])
            max_val = np.max(X[i])
            if max_val - min_val > 1e-8:
                X_norm[i] = (X[i] - min_val) / (max_val - min_val)
            else:
                X_norm[i] = X[i] - min_val

    return X_norm


def to_torch(X: np.ndarray, y: Optional[np.ndarray] = None, device: str = 'cpu') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Convert numpy arrays to torch tensors

    Args:
        X: Time series data
        y: Labels (optional)
        device: Device to place tensors on

    Returns:
        X_torch, y_torch (or just X_torch if y is None)
    """
    X_torch = torch.tensor(X, dtype=torch.float32, device=device)

    if y is not None:
        return X_torch, y
    else:
        return X_torch


def visualize_dictionary(dictionary: torch.Tensor,
                        n_cols: int = 5,
                        figsize: Tuple[int, int] = (15, 10),
                        save_path: Optional[str] = None):
    """
    Visualize dictionary atoms

    Args:
        dictionary: Dictionary atoms (length, n_atoms) or (length, n_atoms, n_dims)
        n_cols: Number of columns in subplot grid
        figsize: Figure size
        save_path: Path to save figure (optional)
    """
    n_atoms = dictionary.shape[1]
    n_rows = (n_atoms + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_atoms > 1 else [axes]

    for i in range(n_atoms):
        atom = dictionary[:, i].cpu().numpy()

        if atom.ndim == 1:
            axes[i].plot(atom)
        else:
            for dim in range(atom.shape[1]):
                axes[i].plot(atom[:, dim], label=f'Dim {dim}')
            axes[i].legend()

        axes[i].set_title(f'Atom {i}')
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_atoms, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dictionary visualization saved to {save_path}")

    plt.show()


def visualize_reconstruction(x_original: torch.Tensor,
                            x_reconstructed: torch.Tensor,
                            title: str = "Reconstruction",
                            save_path: Optional[str] = None):
    """
    Visualize original vs reconstructed time series

    Args:
        x_original: Original time series
        x_reconstructed: Reconstructed time series
        title: Plot title
        save_path: Path to save figure (optional)
    """
    x_orig = x_original.cpu().numpy()
    x_recon = x_reconstructed.cpu().numpy()

    if x_orig.ndim == 1:
        plt.figure(figsize=(12, 4))
        plt.plot(x_orig, label='Original', linewidth=2)
        plt.plot(x_recon, label='Reconstructed', linewidth=2, linestyle='--')
        plt.legend()
        plt.title(title)
        plt.grid(True, alpha=0.3)
    else:
        n_dims = x_orig.shape[1]
        fig, axes = plt.subplots(n_dims, 1, figsize=(12, 3 * n_dims))
        if n_dims == 1:
            axes = [axes]

        for dim in range(n_dims):
            axes[dim].plot(x_orig[:, dim], label='Original', linewidth=2)
            axes[dim].plot(x_recon[:, dim], label='Reconstructed', linewidth=2, linestyle='--')
            axes[dim].legend()
            axes[dim].set_title(f'{title} - Dimension {dim}')
            axes[dim].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Reconstruction visualization saved to {save_path}")

    plt.show()


def visualize_warping_path(beta: torch.Tensor,
                          time_warping,
                          title: str = "Warping Path",
                          save_path: Optional[str] = None):
    """
    Visualize warping path

    Args:
        beta: Warping coefficients
        time_warping: GeneralizedTimeWarping object
        title: Plot title
        save_path: Path to save figure (optional)
    """
    path = time_warping.generate_warping_path(beta).cpu().numpy()
    t = time_warping.t.cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(t * (len(path) - 1), path, linewidth=2, label='Warping path')
    plt.plot(t * (len(path) - 1), t * (len(path) - 1), 'k--', alpha=0.5, label='Identity')
    plt.xlabel('Original time')
    plt.ylabel('Warped time')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Warping path visualization saved to {save_path}")

    plt.show()


def visualize_clustering(X: np.ndarray,
                        labels: np.ndarray,
                        title: str = "Clustering Results",
                        max_samples_per_cluster: int = 10,
                        save_path: Optional[str] = None):
    """
    Visualize clustering results

    Args:
        X: Time series data (n_samples, length)
        labels: Cluster labels
        title: Plot title
        max_samples_per_cluster: Maximum number of samples to show per cluster
        save_path: Path to save figure (optional)
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 3 * n_clusters))
    if n_clusters == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for i, label in enumerate(unique_labels):
        cluster_data = X[labels == label]
        n_show = min(len(cluster_data), max_samples_per_cluster)

        for j in range(n_show):
            axes[i].plot(cluster_data[j], color=colors[i], alpha=0.5)

        axes[i].set_title(f'Cluster {label} (n={len(cluster_data)})')
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.001)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Clustering visualization saved to {save_path}")

    plt.show()


def cross_validate(model_class,
                   model_params: dict,
                   X: np.ndarray,
                   y: np.ndarray,
                   n_splits: int = 5,
                   random_state: int = 42) -> dict:
    """
    Perform cross-validation

    Args:
        model_class: Model class to instantiate
        model_params: Parameters for model
        X: Time series data
        y: Labels
        n_splits: Number of cross-validation splits
        random_state: Random seed

    Returns:
        results: Dictionary with mean and std of metrics
    """
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accuracies = []
    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert to torch
        X_train_torch = torch.tensor(X_train, dtype=torch.float32)
        X_val_torch = torch.tensor(X_val, dtype=torch.float32)

        # Train model
        model = model_class(**model_params)
        model.fit(X_train_torch, y_train)

        # Evaluate
        metrics = model.evaluate(X_val_torch, y_val, verbose=False)

        accuracies.append(metrics['accuracy'])
        f1_scores.append(metrics['f1_score'])

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")

    results = {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores)
    }

    print(f"\n{'='*50}")
    print("Cross-Validation Results:")
    print(f"  Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"  F1 Score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")

    return results


def save_model(model, filepath: str):
    """Save model to file"""
    torch.save({
        'dictionary': model.dictionary,
        'n_atoms': model.n_atoms,
        'atom_length': model.atom_length,
        'n_basis': model.n_basis,
        'basis_type': model.basis_type,
        'lambda_sparse': model.lambda_sparse
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str, device: str = 'cpu'):
    """Load model from file"""
    from gtwidl import GTWIDL

    checkpoint = torch.load(filepath, map_location=device)

    model = GTWIDL(
        n_atoms=checkpoint['n_atoms'],
        atom_length=checkpoint['atom_length'],
        n_basis=checkpoint['n_basis'],
        basis_type=checkpoint['basis_type'],
        lambda_sparse=checkpoint['lambda_sparse'],
        device=device
    )

    model.dictionary = checkpoint['dictionary'].to(device)

    print(f"Model loaded from {filepath}")
    return model