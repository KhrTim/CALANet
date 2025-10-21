"""
Simple example demonstrating basic GTWIDL usage
Shows dictionary learning, reconstruction, and feature extraction
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from gtwidl import GTWIDL
from utils import load_synthetic_dataset, normalize_time_series, to_torch


def main():
    print("="*70)
    print("Simple GTWIDL Example")
    print("="*70)

    # Generate synthetic data
    print("\n1. Generating synthetic time series data...")
    X, y = load_synthetic_dataset(n_samples=50, length=100, n_classes=3, noise_level=0.1)
    X = normalize_time_series(X, method='zscore')
    print(f"   Data shape: {X.shape}")
    print(f"   Classes: {np.unique(y)}")

    # Convert to torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    X_torch = to_torch(X, device=device)

    # Initialize GTWIDL
    print("\n2. Initializing GTWIDL model...")
    model = GTWIDL(
        n_atoms=8,              # Number of dictionary atoms
        atom_length=100,        # Length of each atom
        n_basis=5,              # Number of basis functions for warping
        basis_type='polynomial', # Type of basis functions
        lambda_sparse=0.1,      # Sparsity parameter
        max_iter=30,            # Maximum iterations
        device=device,
        verbose=True
    )

    # Learn dictionary
    print("\n3. Learning dictionary from data...")
    dictionary, alphas, betas = model.fit(X_torch)

    print(f"\n   Dictionary shape: {dictionary.shape}")
    print(f"   Sparse coefficients shape: {alphas.shape}")
    print(f"   Number of warping paths: {len(betas)}")

    # Transform new data
    print("\n4. Transforming data to sparse representation...")
    test_sample = X_torch[0:5]
    alphas_test, betas_test = model.transform(test_sample)
    print(f"   Sparse coefficients for 5 samples: {alphas_test.shape}")

    # Reconstruct a sample
    print("\n5. Reconstructing a sample...")
    sample_idx = 0
    x_original = X_torch[sample_idx]
    alpha_sample = alphas[sample_idx]
    beta_sample = betas[sample_idx]

    # Reconstruct
    x_reconstructed = torch.zeros_like(x_original)
    for k in range(model.n_atoms):
        if alpha_sample[k] > 1e-6:
            atom = dictionary[:, k]
            warped_atom = model.time_warping.warp_time_series(atom, beta_sample)
            x_reconstructed = x_reconstructed + alpha_sample[k] * warped_atom

    # Compute reconstruction error
    recon_error = torch.mean((x_original - x_reconstructed) ** 2).item()
    print(f"   Reconstruction MSE: {recon_error:.6f}")

    # Visualize results
    print("\n6. Visualizing results...")

    # Plot original vs reconstructed
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Plot 1: Original vs Reconstructed
    axes[0, 0].plot(x_original.cpu().numpy(), label='Original', linewidth=2)
    axes[0, 0].plot(x_reconstructed.cpu().numpy(), label='Reconstructed', linewidth=2, linestyle='--')
    axes[0, 0].set_title('Original vs Reconstructed')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Sparse coefficients
    axes[0, 1].bar(range(len(alpha_sample)), alpha_sample.cpu().numpy())
    axes[0, 1].set_title('Sparse Coefficients')
    axes[0, 1].set_xlabel('Atom Index')
    axes[0, 1].set_ylabel('Coefficient Value')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Sample dictionary atoms
    for i in range(min(4, model.n_atoms)):
        axes[1, 0].plot(dictionary[:, i].cpu().numpy(), label=f'Atom {i}', alpha=0.7)
    axes[1, 0].set_title('Sample Dictionary Atoms')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Warping path
    warping_path = model.time_warping.generate_warping_path(beta_sample).cpu().numpy()
    t = np.arange(len(warping_path))
    axes[1, 1].plot(t, warping_path, label='Warping Path', linewidth=2)
    axes[1, 1].plot(t, t, 'k--', label='Identity', alpha=0.5)
    axes[1, 1].set_title('Warping Path')
    axes[1, 1].set_xlabel('Original Time')
    axes[1, 1].set_ylabel('Warped Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gtwidl_example.png', dpi=150, bbox_inches='tight')
    print("   Saved visualization to 'gtwidl_example.png'")
    plt.show()

    # Print sparsity statistics
    print("\n7. Sparsity statistics:")
    sparsity = (alphas > 1e-6).sum(dim=1).float().mean().item()
    print(f"   Average number of active atoms per sample: {sparsity:.2f}/{model.n_atoms}")
    print(f"   Average sparsity: {sparsity/model.n_atoms*100:.1f}%")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()