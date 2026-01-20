"""
Example classification experiment using GTWIDL
Demonstrates dictionary learning and classification on time series data
"""

import torch
import numpy as np
import argparse
from gtwidl import GTWIDL
from classification import GTWIDLClassifier, NearestDictionaryClassifier
from utils import (
    load_synthetic_dataset,
    normalize_time_series,
    to_torch,
    visualize_dictionary,
    visualize_reconstruction
)


def run_classification_experiment(args):
    """Run classification experiment"""
    print("="*70)
    print("GTWIDL Classification Experiment")
    print("="*70)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"\nUsing device: {device}")

    # Load data
    print(f"\nGenerating synthetic dataset...")
    print(f"  Samples: {args.n_samples}")
    print(f"  Length: {args.length}")
    print(f"  Classes: {args.n_classes}")

    X, y = load_synthetic_dataset(
        n_samples=args.n_samples,
        length=args.length,
        n_classes=args.n_classes,
        noise_level=args.noise_level,
        random_state=args.seed
    )

    # Normalize
    X = normalize_time_series(X, method='zscore')

    # Split train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Convert to torch
    X_train_torch = to_torch(X_train, device=device)
    X_test_torch = to_torch(X_test, device=device)

    # Train GTWIDL
    print(f"\n{'='*70}")
    print("Training GTWIDL Dictionary...")
    print(f"{'='*70}")
    print(f"  Dictionary atoms: {args.n_atoms}")
    print(f"  Atom length: {args.atom_length}")
    print(f"  Basis type: {args.basis_type}")
    print(f"  Number of basis functions: {args.n_basis}")
    print(f"  Sparsity parameter: {args.lambda_sparse}")

    gtwidl_model = GTWIDL(
        n_atoms=args.n_atoms,
        atom_length=args.atom_length,
        n_basis=args.n_basis,
        basis_type=args.basis_type,
        lambda_sparse=args.lambda_sparse,
        max_iter=args.max_iter,
        device=device,
        verbose=True
    )

    dictionary, alphas_train, betas_train = gtwidl_model.fit(X_train_torch)

    print("\nDictionary learning completed!")

    # Visualize dictionary
    if args.visualize:
        print("\nVisualizing learned dictionary...")
        visualize_dictionary(dictionary, n_cols=5, save_path='dictionary.png')

    # Visualize reconstruction example
    if args.visualize and len(X_train_torch) > 0:
        print("\nVisualizing reconstruction example...")
        sample_idx = 0
        x_sample = X_train_torch[sample_idx:sample_idx+1]

        # Reconstruct
        alpha_sample = alphas_train[sample_idx]
        beta_sample = betas_train[sample_idx]

        reconstruction = torch.zeros_like(x_sample[0])
        for k in range(gtwidl_model.n_atoms):
            if alpha_sample[k] > 1e-6:
                atom = dictionary[:, k]
                warped_atom = gtwidl_model.time_warping.warp_time_series(atom, beta_sample)
                reconstruction = reconstruction + alpha_sample[k] * warped_atom

        visualize_reconstruction(
            x_sample[0],
            reconstruction,
            title=f"Reconstruction Example (Class {y_train[sample_idx]})",
            save_path='reconstruction.png'
        )

    # Classification
    print(f"\n{'='*70}")
    print("Training Classifier...")
    print(f"{'='*70}")

    if args.classifier_type == 'standard':
        print(f"Using standard classifier: {args.base_classifier}")

        classifier = GTWIDLClassifier(
            gtwidl_model=gtwidl_model,
            classifier_type=args.base_classifier,
            classifier_params={'random_state': args.seed}
        )

        classifier.fit(X_train_torch, y_train)

        print("\nEvaluating on test set...")
        test_metrics = classifier.evaluate(X_test_torch, y_test, verbose=True)

    elif args.classifier_type == 'nearest_dictionary':
        print("Using nearest dictionary classifier")

        classifier = NearestDictionaryClassifier(
            n_atoms=args.n_atoms,
            atom_length=args.atom_length,
            gtwidl_params={
                'n_basis': args.n_basis,
                'basis_type': args.basis_type,
                'lambda_sparse': args.lambda_sparse,
                'max_iter': args.max_iter // 2,  # Less iterations per class
                'device': device,
                'verbose': False
            }
        )

        classifier.fit(X_train_torch, y_train)

        print("\nEvaluating on test set...")
        test_metrics = classifier.evaluate(X_test_torch, y_test, verbose=True)

    print(f"\n{'='*70}")
    print("Experiment Completed!")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GTWIDL Classification Experiment')

    # Data parameters
    parser.add_argument('--n_samples', type=int, default=200, help='Number of samples')
    parser.add_argument('--length', type=int, default=100, help='Time series length')
    parser.add_argument('--n_classes', type=int, default=3, help='Number of classes')
    parser.add_argument('--noise_level', type=float, default=0.1, help='Noise level')

    # GTWIDL parameters
    parser.add_argument('--n_atoms', type=int, default=10, help='Number of dictionary atoms')
    parser.add_argument('--atom_length', type=int, default=100, help='Length of atoms')
    parser.add_argument('--n_basis', type=int, default=5, help='Number of basis functions')
    parser.add_argument('--basis_type', type=str, default='polynomial',
                        choices=['polynomial', 'exponential', 'logarithmic', 'tanh', 'ispline'],
                        help='Type of basis functions')
    parser.add_argument('--lambda_sparse', type=float, default=0.1, help='Sparsity parameter')
    parser.add_argument('--max_iter', type=int, default=50, help='Maximum iterations')

    # Classifier parameters
    parser.add_argument('--classifier_type', type=str, default='standard',
                        choices=['standard', 'nearest_dictionary'],
                        help='Type of classifier')
    parser.add_argument('--base_classifier', type=str, default='svm',
                        choices=['svm', 'knn', 'rf'],
                        help='Base classifier for standard method')

    # Other parameters
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')

    args = parser.parse_args()

    run_classification_experiment(args)