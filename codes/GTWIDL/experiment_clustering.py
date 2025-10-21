"""
Example clustering experiment using GTWIDL
Demonstrates dictionary learning and clustering on time series data
"""

import torch
import numpy as np
import argparse
from gtwidl import GTWIDL
from clustering import GTWIDLClustering, AdaptiveDictionaryClustering, HierarchicalGTWIDLClustering
from utils import (
    load_synthetic_dataset,
    normalize_time_series,
    to_torch,
    visualize_dictionary,
    visualize_clustering
)


def run_clustering_experiment(args):
    """Run clustering experiment"""
    print("="*70)
    print("GTWIDL Clustering Experiment")
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
    print(f"  True clusters: {args.n_clusters}")

    X, y_true = load_synthetic_dataset(
        n_samples=args.n_samples,
        length=args.length,
        n_classes=args.n_clusters,
        noise_level=args.noise_level,
        random_state=args.seed
    )

    # Normalize
    X = normalize_time_series(X, method='zscore')

    print(f"  Dataset shape: {X.shape}")

    # Convert to torch
    X_torch = to_torch(X, device=device)

    # Clustering method selection
    print(f"\n{'='*70}")
    print(f"Clustering Method: {args.clustering_method}")
    print(f"{'='*70}")

    if args.clustering_method == 'standard':
        # First train GTWIDL dictionary
        print("\nTraining GTWIDL Dictionary...")
        print(f"  Dictionary atoms: {args.n_atoms}")
        print(f"  Atom length: {args.atom_length}")
        print(f"  Basis type: {args.basis_type}")

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

        dictionary, alphas, betas = gtwidl_model.fit(X_torch)

        print("\nDictionary learning completed!")

        # Visualize dictionary
        if args.visualize:
            print("\nVisualizing learned dictionary...")
            visualize_dictionary(dictionary, n_cols=5, save_path='clustering_dictionary.png')

        # Perform clustering
        print(f"\nPerforming clustering with {args.base_clustering}...")
        clustering_model = GTWIDLClustering(
            gtwidl_model=gtwidl_model,
            n_clusters=args.n_clusters,
            clustering_type=args.base_clustering
        )

        labels = clustering_model.fit(X_torch)

    elif args.clustering_method == 'adaptive':
        print("\nPerforming adaptive dictionary clustering...")
        print("  (Learning cluster-specific dictionaries)")

        clustering_model = AdaptiveDictionaryClustering(
            n_clusters=args.n_clusters,
            n_atoms=args.n_atoms,
            atom_length=args.atom_length,
            gtwidl_params={
                'n_basis': args.n_basis,
                'basis_type': args.basis_type,
                'lambda_sparse': args.lambda_sparse,
                'max_iter': args.max_iter // 2,
                'device': device,
                'verbose': False
            },
            max_iter=args.adaptive_max_iter
        )

        labels = clustering_model.fit(X_torch)

        # Visualize one cluster dictionary
        if args.visualize and 0 in clustering_model.cluster_models:
            print("\nVisualizing dictionary for cluster 0...")
            visualize_dictionary(
                clustering_model.cluster_models[0].dictionary,
                n_cols=5,
                save_path='cluster_0_dictionary.png'
            )

    elif args.clustering_method == 'hierarchical':
        # First train GTWIDL dictionary
        print("\nTraining GTWIDL Dictionary...")

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

        dictionary, alphas, betas = gtwidl_model.fit(X_torch)

        print("\nDictionary learning completed!")

        # Perform hierarchical clustering
        print(f"\nPerforming hierarchical clustering with {args.linkage} linkage...")
        clustering_model = HierarchicalGTWIDLClustering(
            gtwidl_model=gtwidl_model,
            n_clusters=args.n_clusters,
            linkage=args.linkage
        )

        labels = clustering_model.fit(X_torch)

    # Evaluate clustering
    print(f"\n{'='*70}")
    print("Clustering Evaluation")
    print(f"{'='*70}")

    metrics = clustering_model.evaluate(X_torch, true_labels=y_true, verbose=True)

    # Visualize clustering results
    if args.visualize:
        print("\nVisualizing clustering results...")
        visualize_clustering(
            X,
            labels,
            title=f"Clustering Results ({args.clustering_method})",
            max_samples_per_cluster=10,
            save_path='clustering_results.png'
        )

        print("\nVisualizing ground truth clusters...")
        visualize_clustering(
            X,
            y_true,
            title="Ground Truth Clusters",
            max_samples_per_cluster=10,
            save_path='ground_truth_clusters.png'
        )

    print(f"\n{'='*70}")
    print("Experiment Completed!")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GTWIDL Clustering Experiment')

    # Data parameters
    parser.add_argument('--n_samples', type=int, default=150, help='Number of samples')
    parser.add_argument('--length', type=int, default=100, help='Time series length')
    parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters')
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

    # Clustering parameters
    parser.add_argument('--clustering_method', type=str, default='standard',
                        choices=['standard', 'adaptive', 'hierarchical'],
                        help='Clustering method')
    parser.add_argument('--base_clustering', type=str, default='kmeans',
                        choices=['kmeans', 'hierarchical', 'dbscan', 'spectral'],
                        help='Base clustering algorithm for standard method')
    parser.add_argument('--linkage', type=str, default='average',
                        choices=['average', 'complete', 'single'],
                        help='Linkage for hierarchical clustering')
    parser.add_argument('--adaptive_max_iter', type=int, default=20,
                        help='Max iterations for adaptive clustering')

    # Other parameters
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')

    args = parser.parse_args()

    run_clustering_experiment(args)