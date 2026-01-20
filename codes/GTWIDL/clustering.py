"""
Clustering module for GTWIDL
Uses learned dictionaries and sparse representations for time series clustering
"""

import torch
import numpy as np
from typing import Tuple, Optional
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


class GTWIDLClustering:
    """Clustering using GTWIDL features"""

    def __init__(self,
                 gtwidl_model,
                 n_clusters: int,
                 clustering_type: str = 'kmeans',
                 clustering_params: Optional[dict] = None):
        """
        Args:
            gtwidl_model: Trained GTWIDL model
            n_clusters: Number of clusters
            clustering_type: Type of clustering ('kmeans', 'hierarchical', 'dbscan', 'spectral')
            clustering_params: Parameters for the clustering algorithm
        """
        self.gtwidl_model = gtwidl_model
        self.n_clusters = n_clusters
        self.clustering_type = clustering_type

        if clustering_params is None:
            clustering_params = {}

        # Initialize clustering algorithm
        if clustering_type == 'kmeans':
            self.clustering = KMeans(n_clusters=n_clusters, **clustering_params)
        elif clustering_type == 'hierarchical':
            self.clustering = AgglomerativeClustering(n_clusters=n_clusters, **clustering_params)
        elif clustering_type == 'dbscan':
            self.clustering = DBSCAN(**clustering_params)
        elif clustering_type == 'spectral':
            self.clustering = SpectralClustering(n_clusters=n_clusters, **clustering_params)
        else:
            raise ValueError(f"Unknown clustering type: {clustering_type}")

        self.labels_ = None

    def fit(self, X: torch.Tensor):
        """
        Cluster time series data

        Args:
            X: Time series data (n_samples, length) or (n_samples, length, n_dims)
        """
        # Extract features using GTWIDL
        features = self._extract_features(X)

        # Perform clustering
        self.labels_ = self.clustering.fit_predict(features)

        return self.labels_

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Predict cluster labels for new time series

        Args:
            X: Time series data (n_samples, length) or (n_samples, length, n_dims)

        Returns:
            labels: Cluster labels (n_samples,)
        """
        features = self._extract_features(X)

        if hasattr(self.clustering, 'predict'):
            return self.clustering.predict(features)
        else:
            raise AttributeError(f"Clustering method {self.clustering_type} does not support predict")

    def fit_predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Fit and predict cluster labels

        Args:
            X: Time series data (n_samples, length) or (n_samples, length, n_dims)

        Returns:
            labels: Cluster labels (n_samples,)
        """
        return self.fit(X)

    def evaluate(self, X: torch.Tensor, true_labels: Optional[np.ndarray] = None, verbose: bool = True) -> dict:
        """
        Evaluate clustering quality

        Args:
            X: Time series data
            true_labels: True labels (if available for external validation)
            verbose: Print evaluation results

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        features = self._extract_features(X)

        metrics = {}

        # Internal validation metrics
        if len(np.unique(self.labels_)) > 1:
            metrics['silhouette_score'] = silhouette_score(features, self.labels_)
            metrics['davies_bouldin_score'] = davies_bouldin_score(features, self.labels_)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, self.labels_)

        # External validation metrics (if true labels provided)
        if true_labels is not None:
            metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, self.labels_)
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, self.labels_)

        if verbose:
            print("Clustering Evaluation Metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        return metrics

    def _extract_features(self, X: torch.Tensor) -> np.ndarray:
        """
        Extract features from time series using GTWIDL

        Args:
            X: Time series data

        Returns:
            features: Feature matrix (n_samples, n_features)
        """
        # Get sparse coefficients
        alphas, betas = self.gtwidl_model.transform(X)

        # Convert to numpy
        features = alphas.cpu().numpy()

        return features


class AdaptiveDictionaryClustering:
    """
    Clustering with simultaneous dictionary learning
    Learns cluster-specific dictionaries while clustering
    """

    def __init__(self,
                 n_clusters: int,
                 n_atoms: int,
                 atom_length: int,
                 gtwidl_params: Optional[dict] = None,
                 max_iter: int = 50):
        """
        Args:
            n_clusters: Number of clusters
            n_atoms: Number of atoms per cluster dictionary
            atom_length: Length of dictionary atoms
            gtwidl_params: Parameters for GTWIDL models
            max_iter: Maximum number of iterations
        """
        from gtwidl import GTWIDL

        self.n_clusters = n_clusters
        self.n_atoms = n_atoms
        self.atom_length = atom_length
        self.gtwidl_params = gtwidl_params or {}
        self.max_iter = max_iter

        self.cluster_models = None
        self.labels_ = None

    def fit(self, X: torch.Tensor):
        """
        Cluster time series with simultaneous dictionary learning

        Args:
            X: Time series data (n_samples, length) or (n_samples, length, n_dims)
        """
        from gtwidl import GTWIDL

        n_samples = X.shape[0]

        # Initialize cluster assignments randomly
        self.labels_ = np.random.randint(0, self.n_clusters, n_samples)

        prev_labels = None

        for iteration in range(self.max_iter):
            print(f"Iteration {iteration + 1}/{self.max_iter}")

            # Step 1: Update cluster dictionaries
            self.cluster_models = {}
            for cluster_id in range(self.n_clusters):
                cluster_mask = (self.labels_ == cluster_id)
                X_cluster = X[cluster_mask]

                if len(X_cluster) > 0:
                    # Learn dictionary for this cluster
                    model = GTWIDL(
                        n_atoms=self.n_atoms,
                        atom_length=self.atom_length,
                        verbose=False,
                        **self.gtwidl_params
                    )
                    model.fit(X_cluster)
                    self.cluster_models[cluster_id] = model

            # Step 2: Reassign samples to clusters
            new_labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                x_i = X[i:i+1]
                min_error = float('inf')
                best_cluster = 0

                for cluster_id in range(self.n_clusters):
                    if cluster_id in self.cluster_models:
                        model = self.cluster_models[cluster_id]
                        alphas, betas = model.transform(x_i)
                        error = self._reconstruction_error(x_i, model, alphas[0], betas[0])

                        if error < min_error:
                            min_error = error
                            best_cluster = cluster_id

                new_labels[i] = best_cluster

            # Check convergence
            if prev_labels is not None and np.array_equal(new_labels, prev_labels):
                print(f"Converged at iteration {iteration + 1}")
                break

            self.labels_ = new_labels
            prev_labels = new_labels.copy()

            # Print cluster sizes
            unique, counts = np.unique(self.labels_, return_counts=True)
            print(f"  Cluster sizes: {dict(zip(unique, counts))}")

        return self.labels_

    def _reconstruction_error(self, x: torch.Tensor, model, alpha: torch.Tensor, beta: torch.Tensor) -> float:
        """Compute reconstruction error for a sample"""
        x = x.squeeze(0)

        # Reconstruct signal
        reconstruction = torch.zeros_like(x)
        for k in range(model.n_atoms):
            if alpha[k] > 1e-6:
                atom = model.dictionary[:, k]
                warped_atom = model.time_warping.warp_time_series(atom, beta)
                reconstruction = reconstruction + alpha[k] * warped_atom

        # Compute MSE
        error = torch.mean((x - reconstruction) ** 2).item()
        return error

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Predict cluster labels for new time series

        Args:
            X: Time series data (n_samples, length) or (n_samples, length, n_dims)

        Returns:
            labels: Cluster labels (n_samples,)
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            x_i = X[i:i+1]
            min_error = float('inf')
            best_cluster = 0

            for cluster_id in range(self.n_clusters):
                if cluster_id in self.cluster_models:
                    model = self.cluster_models[cluster_id]
                    alphas, betas = model.transform(x_i)
                    error = self._reconstruction_error(x_i, model, alphas[0], betas[0])

                    if error < min_error:
                        min_error = error
                        best_cluster = cluster_id

            labels[i] = best_cluster

        return labels

    def evaluate(self, X: torch.Tensor, true_labels: Optional[np.ndarray] = None, verbose: bool = True) -> dict:
        """
        Evaluate clustering quality

        Args:
            X: Time series data
            true_labels: True labels (if available)
            verbose: Print evaluation results

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Extract features for metric computation
        from gtwidl import GTWIDL

        # Use first cluster's model for feature extraction (or train a new one)
        if 0 in self.cluster_models:
            model = self.cluster_models[0]
        else:
            model = GTWIDL(
                n_atoms=self.n_atoms,
                atom_length=self.atom_length,
                verbose=False,
                **self.gtwidl_params
            )
            model.fit(X)

        alphas, _ = model.transform(X)
        features = alphas.cpu().numpy()

        metrics = {}

        # Internal validation metrics
        if len(np.unique(self.labels_)) > 1:
            metrics['silhouette_score'] = silhouette_score(features, self.labels_)
            metrics['davies_bouldin_score'] = davies_bouldin_score(features, self.labels_)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, self.labels_)

        # External validation metrics
        if true_labels is not None:
            metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, self.labels_)
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, self.labels_)

        if verbose:
            print("Clustering Evaluation Metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        return metrics


class HierarchicalGTWIDLClustering:
    """
    Hierarchical clustering using GTWIDL-based distance matrix
    """

    def __init__(self,
                 gtwidl_model,
                 n_clusters: int,
                 linkage: str = 'average'):
        """
        Args:
            gtwidl_model: Trained GTWIDL model
            n_clusters: Number of clusters
            linkage: Linkage criterion ('average', 'complete', 'single', 'ward')
        """
        self.gtwidl_model = gtwidl_model
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def fit(self, X: torch.Tensor):
        """
        Perform hierarchical clustering

        Args:
            X: Time series data (n_samples, length) or (n_samples, length, n_dims)
        """
        # Compute pairwise distance matrix
        print("Computing pairwise distance matrix...")
        distance_matrix = self._compute_distance_matrix(X)

        # Perform hierarchical clustering
        print("Performing hierarchical clustering...")
        clustering = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric='precomputed',
            linkage=self.linkage
        )
        self.labels_ = clustering.fit_predict(distance_matrix)

        return self.labels_

    def _compute_distance_matrix(self, X: torch.Tensor) -> np.ndarray:
        """
        Compute pairwise distance matrix using GTWIDL reconstruction error

        Args:
            X: Time series data

        Returns:
            distance_matrix: Pairwise distance matrix (n_samples, n_samples)
        """
        n_samples = X.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))

        # Transform all samples
        alphas, betas = self.gtwidl_model.transform(X)

        # Reconstruct all samples
        reconstructions = []
        for i in range(n_samples):
            recon = self._reconstruct(alphas[i], betas[i])
            reconstructions.append(recon)

        # Compute pairwise distances
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                # Distance based on reconstruction similarity
                dist = torch.mean((reconstructions[i] - reconstructions[j]) ** 2).item()
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix

    def _reconstruct(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Reconstruct time series from sparse coefficients"""
        reconstruction = torch.zeros(self.gtwidl_model.atom_length, device=alpha.device)

        for k in range(self.gtwidl_model.n_atoms):
            if alpha[k] > 1e-6:
                atom = self.gtwidl_model.dictionary[:, k]
                warped_atom = self.gtwidl_model.time_warping.warp_time_series(atom, beta)
                reconstruction = reconstruction + alpha[k] * warped_atom

        return reconstruction

    def evaluate(self, X: torch.Tensor, true_labels: Optional[np.ndarray] = None, verbose: bool = True) -> dict:
        """Evaluate clustering quality"""
        # Extract features
        alphas, _ = self.gtwidl_model.transform(X)
        features = alphas.cpu().numpy()

        metrics = {}

        # Internal validation metrics
        if len(np.unique(self.labels_)) > 1:
            metrics['silhouette_score'] = silhouette_score(features, self.labels_)
            metrics['davies_bouldin_score'] = davies_bouldin_score(features, self.labels_)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, self.labels_)

        # External validation metrics
        if true_labels is not None:
            metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, self.labels_)
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, self.labels_)

        if verbose:
            print("Clustering Evaluation Metrics:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        return metrics