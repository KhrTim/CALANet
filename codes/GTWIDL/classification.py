"""
Classification module for GTWIDL
Uses learned dictionaries and sparse representations for time series classification
"""

import torch
import numpy as np
from typing import Tuple, Optional
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


class GTWIDLClassifier:
    """Classifier using GTWIDL features"""

    def __init__(self,
                 gtwidl_model,
                 classifier_type: str = 'svm',
                 classifier_params: Optional[dict] = None):
        """
        Args:
            gtwidl_model: Trained GTWIDL model
            classifier_type: Type of classifier ('svm', 'knn', 'rf')
            classifier_params: Parameters for the classifier
        """
        self.gtwidl_model = gtwidl_model
        self.classifier_type = classifier_type

        if classifier_params is None:
            classifier_params = {}

        # Initialize classifier
        if classifier_type == 'svm':
            self.classifier = SVC(**classifier_params)
        elif classifier_type == 'knn':
            self.classifier = KNeighborsClassifier(**classifier_params)
        elif classifier_type == 'rf':
            self.classifier = RandomForestClassifier(**classifier_params)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def fit(self, X: torch.Tensor, y: np.ndarray):
        """
        Train classifier on time series data

        Args:
            X: Time series data (n_samples, length) or (n_samples, length, n_dims)
            y: Labels (n_samples,)
        """
        # Extract features using GTWIDL
        features = self._extract_features(X)

        # Train classifier
        self.classifier.fit(features, y)

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Predict labels for time series

        Args:
            X: Time series data (n_samples, length) or (n_samples, length, n_dims)

        Returns:
            predictions: Predicted labels (n_samples,)
        """
        features = self._extract_features(X)
        return self.classifier.predict(features)

    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        """
        Predict class probabilities for time series

        Args:
            X: Time series data (n_samples, length) or (n_samples, length, n_dims)

        Returns:
            probabilities: Class probabilities (n_samples, n_classes)
        """
        features = self._extract_features(X)
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(features)
        else:
            raise AttributeError(f"Classifier {self.classifier_type} does not support predict_proba")

    def evaluate(self, X: torch.Tensor, y: np.ndarray, verbose: bool = True) -> dict:
        """
        Evaluate classifier on test data

        Args:
            X: Time series data (n_samples, length) or (n_samples, length, n_dims)
            y: True labels (n_samples,)
            verbose: Print evaluation results

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        predictions = self.predict(X)

        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average='weighted')

        metrics = {
            'accuracy': accuracy,
            'f1_score': f1
        }

        if verbose:
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score (weighted): {f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(y, predictions))

        return metrics

    def _extract_features(self, X: torch.Tensor) -> np.ndarray:
        """
        Extract features from time series using GTWIDL

        Args:
            X: Time series data

        Returns:
            features: Feature matrix (n_samples, n_features)
        """
        # Get sparse coefficients and warping coefficients
        alphas, betas = self.gtwidl_model.transform(X)

        # Convert to numpy
        features = alphas.cpu().numpy()

        # Optionally add warping-based features
        # For now, use sparse coefficients as features
        return features


class NearestDictionaryClassifier:
    """
    Classifier based on distance to class-specific dictionaries
    Each class has its own dictionary, and classification is based on
    reconstruction error or distance to dictionaries
    """

    def __init__(self,
                 n_atoms: int,
                 atom_length: int,
                 gtwidl_params: Optional[dict] = None):
        """
        Args:
            n_atoms: Number of atoms per class dictionary
            atom_length: Length of dictionary atoms
            gtwidl_params: Parameters for GTWIDL models
        """
        from gtwidl import GTWIDL

        self.n_atoms = n_atoms
        self.atom_length = atom_length
        self.gtwidl_params = gtwidl_params or {}

        self.class_models = {}
        self.classes = None

    def fit(self, X: torch.Tensor, y: np.ndarray):
        """
        Train class-specific dictionaries

        Args:
            X: Time series data (n_samples, length) or (n_samples, length, n_dims)
            y: Labels (n_samples,)
        """
        from gtwidl import GTWIDL

        self.classes = np.unique(y)

        # Train a dictionary for each class
        for class_label in self.classes:
            print(f"Training dictionary for class {class_label}...")

            # Get samples for this class
            class_mask = (y == class_label)
            X_class = X[class_mask]

            # Train GTWIDL model for this class
            model = GTWIDL(
                n_atoms=self.n_atoms,
                atom_length=self.atom_length,
                **self.gtwidl_params
            )
            model.fit(X_class)

            self.class_models[class_label] = model

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Predict labels based on reconstruction error with class dictionaries

        Args:
            X: Time series data (n_samples, length) or (n_samples, length, n_dims)

        Returns:
            predictions: Predicted labels (n_samples,)
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        # For each sample, compute reconstruction error with each class dictionary
        for i in range(n_samples):
            x_i = X[i:i+1]
            min_error = float('inf')
            best_class = None

            for class_label in self.classes:
                model = self.class_models[class_label]

                # Transform using class dictionary
                alphas, betas = model.transform(x_i)

                # Compute reconstruction error
                error = self._reconstruction_error(x_i, model, alphas[0], betas[0])

                if error < min_error:
                    min_error = error
                    best_class = class_label

            predictions[i] = best_class

        return predictions

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

    def evaluate(self, X: torch.Tensor, y: np.ndarray, verbose: bool = True) -> dict:
        """
        Evaluate classifier on test data

        Args:
            X: Time series data
            y: True labels
            verbose: Print evaluation results

        Returns:
            metrics: Dictionary of evaluation metrics
        """
        predictions = self.predict(X)

        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average='weighted')

        metrics = {
            'accuracy': accuracy,
            'f1_score': f1
        }

        if verbose:
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score (weighted): {f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(y, predictions))

        return metrics


class DistanceMetricClassifier:
    """
    k-NN classifier using GTWIDL-based distance metric
    Distance is computed as the minimum reconstruction error after warping
    """

    def __init__(self, gtwidl_model, k: int = 5):
        """
        Args:
            gtwidl_model: Trained GTWIDL model
            k: Number of neighbors
        """
        self.gtwidl_model = gtwidl_model
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: torch.Tensor, y: np.ndarray):
        """Store training data"""
        self.X_train = X
        self.y_train = y

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Predict using k-NN with GTWIDL distance

        Args:
            X: Time series data (n_samples, length) or (n_samples, length, n_dims)

        Returns:
            predictions: Predicted labels (n_samples,)
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            x_i = X[i:i+1]

            # Compute distances to all training samples
            distances = []
            for j in range(len(self.X_train)):
                x_j = self.X_train[j:j+1]
                dist = self._compute_distance(x_i, x_j)
                distances.append(dist)

            # Get k nearest neighbors
            distances = np.array(distances)
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]

            # Vote
            predictions[i] = np.bincount(k_nearest_labels.astype(int)).argmax()

        return predictions

    def _compute_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """
        Compute GTWIDL distance between two time series
        Distance = min reconstruction error with warping
        """
        # Transform both series
        alpha1, beta1 = self.gtwidl_model.transform(x1)
        alpha2, beta2 = self.gtwidl_model.transform(x2)

        # Reconstruct both
        x1_recon = self._reconstruct(alpha1[0], beta1[0])
        x2_recon = self._reconstruct(alpha2[0], beta2[0])

        # Compute distance
        dist = torch.mean((x1_recon - x2_recon) ** 2).item()
        return dist

    def _reconstruct(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Reconstruct time series from sparse coefficients"""
        reconstruction = torch.zeros(self.gtwidl_model.atom_length, device=alpha.device)

        for k in range(self.gtwidl_model.n_atoms):
            if alpha[k] > 1e-6:
                atom = self.gtwidl_model.dictionary[:, k]
                warped_atom = self.gtwidl_model.time_warping.warp_time_series(atom, beta)
                reconstruction = reconstruction + alpha[k] * warped_atom

        return reconstruction

    def evaluate(self, X: torch.Tensor, y: np.ndarray, verbose: bool = True) -> dict:
        """Evaluate classifier"""
        predictions = self.predict(X)

        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions, average='weighted')

        metrics = {
            'accuracy': accuracy,
            'f1_score': f1
        }

        if verbose:
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score (weighted): {f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(y, predictions))

        return metrics