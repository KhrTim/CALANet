"""
Utility functions for SAGoG model training and evaluation.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """Dataset class for multivariate time series classification."""

    def __init__(self, data, labels, transform=None):
        """
        Args:
            data: Time series data [num_samples, num_variables, seq_len]
            labels: Class labels [num_samples]
            transform: Optional data augmentation transform
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


class TimeSeriesAugmentation:
    """Data augmentation techniques for time series."""

    @staticmethod
    def jitter(x, sigma=0.03):
        """Add random noise."""
        noise = torch.randn_like(x) * sigma
        return x + noise

    @staticmethod
    def scaling(x, sigma=0.1):
        """Scale time series."""
        factor = torch.randn(x.shape[0], 1) * sigma + 1.0
        return x * factor.to(x.device)

    @staticmethod
    def time_warp(x, sigma=0.2, knot=4):
        """Time warping augmentation."""
        # Simplified version - random time dilation/compression
        seq_len = x.shape[-1]
        random_warps = torch.cumsum(torch.randn(knot + 2) * sigma + 1.0, dim=0)
        random_warps = torch.nn.functional.interpolate(
            random_warps.unsqueeze(0).unsqueeze(0),
            size=seq_len,
            mode='linear',
            align_corners=False
        ).squeeze()

        # Normalize to [0, seq_len-1]
        random_warps = (random_warps - random_warps.min()) / (random_warps.max() - random_warps.min()) * (seq_len - 1)
        random_warps = random_warps.long().clamp(0, seq_len - 1)

        return x[:, random_warps]

    @staticmethod
    def window_slice(x, reduce_ratio=0.9):
        """Randomly slice a window from time series."""
        seq_len = x.shape[-1]
        target_len = int(seq_len * reduce_ratio)
        if target_len >= seq_len:
            return x

        start = torch.randint(0, seq_len - target_len + 1, (1,)).item()
        end = start + target_len

        sliced = x[:, start:end]

        # Interpolate back to original length
        sliced = torch.nn.functional.interpolate(
            sliced.unsqueeze(0),
            size=seq_len,
            mode='linear',
            align_corners=False
        ).squeeze(0)

        return sliced

    @staticmethod
    def random_augment(x):
        """Apply random augmentation."""
        aug_type = torch.randint(0, 4, (1,)).item()

        if aug_type == 0:
            return TimeSeriesAugmentation.jitter(x)
        elif aug_type == 1:
            return TimeSeriesAugmentation.scaling(x)
        elif aug_type == 2:
            return TimeSeriesAugmentation.time_warp(x)
        else:
            return TimeSeriesAugmentation.window_slice(x)


def compute_metrics(y_true, y_pred):
    """Compute classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall
    }


def train_epoch(model, dataloader, optimizer, criterion, device, use_augmentation=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Apply augmentation
        if use_augmentation:
            x = TimeSeriesAugmentation.random_augment(x)

        optimizer.zero_grad()

        # Forward pass
        logits = model(x)
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Predictions
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = avg_loss

    return metrics


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = avg_loss

    return metrics


def load_ucr_dataset(dataset_name, data_path='./data'):
    """
    Load UCR time series dataset.

    Args:
        dataset_name: Name of the UCR dataset
        data_path: Path to data directory

    Returns:
        X_train, y_train, X_test, y_test
    """
    import os
    from scipy.io import arff
    import pandas as pd

    train_file = os.path.join(data_path, dataset_name, f"{dataset_name}_TRAIN.arff")
    test_file = os.path.join(data_path, dataset_name, f"{dataset_name}_TEST.arff")

    # Load ARFF files
    train_data, _ = arff.loadarff(train_file)
    test_data, _ = arff.loadarff(test_file)

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Separate features and labels
    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values

    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Add channel dimension if univariate
    if len(X_train.shape) == 2:
        X_train = X_train[:, np.newaxis, :]
        X_test = X_test[:, np.newaxis, :]

    return X_train, y_train, X_test, y_test


def generate_synthetic_data(num_samples=1000, num_variables=10, seq_len=100, num_classes=3, seed=42):
    """
    Generate synthetic multivariate time series data for testing.

    Args:
        num_samples: Number of samples
        num_variables: Number of variables in time series
        seq_len: Length of time series
        num_classes: Number of classes
        seed: Random seed

    Returns:
        X_train, y_train, X_test, y_test
    """
    np.random.seed(seed)

    def generate_class_pattern(n, class_id):
        """Generate time series with class-specific patterns."""
        t = np.linspace(0, 4 * np.pi, seq_len)
        X = np.zeros((n, num_variables, seq_len))

        for i in range(n):
            for v in range(num_variables):
                # Different classes have different frequency components
                freq1 = (class_id + 1) * 0.5 + np.random.rand() * 0.3
                freq2 = (class_id + 1) * 1.0 + np.random.rand() * 0.5

                signal = (np.sin(freq1 * t + np.random.rand() * 2 * np.pi) +
                          0.5 * np.sin(freq2 * t + np.random.rand() * 2 * np.pi))

                # Add noise
                signal += np.random.randn(seq_len) * 0.1

                # Add trend for some classes
                if class_id == 1:
                    signal += np.linspace(0, 0.5, seq_len)
                elif class_id == 2:
                    signal += np.linspace(0.5, 0, seq_len)

                X[i, v, :] = signal

        return X

    # Generate data for each class
    samples_per_class = num_samples // num_classes
    X_list = []
    y_list = []

    for c in range(num_classes):
        X_c = generate_class_pattern(samples_per_class, c)
        y_c = np.full(samples_per_class, c)
        X_list.append(X_c)
        y_list.append(y_c)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split train/test
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, y_train, X_test, y_test


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' or 'max' for loss or accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, filepath)


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']

    return epoch, metrics