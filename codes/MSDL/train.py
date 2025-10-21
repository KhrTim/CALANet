import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from msdl import MSDL, MSDLWithAttention


class MSDLTrainer:
    """Trainer class for MSDL models"""

    def __init__(
        self,
        model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=0.001,
        weight_decay=1e-4
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, train_loader, test_loader, epochs=100, early_stopping_patience=20):
        best_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            test_loss, test_acc = self.evaluate(test_loader)

            self.scheduler.step(test_loss)

            print(f'Epoch: {epoch+1:03d}/{epochs:03d} | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')

            if test_acc > best_acc:
                best_acc = test_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f'  -> Best model saved with accuracy: {best_acc:.2f}%')
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break

        print(f'\nTraining completed. Best test accuracy: {best_acc:.2f}%')
        return best_acc


def load_ucr_dataset(dataset_name, data_path='./data/'):
    """
    Load UCR time series dataset
    Expected format: .txt or .tsv files with format:
    label value1 value2 value3 ...
    """
    try:
        train_data = np.loadtxt(f'{data_path}/{dataset_name}/{dataset_name}_TRAIN.tsv')
        test_data = np.loadtxt(f'{data_path}/{dataset_name}/{dataset_name}_TEST.tsv')
    except:
        try:
            train_data = np.loadtxt(f'{data_path}/{dataset_name}/{dataset_name}_TRAIN.txt')
            test_data = np.loadtxt(f'{data_path}/{dataset_name}/{dataset_name}_TEST.txt')
        except:
            raise FileNotFoundError(f"Could not find dataset {dataset_name} in {data_path}")

    # Separate labels and features
    X_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0]

    # Convert labels to start from 0
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_train = np.array([label_map[label] for label in y_train])
    y_test = np.array([label_map[label] for label in y_test])

    # Add channel dimension
    X_train = X_train[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

    # Normalize
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    return X_train, y_train, X_test, y_test, len(unique_labels)


def create_synthetic_dataset(n_samples=1000, seq_length=128, n_classes=10):
    """Create synthetic time series data for testing"""
    X = np.random.randn(n_samples, 1, seq_length)
    y = np.random.randint(0, n_classes, n_samples)

    # Add some pattern to make it learnable
    for i in range(n_samples):
        class_label = y[i]
        frequency = (class_label + 1) * 0.1
        t = np.linspace(0, 10, seq_length)
        X[i, 0, :] += np.sin(2 * np.pi * frequency * t)

    return X, y


if __name__ == "__main__":
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    USE_SYNTHETIC = True  # Set to False to use real UCR dataset

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if USE_SYNTHETIC:
        # Create synthetic dataset
        print("Creating synthetic dataset...")
        X_train, y_train = create_synthetic_dataset(n_samples=800, seq_length=128, n_classes=5)
        X_test, y_test = create_synthetic_dataset(n_samples=200, seq_length=128, n_classes=5)
        num_classes = 5
        input_channels = 1
        sequence_length = 128
    else:
        # Load UCR dataset (you need to specify dataset name)
        dataset_name = "ECG200"  # Change this to your dataset
        print(f"Loading {dataset_name} dataset...")
        X_train, y_train, X_test, y_test, num_classes = load_ucr_dataset(dataset_name)
        input_channels = X_train.shape[1]
        sequence_length = X_train.shape[2]

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Number of classes: {num_classes}")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    print("\nInitializing MSDL model...")
    model = MSDL(
        input_channels=input_channels,
        num_classes=num_classes,
        multiscale_channels=64,
        kernel_sizes=[3, 5, 7],
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.5
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    trainer = MSDLTrainer(model, device=device, learning_rate=LEARNING_RATE)
    best_acc = trainer.train(train_loader, test_loader, epochs=EPOCHS)

    print(f"\nFinal best accuracy: {best_acc:.2f}%")