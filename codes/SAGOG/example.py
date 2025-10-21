"""
Simple example demonstrating SAGoG usage.
"""

import torch
from sagog_model import SAGoG, SAGoGLite
from utils import generate_synthetic_data, TimeSeriesDataset, train_epoch, evaluate
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def quick_test():
    """Quick test of SAGoG model."""
    print("SAGoG Quick Test")
    print("=" * 60)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X_train, y_train, X_test, y_test = generate_synthetic_data(
        num_samples=500,
        num_variables=5,
        seq_len=50,
        num_classes=3,
        seed=42
    )
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")

    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create model
    print("\n2. Creating SAGoG model...")
    model = SAGoG(
        num_variables=5,
        seq_len=50,
        num_classes=3,
        hidden_dim=32,
        graph_hidden_dim=64,
        num_graph_layers=2,
        num_windows=3,
        graph_construction='adaptive',
        gnn_type='gcn'
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for a few epochs
    print("\n3. Training for 10 epochs...")
    for epoch in range(10):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        print(f"   Epoch {epoch+1:2d} - "
              f"Train Acc: {train_metrics['accuracy']:.3f}, "
              f"Test Acc: {test_metrics['accuracy']:.3f}, "
              f"Test F1: {test_metrics['f1_macro']:.3f}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")


def test_different_configurations():
    """Test different model configurations."""
    print("\n\nTesting Different Configurations")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate data
    X_train, y_train, X_test, y_test = generate_synthetic_data(
        num_samples=200,
        num_variables=8,
        seq_len=100,
        num_classes=4,
        seed=42
    )

    # Test different graph construction methods
    methods = ['correlation', 'adaptive', 'dtw']

    for method in methods:
        print(f"\n{method.capitalize()} Graph Construction:")

        model = SAGoG(
            num_variables=8,
            seq_len=100,
            num_classes=4,
            hidden_dim=32,
            graph_hidden_dim=64,
            num_graph_layers=1,
            num_windows=4,
            graph_construction=method,
            gnn_type='gcn'
        ).to(device)

        # Quick forward pass
        x = torch.FloatTensor(X_test[:4]).to(device)
        with torch.no_grad():
            output = model(x)

        print(f"   Output shape: {output.shape}")
        print(f"   Predictions: {torch.argmax(output, dim=1).cpu().numpy()}")


def test_lite_model():
    """Test SAGoGLite model."""
    print("\n\nTesting SAGoGLite (Lightweight Version)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Generate data
    X_train, y_train, X_test, y_test = generate_synthetic_data(
        num_samples=300,
        num_variables=6,
        seq_len=80,
        num_classes=2,
        seed=42
    )

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Create lite model
    model = SAGoGLite(
        num_variables=6,
        seq_len=80,
        num_classes=2,
        hidden_dim=32,
        graph_hidden_dim=64,
        num_windows=3
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train briefly
    print("\nTraining for 5 epochs...")
    for epoch in range(5):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        print(f"   Epoch {epoch+1} - "
              f"Train Acc: {train_metrics['accuracy']:.3f}, "
              f"Test Acc: {test_metrics['accuracy']:.3f}")

    print("\nLite model test completed!")


def inference_example():
    """Example of using trained model for inference."""
    print("\n\nInference Example")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model
    model = SAGoG(
        num_variables=5,
        seq_len=50,
        num_classes=3,
        hidden_dim=32,
        graph_hidden_dim=64,
        num_graph_layers=2,
        num_windows=3
    ).to(device)

    model.eval()

    # Create sample input
    # Shape: [batch_size, num_variables, seq_len]
    sample_input = torch.randn(1, 5, 50).to(device)

    print(f"Input shape: {sample_input.shape}")

    # Inference
    with torch.no_grad():
        logits = model(sample_input)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1)

    print(f"Logits: {logits[0].cpu().numpy()}")
    print(f"Probabilities: {probabilities[0].cpu().numpy()}")
    print(f"Predicted class: {prediction.item()}")


if __name__ == '__main__':
    # Run all tests
    quick_test()
    test_different_configurations()
    test_lite_model()
    inference_example()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)