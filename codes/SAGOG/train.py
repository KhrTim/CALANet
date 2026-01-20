"""
Training script for SAGoG model on multivariate time series classification.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

from sagog_model import SAGoG, SAGoGLite
from utils import (
    TimeSeriesDataset,
    generate_synthetic_data,
    train_epoch,
    evaluate,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train SAGoG model')

    # Data parameters
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        help='Dataset name (synthetic or UCR dataset name)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples for synthetic data')
    parser.add_argument('--num_variables', type=int, default=10,
                        help='Number of variables in time series')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='Length of time series')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes')

    # Model parameters
    parser.add_argument('--model', type=str, default='sagog', choices=['sagog', 'sagog_lite'],
                        help='Model architecture')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--graph_hidden_dim', type=int, default=128,
                        help='Graph hidden dimension')
    parser.add_argument('--num_graph_layers', type=int, default=2,
                        help='Number of graph layers')
    parser.add_argument('--num_windows', type=int, default=5,
                        help='Number of windows to split time series')
    parser.add_argument('--graph_construction', type=str, default='adaptive',
                        choices=['correlation', 'adaptive', 'dtw'],
                        help='Graph construction method')
    parser.add_argument('--gnn_type', type=str, default='gcn',
                        choices=['gcn', 'gat'],
                        help='Type of GNN')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Use data augmentation')

    # Other parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='Path to checkpoint to load')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 80)
    print("SAGoG Training Script")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    if args.dataset == 'synthetic':
        X_train, y_train, X_test, y_test = generate_synthetic_data(
            num_samples=args.num_samples,
            num_variables=args.num_variables,
            seq_len=args.seq_len,
            num_classes=args.num_classes,
            seed=args.seed
        )
        print(f"Generated synthetic data:")
        print(f"  Train: {X_train.shape}, Labels: {y_train.shape}")
        print(f"  Test: {X_test.shape}, Labels: {y_test.shape}")

        # Update parameters based on data
        args.num_variables = X_train.shape[1]
        args.seq_len = X_train.shape[2]
        args.num_classes = len(np.unique(y_train))
    else:
        # Load UCR dataset
        from utils import load_ucr_dataset
        X_train, y_train, X_test, y_test = load_ucr_dataset(args.dataset, args.data_path)

        args.num_variables = X_train.shape[1]
        args.seq_len = X_train.shape[2]
        args.num_classes = len(np.unique(y_train))

        print(f"Loaded {args.dataset} dataset:")
        print(f"  Train: {X_train.shape}, Labels: {y_train.shape}")
        print(f"  Test: {X_test.shape}, Labels: {y_test.shape}")

    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print("\nCreating model...")
    if args.model == 'sagog':
        model = SAGoG(
            num_variables=args.num_variables,
            seq_len=args.seq_len,
            num_classes=args.num_classes,
            hidden_dim=args.hidden_dim,
            graph_hidden_dim=args.graph_hidden_dim,
            num_graph_layers=args.num_graph_layers,
            num_windows=args.num_windows,
            graph_construction=args.graph_construction,
            gnn_type=args.gnn_type
        )
    else:  # sagog_lite
        model = SAGoGLite(
            num_variables=args.num_variables,
            seq_len=args.seq_len,
            num_classes=args.num_classes,
            hidden_dim=args.hidden_dim,
            graph_hidden_dim=args.graph_hidden_dim,
            num_windows=args.num_windows
        )

    model = model.to(args.device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='min')

    # Load checkpoint if specified
    start_epoch = 0
    if args.load_checkpoint:
        print(f"\nLoading checkpoint from {args.load_checkpoint}...")
        start_epoch, _ = load_checkpoint(model, optimizer, args.load_checkpoint, args.device)
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    print("\nStarting training...")
    print("=" * 80)

    best_test_acc = 0.0
    best_test_f1 = 0.0

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            args.device, args.use_augmentation
        )

        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, args.device)

        # Update learning rate
        scheduler.step(test_metrics['loss'])

        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1_macro']:.4f}")
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, "
              f"Acc: {test_metrics['accuracy']:.4f}, "
              f"F1: {test_metrics['f1_macro']:.4f}")

        # Save best model
        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']
            best_test_f1 = test_metrics['f1_macro']

            save_path = os.path.join(args.save_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, test_metrics, save_path)
            print(f"✓ Saved best model (Acc: {best_test_acc:.4f}, F1: {best_test_f1:.4f})")

        # Early stopping
        if early_stopping(test_metrics['loss']):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    # Final results
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best Test Accuracy: {best_test_acc:.4f}")
    print(f"Best Test F1 Score: {best_test_f1:.4f}")
    print("=" * 80)

    # Load best model and evaluate
    print("\nEvaluating best model on test set...")
    best_model_path = os.path.join(args.save_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        load_checkpoint(model, optimizer, best_model_path, args.device)
        final_metrics = evaluate(model, test_loader, criterion, args.device)

        print("\nFinal Test Metrics:")
        print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
        print(f"  F1 (Macro): {final_metrics['f1_macro']:.4f}")
        print(f"  F1 (Weighted): {final_metrics['f1_weighted']:.4f}")
        print(f"  Precision: {final_metrics['precision']:.4f}")
        print(f"  Recall:    {final_metrics['recall']:.4f}")


if __name__ == '__main__':
    main()