import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiscaleConvBlock(nn.Module):
    """Multiscale convolutional block for extracting temporal features at different scales"""

    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super(MultiscaleConvBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(out_channels * len(kernel_sizes))

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        out = torch.cat(outputs, dim=1)
        out = self.bn(out)
        return F.relu(out)


class TemporalDynamicModule(nn.Module):
    """Temporal dynamic learning module using LSTM/GRU"""

    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=True):
        super(TemporalDynamicModule, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.output_size = hidden_size * 2 if bidirectional else hidden_size

    def forward(self, x):
        # x: (batch, channels, length) -> (batch, length, channels)
        x = x.transpose(1, 2)
        output, (hidden, cell) = self.lstm(x)
        # Use last hidden state from both directions
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        return hidden


class AttentionModule(nn.Module):
    """Self-attention mechanism for temporal feature weighting"""

    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        # x: (batch, length, features)
        attention_weights = self.attention(x)
        attention_weights = F.softmax(attention_weights, dim=1)
        weighted = x * attention_weights
        return weighted.sum(dim=1)


class MSDL(nn.Module):
    """
    Multiscale Temporal Dynamic Learning for Time Series Classification

    Architecture:
    1. Multiscale feature extraction using parallel convolutions
    2. Temporal dynamic learning using recurrent layers
    3. Attention-based feature aggregation
    4. Classification head
    """

    def __init__(
        self,
        input_channels=1,
        num_classes=10,
        multiscale_channels=64,
        kernel_sizes=[3, 5, 7, 9],
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.5
    ):
        super(MSDL, self).__init__()

        # Multiscale feature extraction
        self.multiscale1 = MultiscaleConvBlock(
            input_channels,
            multiscale_channels,
            kernel_sizes
        )

        multiscale_out_channels = multiscale_channels * len(kernel_sizes)

        self.multiscale2 = MultiscaleConvBlock(
            multiscale_out_channels,
            multiscale_channels,
            kernel_sizes
        )

        # Temporal dynamic learning
        self.temporal_dynamic = TemporalDynamicModule(
            input_size=multiscale_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True
        )

        # Attention mechanism
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        feature_dim = self.temporal_dynamic.output_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(feature_dim // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, channels, length)

        # Multiscale feature extraction
        x = self.multiscale1(x)
        x = self.multiscale2(x)

        # Temporal dynamic learning
        temporal_features = self.temporal_dynamic(x)

        # Classification
        out = self.classifier(temporal_features)

        return out


class MSDLWithAttention(nn.Module):
    """
    Enhanced MSDL with explicit attention mechanism
    """

    def __init__(
        self,
        input_channels=1,
        num_classes=10,
        multiscale_channels=64,
        kernel_sizes=[3, 5, 7, 9],
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.5
    ):
        super(MSDLWithAttention, self).__init__()

        # Multiscale feature extraction
        self.multiscale1 = MultiscaleConvBlock(
            input_channels,
            multiscale_channels,
            kernel_sizes
        )

        multiscale_out_channels = multiscale_channels * len(kernel_sizes)

        self.multiscale2 = MultiscaleConvBlock(
            multiscale_out_channels,
            multiscale_channels,
            kernel_sizes
        )

        # Temporal dynamic learning with LSTM
        self.lstm = nn.LSTM(
            input_size=multiscale_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0
        )

        lstm_output_size = lstm_hidden * 2

        # Attention mechanism
        self.attention = AttentionModule(lstm_output_size)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(lstm_output_size // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, channels, length)

        # Multiscale feature extraction
        x = self.multiscale1(x)
        x = self.multiscale2(x)

        # Transpose for LSTM: (batch, channels, length) -> (batch, length, channels)
        x = x.transpose(1, 2)

        # Temporal dynamic learning
        lstm_out, _ = self.lstm(x)

        # Attention-based aggregation
        attended_features = self.attention(lstm_out)

        # Classification
        out = self.classifier(attended_features)

        return out


if __name__ == "__main__":
    # Example usage
    batch_size = 32
    input_channels = 1
    sequence_length = 128
    num_classes = 10

    # Create sample input
    x = torch.randn(batch_size, input_channels, sequence_length)

    # Standard MSDL model
    model = MSDL(
        input_channels=input_channels,
        num_classes=num_classes,
        multiscale_channels=64,
        kernel_sizes=[3, 5, 7],
        lstm_hidden=128,
        lstm_layers=2
    )

    output = model(x)
    print(f"MSDL output shape: {output.shape}")  # Should be (batch_size, num_classes)

    # MSDL with attention
    model_att = MSDLWithAttention(
        input_channels=input_channels,
        num_classes=num_classes,
        multiscale_channels=64,
        kernel_sizes=[3, 5, 7],
        lstm_hidden=128,
        lstm_layers=2
    )

    output_att = model_att(x)
    print(f"MSDL with Attention output shape: {output_att.shape}")

    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")