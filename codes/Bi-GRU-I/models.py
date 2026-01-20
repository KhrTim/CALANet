import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(InceptionBlock, self).__init__()

        c_out = c_out//4

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=1), 
            nn.BatchNorm1d(c_out), 
            nn.ReLU()
        )

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=1),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
            nn.Conv1d(c_out, c_out, kernel_size=3, padding=1),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
        )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=1),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
            nn.Conv1d(c_out, c_out, kernel_size=5, padding=2),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
            nn.Conv1d(c_in, c_out, kernel_size=1),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out

class Bi_GRU_I(nn.Module):
    def __init__(self, nc_input, n_classes):
        super(Bi_GRU_I, self).__init__()

        self.gru = nn.GRU(input_size=nc_input, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        self.conv1 = InceptionBlock(128,128)
        self.conv2 = InceptionBlock(128,128)
        self.conv3 = InceptionBlock(128,128)
        
        self.fc = nn.Linear(128, n_classes)
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x, _ = self.gru(x)
        x = torch.transpose(x, 1, 2)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = F.adaptive_avg_pool1d(x,1)
        x = x.view(x.size(0), -1)
        
        logits = self.fc(x)

        return logits
