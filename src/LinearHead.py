import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()

        self.L1 = nn.Linear(dim, hidden_dim)
        self.L2 = nn.Linear(hidden_dim, dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.L1(x)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.L2(out)
        out = self.dropout2(out)
        out = out + x
        out = self.layer_norm(out)
        return out