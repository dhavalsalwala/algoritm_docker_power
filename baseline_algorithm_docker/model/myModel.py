
import torch
from torch import nn


class ProbMCdropoutDNN(nn.Module):
    """
    Monte Carlo (MC) dropout neural network with 2 hidden layers.
    """

    def __init__(self, input_size, hidden_size_1=50, hidden_size_2=20, dropout=0.005):
        super(ProbMCdropoutDNN, self).__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size_1)
        self.linear2 = nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2)
        self.linear3 = nn.Linear(in_features=hidden_size_2, out_features=2)
        self.dropout = nn.Dropout(dropout)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # first linear layer
        x = self.linear1(x)
        x = self.softplus(x)
        x = self.dropout(x)

        # second linear layer
        x = self.linear2(x)
        x = self.softplus(x)
        x = self.dropout(x)

        x = self.linear3(x)
        return torch.distributions.Normal(
            loc=x[:, 0:1].squeeze(),
            scale=self.softplus(x[:, 1:2].squeeze()).add(other=1e-6)
        )

    def predict(self, x):
        distrs = self.forward(x)
        y_pred = distrs.sample()
        return y_pred
