import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, hidden_size=50, num_layers=2, dropout=0.2):
        """
        LSTM that maps a window of prices to the next price
        :param hidden_size: number of units per LSTM layer
        :param num_layers: number of stacked LSTM layers
        :param dropout: dropout between LSTM layers
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        :param x: tensor of shape (batch, seq_len)
        :return: tensor of shape (batch, 1)
        """
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.linear(out[:, -1, :])


def train_model(model, x_train, y_train, epochs=50, batch_size=32, lr=1e-3, seed=0):
    """
    Trains the model with mean squared error
    :param x_train: numpy array of shape (n, seq_len), already scaled
    :param y_train: numpy array of shape (n, 1), already scaled
    :return: list of mean training loss per epoch
    """
    torch.manual_seed(seed)
    x = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y),
                                         batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []
    model.train()
    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        history.append(total / len(x))
    return history


def predict(model, x):
    """
    :param x: numpy array of shape (n, seq_len), already scaled
    :return: numpy array of shape (n, 1)
    """
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy()
