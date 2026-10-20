import argparse
import numpy as np
import torch
from preprocessing import DataProcessing
from model import LSTMModel, train_model, predict


def evaluate(y_true, y_pred, last_prices):
    """
    Compares predictions with the actual next prices
    :param y_true: actual next prices, shape (n, 1)
    :param y_pred: predicted next prices, shape (n, 1)
    :param last_prices: last price of each input window, shape (n, 1)
    :return: dict of RMSE, MAE and directional accuracy
    """
    err = y_pred - y_true
    direction = np.sign(y_pred - last_prices) == np.sign(y_true - last_prices)
    return {
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mae': float(np.mean(np.abs(err))),
        'direction_acc': float(np.mean(direction)),
    }


def run(file, train=0.8, seq_len=50, epochs=50, batch_size=32, lr=1e-3, model_path='model.pt'):
    """
    Trains an LSTM on a price csv and reports test metrics against a naive baseline
    that predicts tomorrow's price is today's price
    """
    data = DataProcessing(file, train)
    data.fit_scaler()
    data.gen_train(seq_len)
    data.gen_test(seq_len)

    model = LSTMModel()
    history = train_model(model, data.scale(data.x_train), data.scale(data.y_train),
                          epochs=epochs, batch_size=batch_size, lr=lr)
    y_pred = data.unscale(predict(model, data.scale(data.x_test)))

    last = data.x_test[:, -1:]
    results = {
        'final_train_loss': history[-1],
        'lstm': evaluate(data.y_test, y_pred, last),
        'naive': evaluate(data.y_test, last, last),
    }
    # the naive baseline never predicts a move, so its directional accuracy is meaningless
    results['naive']['direction_acc'] = None
    torch.save({'state_dict': model.state_dict(), 'min': data.min, 'max': data.max,
                'seq_len': seq_len}, model_path)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an LSTM next-day price model')
    parser.add_argument('--file', default='stock_data.csv')
    parser.add_argument('--train', type=float, default=0.8)
    parser.add_argument('--seq-len', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model-path', default='model.pt')
    args = parser.parse_args()
    results = run(args.file, args.train, args.seq_len, args.epochs, args.batch_size, args.lr,
                  args.model_path)
    print(f"final train loss: {results['final_train_loss']:.6f}")
    for name in ('lstm', 'naive'):
        m = results[name]
        direction = 'n/a' if m['direction_acc'] is None else f"{m['direction_acc']:.3f}"
        print(f"{name:>5}: rmse={m['rmse']:.4f} mae={m['mae']:.4f} direction_acc={direction}")
