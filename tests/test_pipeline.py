import numpy as np
import pandas as pd
import pytest
from preprocessing import DataProcessing
from train import evaluate, run


@pytest.fixture
def price_csv(tmp_path):
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    dates = pd.date_range('2020-01-01', periods=300, freq='B')
    path = tmp_path / 'stock_data.csv'
    pd.DataFrame({'Date': dates, 'AAPL': prices}).to_csv(path, index=False)
    return path, prices


def test_windows_target_is_next_price(price_csv):
    path, prices = price_csv
    data = DataProcessing(path, 0.8)
    data.gen_train(10)
    assert data.x_train.shape == (240 - 10, 10)
    assert data.y_train.shape == (240 - 10, 1)
    np.testing.assert_allclose(data.x_train[0], prices[0:10])
    assert data.y_train[0, 0] == pytest.approx(prices[10])
    assert data.y_train[-1, 0] == pytest.approx(prices[239])


def test_scaler_fit_on_train_only(price_csv):
    path, prices = price_csv
    data = DataProcessing(path, 0.8)
    data.fit_scaler()
    assert data.min == pytest.approx(prices[:240].min())
    assert data.max == pytest.approx(prices[:240].max())
    np.testing.assert_allclose(data.unscale(data.scale(prices)), prices)


def test_evaluate():
    last = np.array([[1.0], [1.0]])
    m = evaluate(np.array([[2.0], [0.0]]), np.array([[3.0], [2.0]]), last)
    assert m['mae'] == pytest.approx(1.5)
    assert m['direction_acc'] == pytest.approx(0.5)


def test_run_end_to_end(price_csv, tmp_path):
    path, _ = price_csv
    results = run(path, seq_len=10, epochs=2, model_path=tmp_path / 'model.pt')
    assert set(results) == {'final_train_loss', 'lstm', 'naive'}
    assert np.isfinite(results['lstm']['rmse'])
    assert (tmp_path / 'model.pt').exists()
