from utils import *
from backtest_engine import *
from tradeanalysis import *
from ema_arima import *

# === Parameters ===
ticker = 'AAPL'
end_date = dt.date.today()
days_nbr = 365 * 2
interval = '1d'
initial_capital = 10000

# === Run and backtest ===
strategy = EmaCrossStrategy(
    ticker=ticker,
    end_date=end_date,
    days_nbr=days_nbr,
    interval=interval,
    amount=initial_capital,
    short_window=20,
    long_window=50,
    allow_negative_balance=True
)

strategy_arima = ArimaTickStrategy(
    ticker=ticker,
    end_date=end_date,
    days_nbr=days_nbr,
    interval=interval,
    amount=initial_capital,
    allow_negative_balance=True,
    arima_order=(1, 1, 0), 
    window=100,
    refit_every=5,
    long_threshold=0.0007,
    short_threshold=0.0007,
    max_position_fraction=0.5)

strategy.run_backtest()
strategy_arima.run_backtest() 

# === Analysis and Visualisation ===
ta = TradeAnalysis(
    trade_performance=strategy.trade_performance,
    price_series=strategy.data['Close'],
    capital=initial_capital,
    cost_per_trade=0,
    slippage=0
)

ta_arima = TradeAnalysis(
    trade_performance=strategy_arima.trade_performance,
    capital=initial_capital,
    cost_per_trade=0,
    slippage=0,
    price_series=strategy.data['Close'])


print('\n=== EMA Strategy Analysis and Visualisation ===')
ta.report()
ta.plot_performance()

print('\n=== ARIMA Strategy Analysis and Visualisation ===')
ta_arima.report()
ta_arima.plot_performance()
