from utils import *
from backtest_engine import *

class EmaCrossStrategy(EventBased):

    def __init__(self, ticker, end_date, days_nbr, interval, amount,
                 short_window=20, long_window=50, allow_negative_balance: bool = True):
        super().__init__(ticker, end_date, days_nbr, interval, amount, allow_negative_balance)
        self.short_window = short_window
        self.long_window = long_window
        self.data['ema_short'] = self.data['Close'].ewm(span=short_window, adjust=False).mean()
        self.data['ema_long'] = self.data['Close'].ewm(span=long_window, adjust=False).mean()

    def run_backtest(self):
        '''
        - long: ema_short > ema_long
        - short: ema_short < ema_long
        '''
        for i in range(len(self.data) - 1):  # -1 because get_execution_price(i+1)
            if self.data['ema_short'].iloc[i] > self.data['ema_long'].iloc[i]:
                # Entrer long
                if self.position['side'] == 'short':
                    self.close_position(i)
                if self.position['side'] is None:
                    self.enter_long(i, amount=self.current_balance)

            elif self.data['ema_short'].iloc[i] < self.data['ema_long'].iloc[i]:
                # Entrer short
                if self.position['side'] == 'long':
                    self.close_position(i)
                if self.position['side'] is None:
                    self.enter_short(i, amount=self.current_balance)

        # Close last position
        if self.position['side'] is not None:
            self.close_position(len(self.data) - 2)

class ArimaTickStrategy(EventBased):
    
    def __init__(
        self,
        ticker,
        end_date,
        days_nbr,
        interval,
        amount,
        allow_negative_balance: bool = True,
        arima_order=(1, 0, 0),
        window=200,
        refit_every=10,
        long_threshold=0.0005,
        short_threshold=0.0005,  
        alpha=0.2,
        max_position_fraction=1.0  # fraction of capital to deploy (1.0 = all-in)
    ):
        super().__init__(ticker, end_date, days_nbr, interval, amount, allow_negative_balance)
        self.p, self.d, self.q = arima_order
        self.window = int(window)
        self.refit_every = int(refit_every)
        self.long_threshold = float(long_threshold)
        self.short_threshold = float(short_threshold)
        self.alpha = float(alpha)
        self.max_position_fraction = float(max_position_fraction)

        self.data['log_ret'] = np.log(self.data['Close']).diff()
        self.data['log_ret'] = (
            self.data['log_ret']
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        self._model = None
        self._last_fit_index = None

    def _fit_model(self, end_idx):
        start_idx = max(0, end_idx - self.window + 1)
        y = self.data['log_ret'].iloc[start_idx:end_idx + 1].astype(float)

        # Guard against degenerate windows
        if len(y) < max(5, self.p + self.q + 1):
            return None, None, None

        try:
            model = ARIMA(y, order=(self.p, self.d, self.q))
            res = model.fit(method_kwargs={'warn_convergence': False})
        except Exception as e:
            print(f'[WARN] ARIMA fit error at idx {end_idx}: {e}')
            return None, None, None

        fcast = res.get_forecast(steps=1)
        mean = float(fcast.predicted_mean.iloc[0])
        ci = fcast.conf_int(alpha=self.alpha)
        lower = float(ci.iloc[0, 0])
        upper = float(ci.iloc[0, 1])

        self._model = res
        self._last_fit_index = end_idx
        return mean, lower, upper

    def _maybe_refit(self, idx):
        should_refit = (self._model is None) or (self._last_fit_index is None) or ((idx - self._last_fit_index) >= self.refit_every)
        if should_refit:
            return self._fit_model(idx)
        else:
            # Still recompute forecast with the same approach to keep API consistent
            return self._fit_model(idx)

    def _position_units(self, price):
        deploy_amount = self.current_balance * self.max_position_fraction
        units = max(int(abs(deploy_amount) / price), 1)
        return units

    def run_backtest(self):
        n = len(self.data)
        # Ensure we have at least one future tick for execution
        if n < 3:
            print('[INFO] Not enough data for tick-by-tick backtest.')
            return

        start_i = max(self.window, 2)
        for i in range(start_i, n - 1):  # -1 to allow execution at i+1
            forecast, lower, upper = self._maybe_refit(i)
            if forecast is None:
                continue
            dt_i, px_i = self.get_date_price(i)

            go_long = forecast >= self.long_threshold
            go_short = forecast <= -self.short_threshold
            neutral = (not go_long) and (not go_short)

            if neutral:
                # Optional flattening: close if position exists and forecast lacks conviction
                if self.position['side'] is not None:
                    self.close_position(i)
                continue

            if go_long:
                if self.position['side'] == 'short':
                    self.close_position(i)
                if self.position['side'] is None:
                    exec_date, exec_price = self.get_execution_price(i)
                    units = self._position_units(exec_price)
                    self.enter_long(i, units=units)

            elif go_short:
                if self.position['side'] == 'long':
                    self.close_position(i)
                if self.position['side'] is None:
                    exec_date, exec_price = self.get_execution_price(i)
                    units = self._position_units(exec_price)
                    self.enter_short(i, units=units)

        # Close any remaining open position at the last executable tick
        if self.position['side'] is not None:
            self.close_position(n - 2)