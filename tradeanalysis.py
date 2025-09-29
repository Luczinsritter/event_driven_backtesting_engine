from utils import *

class TradeAnalysis():
    '''
    TradeAnalysis: post-trade performance analytics.
    Computes adjusted returns (slippage, costs), risk metrics (Sharpe, Sortino, Kelly),
    drawdowns, and generates performance reports and visualisations.
    '''


    def __init__(self, trade_performance: dict, capital: int, cost_per_trade: float, slippage: float, price_series: pd.Series):
        if isinstance(trade_performance, pd.DataFrame):
            self.trade_performance = trade_performance.copy()
        else:
            self.trade_performance = pd.DataFrame.from_dict(trade_performance, orient='index')
        self.capital = capital
        self.cost_per_trade = cost_per_trade
        self.slippage = slippage
        self.price_series = price_series

    def _prepare_df(self) -> pd.DataFrame:
        df = self.trade_performance.copy()
        if 'Log Return' in df.columns:
            df = df.rename(columns={'Log Return': 'log_returns'})
            df = convert_to_simple_returns(df)
        if self.slippage != 0:
            df['simple_returns'] -= self.slippage
        if self.cost_per_trade != 0:
            df['simple_returns'] -= self.cost_per_trade / self.capital
        if 'simple_returns' in df.columns:
            df['Net Log Return'] = np.log(1 + df['simple_returns'])
        return df

    def df_analysis(self) -> pd.Series:
        df = self._prepare_df()
        nbr_long = (df['Type of trade'] == 'long').sum()
        nbr_short = (df['Type of trade'] == 'short').sum()
        total_trade = len(df)
        avg_period_trade = df['Duration'].mean()
        avg_profit = df['Performance [%]'].mean()
        avg_net_profit = df['PnL'].mean() if 'PnL' in df.columns else 0
        pnl = df['Capital After Trade'].iloc[-1] - self.capital if total_trade > 0 and 'Capital After Trade' in df.columns else 0.0
        summary = {
            'nbr_long': nbr_long,
            'nbr_short': nbr_short,
            'total_trade': total_trade,
            'avg_period_trade': avg_period_trade,
            'avg_profit': avg_profit,
            'avg_net_profit': avg_net_profit,
            'pnl': pnl
        }
        return pd.Series(summary)

    def max_drawdown(self) -> tuple:
        # Max drawdown and recovery time
        df = self.trade_performance.copy()
        if 'Capital After Trade' not in df.columns:
            return 0.0, pd.NaT

        df['equity'] = df['Capital After Trade']
        df['cummax'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] / df['cummax']) - 1
        df['drawdown'] = df['drawdown'].clip(lower=-1.0)

        max_dd = df['drawdown'].min()

        recovery = pd.NaT
        if max_dd < 0:
            trough = df['drawdown'].idxmin()
            peak_idx = df.loc[:trough, 'equity'].idxmax()
            after = df.loc[trough:]
            if not after.empty:
                recov_idx = after[after['equity'] >= df.loc[peak_idx, 'equity']].index.min()
                if pd.notna(recov_idx):
                    recovery = recov_idx - peak_idx

        return max_dd, recovery

    def sharpe_ratio(self, risk_free: float) -> float:
        df = self.trade_performance.copy()
        if 'PnL' not in df.columns:
            return 0.0
        df['return_on_initial'] = df['PnL'] / self.capital
        excess_returns = df['return_on_initial'] - (risk_free / 252)
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std(ddof=1) if excess_returns.std(ddof=1) != 0 else 0.0

    def sortino_ratio(self, risk_free: float) -> float:
        df = self.trade_performance.copy()
        if 'PnL' not in df.columns:
            return 0.0
        df['return_on_initial'] = df['PnL'] / self.capital
        excess_returns = df['return_on_initial'] - (risk_free / 252)
        downside = excess_returns[excess_returns < 0]
        denom = downside.std(ddof=1)
        return np.sqrt(252) * excess_returns.mean() / denom if denom != 0 else 0.0

    def gain_loss_stats(self) -> dict:
        df = self.trade_performance.copy()
        if 'PnL' not in df.columns:
            return {'win_rate': 0, 'gain_loss_ratio': 0}
        wins = df[df['PnL'] > 0]['PnL']
        losses = df[df['PnL'] <= 0]['PnL']
        win_rate = len(wins) / len(df) if len(df) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
        gain_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        return {'win_rate': win_rate, 'gain_loss_ratio': gain_loss_ratio}

    def kelly_criterion(self) -> float:
        df = self.trade_performance.copy()
        if 'PnL' not in df.columns:
            return 0.0
        wins = df[df['PnL'] > 0]['PnL']
        losses = df[df['PnL'] <= 0]['PnL']
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        p = len(wins) / len(df)
        q = 1 - p
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        R = avg_win / avg_loss
        kelly_fraction = p - (q / R)
        return max(kelly_fraction, 0.0)

    def report(self, risk_free: float = 0.02):
        summary = self.df_analysis()
        sharpe = self.sharpe_ratio(risk_free)
        sortino = self.sortino_ratio(risk_free)
        max_dd, recovery = self.max_drawdown()
        gl_stats = self.gain_loss_stats()
        kelly = self.kelly_criterion()
        print('=== Strategy Report ===')
        print(summary)
        print(f"\nSharpe Ratio (corrected): {sharpe:.2f}")
        print(f"Sortino Ratio: {sortino:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Recovery Time: {recovery}")
        print(f"Win Rate: {gl_stats['win_rate']:.2%}")
        print(f"Gain/Loss Ratio: {gl_stats['gain_loss_ratio']:.2f}")
        print(f"Kelly Fraction: {kelly:.2f}")

    def plot_performance(self):
        if self.trade_performance.empty:
            raise ValueError('trade_performance is empty.')
        df = self.trade_performance.copy()
        required_cols = {'PnL', 'Capital After Trade'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing columns in trade_performance : {required_cols - set(df.columns)}")
        df = df.sort_index()
        wealth_series = df['Capital After Trade']
        pnl_series = df['PnL'].cumsum()
        bh_series = self.capital * (self.price_series / self.price_series.iloc[0])

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Capital / Wealth')
        wealth_series.plot(ax=ax1, color='blue', label='Global Wealth')
        bh_series.plot(ax=ax1, color='orange', linestyle='--', label='Buy & Hold')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Cumulative PnL')
        pnl_series.plot(ax=ax2, color='green', label='Cumulative PnL')

        fig.tight_layout()
        plt.title('Performance: Buy & Hold vs Strategy')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.grid(True)
        plt.show()