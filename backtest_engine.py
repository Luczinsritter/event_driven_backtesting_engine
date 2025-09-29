from utils import *

class FinancialData () :

    def __init__ (self, ticker, end_date, days_nbr, interval) :
        self.ticker = ticker
        self.days_nbr = days_nbr
        self.end_date = end_date
        self.interval = interval
        self.data = self.get_data(self.ticker, self.end_date, self.interval)
        self.data = self.add_log_returns (self.data)

    def get_data (self, ticker, end_date, interval) :
        start_date = self.get_start_date (self.end_date, self.days_nbr)
        raw_df = yf.download(ticker, start = start_date, end = end_date, interval = interval, auto_adjust = True)
        if isinstance(raw_df.columns, pd.MultiIndex) :
            raw_df.columns = raw_df.columns.droplevel(1)
        clean_df = raw_df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return clean_df

    def get_start_date (self, end_date, days_nbr) :
        duration = dt.timedelta(days=days_nbr)
        return end_date - duration

    def add_log_returns (self, clean_df) :
        clean_df ['log_returns'] = np.log(clean_df ['Close'] / clean_df ['Close'].shift(1))
        clean_df.dropna(inplace=True)
        return clean_df

class EventBased (FinancialData) :
    '''
    EventBased class for event‑driven backtesting of trading strategies.
    - Execution at next tick/bar open to eliminate look‑ahead bias
    - Single‑position management (long or short) with entry/exit logic
    - Trade journaling: side, units, entry/exit date & price, PnL, duration
    '''
    
    def __init__ (self, ticker: str, end_date: dt.date, days_nbr: int, interval: str, amount: float, allow_negative_balance: bool) :
        super().__init__(ticker, end_date, days_nbr, interval) 
        self.initial_balance = amount
        self.current_balance = amount
        self.allow_negative_balance = allow_negative_balance
        self.position = {
            'side': None,  
            'units': 0,
            'entry_date': None,
            'entry_price': None
            }
        self.buy_trades = 0 
        self.sell_trades = 0
        self.close_trades = 0 
        self.trade_performance = {} 

    def get_date_price (self, ind_nbr) : 
        date = self.data.index[ind_nbr]
        price = self.data['Close'].iloc[ind_nbr]
        return date, price

    def get_execution_price(self, ind_nbr): 
        # Function used to avoid look ahead bias by taking position at the next period
        exec_date = self.data.index[ind_nbr + 1]
        exec_price = self.data['Open'].iloc[ind_nbr + 1]
        return exec_date, exec_price

    def print_balance (self, ind_nbr) : 
        date, price = self.get_date_price (ind_nbr)
        print(f'Date : {str(date)} | Account Current Balance : {self.current_balance}')

    def print_wealth (self, ind_nbr) : 
        date, price = self.get_date_price (ind_nbr)
        total_wealth = self.current_balance + self.position['units'] * price
        print(f'Date : {str(date)} | Total Wealth : {total_wealth:.2f}')

    def enter_long (self, ind_nbr, units=None, amount=None) :
        date, price = self.get_execution_price (ind_nbr)
        base_amount = amount if amount is not None else self.current_balance
        if not self.allow_negative_balance and base_amount < price:
            print(f'[{date}] Not enought capital to enter long')
            return
        if units is None:
            units = max(int(abs(base_amount) / price), 1)
        self.current_balance -= units * price
        self.position['side'] = 'long'
        self.position['units'] = units
        self.position['entry_date'] = date
        self.position['entry_price'] = price
        self.buy_trades += 1
        print(f'Date : {date} | ORDER = Long : {units} shares at {price}€')
        self.print_balance (ind_nbr)
        self.print_wealth (ind_nbr)

    def enter_short (self, ind_nbr, units=None, amount=None) : 
        date, price = self.get_execution_price (ind_nbr)
        base_amount = amount if amount is not None else self.current_balance
        if not self.allow_negative_balance and base_amount < price:
            print(f'[{date}] Not enought capital to enter short')
            return
        if units is None:
            units = max(int(abs(base_amount) / price), 1)
        self.current_balance += units * price
        self.position['side'] = 'short'
        self.position['units'] = units
        self.position['entry_date'] = date
        self.position['entry_price'] = price
        self.sell_trades += 1
        print(f'Date : {str(date)} | ORDER = Short : {units} shares at {price}€')
        self.print_balance (ind_nbr)
        self.print_wealth (ind_nbr)
    
    def close_position (self, ind_nbr) :
        date, price = self.get_execution_price (ind_nbr)
        self.closing_date = date
        self.closing_price = price
        units = self.position['units']
        entry_price = self.position['entry_price']
        if self.position['side'] == 'long':
            pnl = units * (self.closing_price - entry_price)
            self.current_balance += pnl
            trade_performance = np.log(self.closing_price / entry_price)
            direction = 'long'
        elif self.position['side'] == 'short':
            pnl = units * (entry_price - self.closing_price)
            self.current_balance += pnl
            trade_performance = np.log(entry_price / self.closing_price)
            direction = 'short'
        else:
            raise ValueError('No Open Positions')
        self.close_trades += 1
        duration = self.closing_date - self.position['entry_date']
        print(f'Date : {str(date)} | Close Position : {units} shares at {self.closing_price}€') 
        print(f'Perfomance : {trade_performance * 100 :.2f} % | PnL : {pnl:.2f}€ | Capital : {self.current_balance:.2f}€') 
        # Save every trade performance and details in a dictionnary to help the metrics computing
        self.trade_performance.update({date : {'Type of trade': direction, 
                                               'Number of shares': units, 
                                               'Duration': duration, 
                                               'Performance [%]': trade_performance *100, 
                                               'Log Return': trade_performance,
                                               'PnL': pnl,
                                               'Capital After Trade': self.current_balance}})
        self.position = {'side': None, 'units': 0, 'entry_date': None, 'entry_price': None} # Reset position
