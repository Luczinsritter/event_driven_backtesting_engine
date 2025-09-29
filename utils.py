'''
Utility functions and imports for the framework.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import quantstats as qs
import datetime as dt

def convert_to_simple_returns (df: pd.DataFrame) -> pd.DataFrame :
    df['simple_returns'] = np.exp(df['log_returns']) -1
    return df
    
def convert_to_log_returns (df: pd.DataFrame) -> pd.DataFrame :
    df['log_returns'] = np.log(df['simple_returns'] +1)
    return df


from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', message='.*No supported index is available.*')
warnings.filterwarnings('ignore', message='.*no associated frequency information.*')
warnings.filterwarnings('ignore', message='Non-stationary starting autoregressive parameters')
warnings.filterwarnings('ignore', message='No frequency information was provided')

__all__ = [
    'np','plt','pd', 'yf','dt','qs',
    'warnings',
    'ConvergenceWarning',
    'convert_to_simple_returns',
    'convert_to_log_returns',
    'ARIMA',
    'train_test_split',
]