import pandas as pd

def load_data(stock_symbol, start_date, end_date):
    return pd.read_csv(f'{stock_symbol}_{start_date}_{end_date}.csv')

def preprocess_data(data):
    # Perform normalization or feature engineering here
    return data
