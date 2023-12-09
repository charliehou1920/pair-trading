import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
import os


# Function to fetch S&P 500 tickers and sectors
def get_sp500_tickers_and_sectors():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url, header=0)[0]
    return table[['Symbol', 'GICS Sector']].rename(columns={'Symbol': 'Ticker'})

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    ticker = ticker.replace('.', '-')
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        hist.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
        hist['Return'] = hist['Close'] / hist['Close'].shift(1)
        hist.dropna(subset=['Return'], inplace=True)
        return hist
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
    

# Function to generate SP500 stock tables
def generate_sp500_stock_tables(start_date='2020-01-01', end_date='2023-12-05'):
    sp500_tickers_sectors = get_sp500_tickers_and_sectors()
    sp500_data = {}

    for ticker, sector in sp500_tickers_sectors.itertuples(index=False):
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        stock_data['Sector'] = sector
        sp500_data[ticker] = stock_data

    return sp500_data

# Function to standardize the returns in each stock
def standardize_returns(stock_tables):
    scaler = StandardScaler()

    for ticker, df in stock_tables.items():
        if not df.empty and 'Return' in df.columns:
            # Reshape the data for standardization (needs to be 2D)
            returns = df['Return'].values.reshape(-1, 1)

            # Standardize the 'Return' feature
            standardized_returns = scaler.fit_transform(returns)

            # Replace the original 'Return' column with the standardized values
            df['Return'] = standardized_returns

    return stock_tables

# Get tables by sector
def get_tables_by_sector(stock_tables, target_sector):
    filtered_tables = {}
    for ticker, df in stock_tables.items():
        if not df.empty and 'Sector' in df.columns and df['Sector'].iloc[0] == target_sector:
            filtered_tables[ticker] = df
    return filtered_tables

# Calculate PCA for Stocks
def perform_pca(stock_tables, n_components):
    # 1. Create a dataframe which contains the 'Return' of each stock
    returns_data = pd.DataFrame()

    for ticker, df in stock_tables.items():
        if not df.empty and 'Return' in df.columns:
            returns_data[ticker] = df['Return']

    # 2. Compute Covariance Matrix
    cov_matrix = returns_data.cov()

    # 3. Apply PCA with n_components
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(cov_matrix)

    # 4. Create a dataframe with the ticker name and n_components
    principal_components_df = pd.DataFrame(principal_components, index=cov_matrix.index)

    return principal_components_df


def cluster_stocks_with_optics(principal_components_df, min_samples):
    # Apply OPTICS clustering algorithm
    optics = OPTICS(min_samples=min_samples)
    labels = optics.fit_predict(principal_components_df)

    # labeling
    principal_components_df['Cluster'] = labels

    # Plot
    if principal_components_df.shape[1] in [3, 4]:
        plt.figure(figsize=(10, 7))

        if principal_components_df.shape[1] == 4:
            ax = plt.axes(projection='3d')
            ax.scatter(principal_components_df.iloc[:, 0], principal_components_df.iloc[:, 1], principal_components_df.iloc[:, 2], c=labels, cmap='viridis', marker='o')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
        else:
            plt.scatter(principal_components_df.iloc[:, 0], principal_components_df.iloc[:, 1], c=labels, cmap='viridis', marker='o')
            plt.xlabel('PC1')
            plt.ylabel('PC2')

        plt.title('OPTICS Clustering of Stocks')
        plt.show()

    return principal_components_df





