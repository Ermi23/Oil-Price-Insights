# eda_utils.py

import pandas as pd
import matplotlib.pyplot as plt

def calculate_moving_average(data, column, window):
    """
    Calculate and plot the moving average of a specified column in the dataset.
    
    Parameters:
        data (pd.DataFrame): The dataframe containing the data.
        column (str): The column for which to calculate the moving average.
        window (int): The window size for the moving average.
        
    Returns:
        pd.Series: The moving average series.
    """
    moving_avg = data[column].rolling(window=window).mean()
    return moving_avg

def plot_moving_average(data, column, window):
    """
    Plot the original data and its moving average.
    
    Parameters:
        data (pd.DataFrame): The dataframe containing the data.
        column (str): The column to plot.
        window (int): The window size for the moving average.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data[column], color='blue', label='Daily Price')
    plt.plot(calculate_moving_average(data, column, window), color='orange', label=f'{window}-Day Moving Average')
    plt.title(f"{column} with {window}-Day Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()
