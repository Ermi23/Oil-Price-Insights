# eda_utils.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
class EDA:
    def __init__(self, data):
        """
        Initialize the EDA class with data.

        Parameters:
            data (pd.DataFrame): The dataframe containing the data.
        """
        self.data = data

    def calculate_moving_average(self, column, window):
        """
        Calculate the moving average of a specified column in the dataset.

        Parameters:
            column (str): The column for which to calculate the moving average.
            window (int): The window size for the moving average.

        Returns:
            pd.Series: The moving average series.
        """
        moving_avg = self.data[column].rolling(window=window).mean()
        return moving_avg

    def plot_moving_average(self, column, window):
        """
        Plot the original data and its moving average.

        Parameters:
            column (str): The column to plot.
            window (int): The window size for the moving average.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(self.data[column], color='blue', label='Daily Price')
        plt.plot(self.calculate_moving_average(column, window), color='orange', label=f'{window}-Day Moving Average')
        plt.title(f"{column} with {window}-Day Moving Average")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.show()

    def plot_seasonal_decomposition(self, column, period=365):
        """
        Perform seasonal decomposition on the specified column.
        
        Parameters:
            column (str): Column to decompose.
            period (int): Periodicity of the data (e.g., 365 for yearly).
        """
        decomposition = seasonal_decompose(self.data[column], model='multiplicative', period=period)
        decomposition.plot()
        plt.suptitle(f'Seasonal Decomposition of {column}', fontsize=16)
        plt.show()
        
    def plot_volatility(self, column, window=30):
        """
        Plot rolling standard deviation to visualize volatility.
        
        Parameters:
            column (str): Column for which to calculate volatility.
            window (int): Rolling window size for standard deviation.
        """
        rolling_std = self.data[column].rolling(window=window).std()
        
        plt.figure(figsize=(14, 7))
        plt.plot(self.data[column], color='blue', alpha=0.5, label='Daily Price')
        plt.plot(rolling_std, color='red', label=f'{window}-Day Rolling Volatility')
        plt.title(f"{column} Volatility with {window}-Day Rolling Window")
        plt.xlabel("Date")
        plt.ylabel("Price Volatility")
        plt.legend()
        plt.show()

    def merge_economic_data(self, economic_data, on='Date'):
        """
        Merge economic indicators or events with the main dataset on a common date column.

        Parameters:
            economic_data (pd.DataFrame): Dataframe containing economic indicators or events.
            on (str): Column to merge on, typically 'Date'.

        Returns:
            pd.DataFrame: Dataset with events or indicators as separate columns.
        """
        # Perform a left join to keep all dates from the main data
        self.data = self.data.merge(economic_data, on=on, how='left', indicator=False)


    def plot_correlation_matrix(self):
        """
        Plot a correlation matrix to visualize relationships between oil prices and economic indicators.
        """
        plt.figure(figsize=(10, 8))
        correlation = self.data.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix of Brent Oil Prices and Economic Indicators")
        plt.show()

    def calculate_event_impact(self, events_data, window=30):
        """
        Calculate the percentage change in oil prices around each event.

        Parameters:
            events_data (dict): Dictionary of events with dates and descriptions.
            window (int): Number of days before and after each event to analyze.

        Returns:
            pd.DataFrame: DataFrame showing percentage change before and after each event.
        """
        impacts = []
        
        for event_date, event_name in events_data.items():
            date = pd.to_datetime(event_date)
            
            # Check if the date is within the dataset range
            if date not in self.data.index:
                print(f"Data unavailable around {event_name} due to missing date")
                continue
            
            try:
                # Calculate prices before and after the event
                price_before = self.data.loc[date - pd.Timedelta(days=window), 'Price']
                price_after = self.data.loc[date + pd.Timedelta(days=window), 'Price']
                
                # Calculate percentage change only if prices are available
                if pd.notna(price_before) and pd.notna(price_after):
                    pct_change = ((price_after - price_before) / price_before) * 100
                    impacts.append({
                        'Event': event_name,
                        'Date': date,
                        'Price Before': price_before,
                        'Price After': price_after,
                        'Percentage Change (%)': pct_change
                    })
                else:
                    print(f"Price data unavailable around {event_name} due to missing price")
                    
            except KeyError:
                print(f"Data unavailable around {event_name} due to KeyError")
                continue

        return pd.DataFrame(impacts)
