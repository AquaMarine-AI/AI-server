import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def replace_outliers_zscore(df, column, threshold=2):
    """Replace outliers based on the Z-Score method with the maximum of the last 3 values."""
    mean = df[column].mean()
    std = df[column].std()
    df['z_score'] = (df[column] - mean) / std

    # Iterate through the dataframe and replace outliers with the maximum of the last 3 values
    for i in range(len(df)):
        if np.abs(df.loc[i, 'z_score']) > threshold:
            if i >= 3:
                df.loc[i, column] = df.loc[i-3:i-1, column].max()
            else:
                df.loc[i, column] = df.loc[0:i, column].max()
    
    # Drop the z_score column as it's no longer needed
    df = df.drop(columns=['z_score'])
    return df

def main(file_path):
    """Preprocess the data and return the processed DataFrame."""
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Convert 'Time' column to datetime for easier manipulation
    data['time'] = pd.to_datetime(data['time'])
    
    # Create a rolling window of size 10 and calculate the average
    data['rolling_mean'] = data['feed_pressure'].rolling(window=10).mean()

    # Replace values below the rolling mean with the maximum of the last 5 values
    for i in range(len(data)):
        if data.loc[i, 'feed_pressure'] < data.loc[i, 'rolling_mean']:
            data.loc[i, 'feed_pressure'] = data.loc[max(0, i-5):i, 'feed_pressure'].max()

    # Replace Z-Score-based outliers with the maximum of the last 3 values
    data_cleaned = replace_outliers_zscore(data, 'feed_pressure', threshold=2)
    
    # Apply smoothing using a rolling mean with a window size of 5 for further smoothing
    data_cleaned['processed_feed_pressure'] = data_cleaned['feed_pressure'].rolling(window=5, min_periods=1).mean()

    # Round the 'processed_feed_pressure' column to 6 decimal places
    data_cleaned['processed_feed_pressure'] = data_cleaned['processed_feed_pressure'].round(6)

    # Drop the unnecessary columns (feed_pressure, rolling_mean)
    data_cleaned = data_cleaned.drop(columns=['feed_pressure', 'rolling_mean'])

    # Return the processed DataFrame
    return data_cleaned
