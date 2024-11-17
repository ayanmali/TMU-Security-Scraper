"""
TODO: Implement SARIMA/RNN/Gradient boosted trees for time series analysis
"""

from matplotlib.ticker import MaxNLocator
#import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from sqlalchemy import create_engine

from recommend_tfidf_algo import process_locations, process_type, replace_other_incident_type

import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import user, password, host, port, dbname

TABLE_NAME = "incidents"

"""
Loads and preprocesses the data for training.
"""
def load_and_transform_data(df):
    # Loading the data into a DataFrame
    # df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY id", engine)
    # Storing a duplicate of the DataFrame for reference when recommendations are suggested
    copied_df = df.copy(deep=True)
    copied_df = copied_df.drop(columns=['id', 'page', 'otherincidenttype', 'detailsembed', 'locdetailsembed', 'locdescrembed', 'locationembed', 'descrembed'], axis=1)

    # For specifying the exact incident type for any incident type that is "Other"
    copied_df = replace_other_incident_type(copied_df)

    # For one hot encoding the locations
    copied_df = process_locations(copied_df)

    # For incident type
    copied_df = process_type(copied_df)

    # For the date/time of the incident
    copied_df, monthly_incidents = process_dates(copied_df)

    # text_features = {}
    # vectorizers = {}

    # # For incident details and suspect descriptions
    # for col in ('incidentdetails', 'description'):
    #     tfidf_df, vectorizers[col], _ = extract_text_features(copied_df, col=col)
    #     text_features[col] = scale_text_features(tfidf_df)

    # Dropping features that we don't need anymore
    copied_df = copied_df.drop(['incidentdetails', 'description'], axis=1)

    # scaler = StandardScaler()
    # # add target var
    # # Only scaling the day of week, day of month, hour, month, and target columns
    # numerical_columns = ['target'] + [col for col in copied_df.columns 
    #                                     if col.endswith(('_sin', '_cos'))]
    # copied_df[numerical_columns] = scaler.fit_transform(copied_df[numerical_columns])

    # Concatenate all features
    # result_df = pd.concat([copied_df] + list(text_features.values()), axis=1)

    return copied_df, monthly_incidents

def process_dates(df):
    df['dateofincident'] = pd.to_datetime(df['dateofincident'])
    monthly_incidents = df.resample('M', on='dateofincident').size()

    # Extracting the day of the week, month, and hour from the datetime column and cyclical encoding them
    df['day_of_week_sin'], df['day_of_week_cos'] = cyclical_encode(df['dateofincident'].dt.dayofweek, 7)
    df['month_sin'], df['month_cos'] = cyclical_encode(df['dateofincident'].dt.month, 12)
    df['hour_sin'], df['hour_cos'] = cyclical_encode(df['dateofincident'].dt.hour, 24)
    df['day_of_month_sin'], df['day_of_month_cos'] = cyclical_encode(df['dateofincident'].dt.day, 31)

    df = df.drop(columns=['dateposted', 'datereported'])

    return df, monthly_incidents

def visualize_time_series(monthly_incidents):
    # Create the figure
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot the data
    # monthly_incidents = df.resample('M', on='dateofincident').size()

    # Calculate moving average (6-month window)
    moving_avg = monthly_incidents.rolling(window=6).mean()

    # Plot both the actual data and moving average
    ax.plot(monthly_incidents.index, monthly_incidents.values, 
            label='Monthly Incidents', color='#2E86C1', alpha=0.6)
    ax.plot(moving_avg.index, moving_avg.values, 
            label='6-Month Moving Average', color='#E74C3C', linewidth=2)

    # Fill the area under the curve
    ax.fill_between(monthly_incidents.index, monthly_incidents.values, 
                    alpha=0.2, color='#2E86C1')

    # Customize the plot
    ax.set_title('TMU Campus Security Incidents Over Time', fontsize=14, pad=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Incidents', fontsize=12)

    # Format y-axis to show whole numbers only
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    ax.legend()

    # Add padding
    plt.tight_layout()

    # Show the plot
    plt.show()

def cyclical_encode(data, max_val):
    """
    Create sine and cosine encoding for cyclical features
    """
    data = 2 * np.pi * data / max_val
    return np.sin(data), np.cos(data)

def main():
    # Loading the data
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)

    # Cleaning/preprocessing the data
    df, monthly_incidents = load_and_transform_data(df)

    print(df.head())

    visualize_time_series(monthly_incidents)

# class Model(nn.Module):
#     def __init__():
#         pass 

#     def forward():
#         pass

if __name__ == "__main__":
    main()