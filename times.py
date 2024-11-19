"""
TODO: Implement SARIMA/RNN/Gradient boosted trees for time series analysis
"""

from matplotlib.ticker import MaxNLocator
#import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sqlalchemy import create_engine
from datetime import datetime

from sklearn.metrics import r2_score

from recommend_tfidf_algo import process_locations, process_type, replace_other_incident_type

import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import user, password, host, port, dbname

TABLE_NAME = "incidents"

SEASON_DICT = {1: 'Winter',
               2: 'Winter',
               3: 'Spring', 
               4: 'Spring',
               5: 'Spring',
               6: 'Summer',
               7: 'Summer',
               8: 'Summer',
               9: 'Fall',
               10: 'Fall',
               11: 'Fall',
               12: 'Winter'}

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# X tensor needs to be of shape (batch_size, seq_len, num_features)
# Y tensor needs to be of shape (batch_size, timesteps)   --- ex. for one step forecasting, timesteps = 1. For multi-step forecasting, timesteps = seq_len
BATCH_SIZE = 6
SEQUENCE_LENGTH = 5

# Defining the class to be used for the DataLoader
class IncidentDataset(torch.utils.data.Dataset):
    def __init__(self, df, sequence_length, prediction_length=1):
        self.features = df.drop('target', axis=1).values
        self.target = df['target'].values

        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

        # Print debug information
        print("Feature names:", df.columns.tolist())
        print("Number of features:", self.features.shape[1])
        
        # Sample a few rows to verify feature values
        print("Sample feature values:")
        print(df.head())

        # X = self.features[0:0 + self.sequence_length]
        # y = self.target[0 + self.sequence_length : 0 + self.sequence_length + self.prediction_length]
        # print("X Tensor Example Shape:", torch.FloatTensor(X).shape)
        # print("Y Tensor Example Shape:", torch.FloatTensor(y).shape)

    def __len__(self):
        return len(self.features) - self.sequence_length - self.prediction_length + 1
    
    def __getitem__(self, idx):
        X = self.features[idx:idx + self.sequence_length]
        # y = self.target[idx + self.sequence_length : idx + self.sequence_length + self.prediction_length]
        y = self.target[idx + 1 : idx + self.sequence_length + 1]
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def get_monthly_incidents(self):
        return self.monthly_incidents

"""
Loads and preprocesses the data for training.
"""
def load_and_transform_data(df):
    # Loading the data into a DataFrame
    # df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY id", engine)
    # Storing a duplicate of the DataFrame for reference when recommendations are suggested
    copied_df = df.copy(deep=True)
    copied_df = copied_df.drop(columns=['page', 'otherincidenttype', 'detailsembed', 'locdetailsembed', 'locdescrembed', 'locationembed', 'descrembed'], axis=1)

    # Sorting the DataFrame chronologically
    copied_df['dateofincident'] = pd.to_datetime(copied_df['dateofincident'])
    copied_df = copied_df.sort_values('dateofincident')

    # Remove rows where target is NaN (at the end of the dataset)
    # copied_df = copied_df.dropna(subset=['target'])

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

    # Setting the target variable (# of incidents per time period)
    # copied_df['target'] = copied_df.groupby('dateofincident').size()
    copied_df['dateofincident'] = copied_df['dateofincident'].dt.tz_localize(None)
    copied_df['date_for_merge'] = pd.to_datetime(copied_df['dateofincident'].dt.date)

    # Adding the target variable (rolling window) to the DataFrame
    copied_df = create_forecast_target(copied_df, forecast_window=7)

    # Scaling numerical (continuous) features
    scaler = StandardScaler()
    # Only scaling the day of week, day of month, hour, month, and target columns
    numerical_columns = [col for col in copied_df.columns 
                                        if col.endswith(('_sin', '_cos'))]
    copied_df[numerical_columns] = scaler.fit_transform(copied_df[numerical_columns])

    # Scaling the target separately from the features
    scaler = StandardScaler()
    copied_df['target'] = scaler.fit_transform(copied_df['target'])

    # Removing the incident date column since it's no longer needed
    copied_df = copied_df.drop(['id', 'dateofincident'], axis=1)

    # Concatenate all features
    # result_df = pd.concat([copied_df] + list(text_features.values()), axis=1)

    return copied_df, monthly_incidents

def create_forecast_target(df, forecast_window, date_column='dateofincident'):
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract just the date part (removing time) and count incidents per day
    daily_incidents = df.groupby(df[date_column].dt.date).size().reset_index()
    daily_incidents.columns = [date_column, 'incident_count']
    
    # Create complete date range (filling gaps)
    date_range = pd.date_range(
        start=daily_incidents[date_column].min(),
        end=daily_incidents[date_column].max(),
        freq='D'
    )
    
    # Reindex to include all dates, filling missing values with 0
    complete_df = daily_incidents.set_index(date_column).reindex(date_range).fillna(0)
    complete_df = complete_df.reset_index()
    complete_df.columns = [date_column, 'incident_count']
    
    # Create target variable (incidents in next N days)
    complete_df['target'] = complete_df['incident_count'].rolling(
        window=forecast_window,
        min_periods=1,
        center=False
    ).sum().shift(-forecast_window)

    # Renaming 'dateofincident' to 'date' for merging
    complete_df['date_for_merge'] = complete_df['dateofincident']
    complete_df = complete_df.drop('dateofincident', axis=1)
    # Removing rows that don't have a corresponding date in the original DataFrame
    complete_df = complete_df[complete_df['incident_count'] > 0]

    # print(complete_df.head(30))
    # print(complete_df.tail(30))
    # print(complete_df['date'])

    # Merging the two DataFrames to add the target column to the original
    result = pd.merge(df, complete_df, 'left', on='date_for_merge')

    # Dropping rows where target is NaN
    result = result[result['target'].notna()]

    # Adjusting the target column to also include incidents that occurred later in the same day
    result = result.sort_values('dateofincident')

    # Group by date and apply the increment logic
    def apply_chronological_increment(group):
        # Count number of rows in the group
        group_size = len(group)
        
        # If only one row, return the group unchanged
        if group_size == 1:
            return group
        
        # Create a copy of the group to modify
        modified_group = group.copy()
        
        # Increment target for rows based on their position
        for i in range(group_size - 1):
            modified_group.iloc[i, modified_group.columns.get_loc('target')] += (group_size - 1 - i)
        
        return modified_group
    
    # Apply the increment logic to groups with the same date
    result = result.groupby(result['date_for_merge'].dt.date, group_keys=False).apply(apply_chronological_increment)
    
    # Removing extra columns that we don't need anymore
    result = result.drop(['incident_count', 'date_for_merge'], axis=1)
    # print(result.head(30))

    return result

def process_dates(df):
    df['dateofincident'] = pd.to_datetime(df['dateofincident'])
    monthly_incidents = df.resample('M', on='dateofincident').size()

    # Extracting the day of the week, month, and hour from the datetime column and cyclical encoding them
    df['day_of_week_sin'], df['day_of_week_cos'] = cyclical_encode(df['dateofincident'].dt.dayofweek, 7)
    df['month_sin'], df['month_cos'] = cyclical_encode(df['dateofincident'].dt.month, 12)
    df['hour_sin'], df['hour_cos'] = cyclical_encode(df['dateofincident'].dt.hour, 24)
    df['day_of_month_sin'], df['day_of_month_cos'] = cyclical_encode(df['dateofincident'].dt.day, 31)

    # Adding a flag to capture whether the incident occurred on a weekend or not
    df['weekend'] = np.where(df['dateofincident'].dt.dayofweek > 4, 1, 0)

    # Capturing the season when a particular incident took place
    df['season'] = df['dateofincident'].dt.month.apply(lambda x: SEASON_DICT[x])
    season_ohe = pd.get_dummies(df['season'], prefix="season")
    df = pd.concat([df, season_ohe], axis=1)

    df = df.drop(columns=['dateposted', 'datereported', 'season'], axis=1)

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

"""
Create sine and cosine encoding for cyclical features
"""
def cyclical_encode(data, max_val):
    data = 2 * np.pi * data / max_val
    return np.sin(data), np.cos(data)

"""
Creates the dataloader and model objects to use for training.
"""
def prepare_data(sequence_length=7, batch_size=5):
    # Loading the data
    print("Loading the data...")
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)

    df, monthly_incidents = load_and_transform_data(df)

    # visualize_time_series(monthly_incidents)

    print("New DataFrame columns:", df.columns.tolist())

    print("Splitting the data...")
    train_df, test_df, val_df = split_data(df, TRAIN_RATIO, VAL_RATIO)

    print("Creating datasets...")
    train_dataset = IncidentDataset(train_df, sequence_length=sequence_length)
    test_dataset = IncidentDataset(test_df, sequence_length=sequence_length)
    val_dataset = IncidentDataset(val_df, sequence_length=sequence_length)

    # monthly_incidents = dataset.get_monthly_incidents()
    print("Number of features in training data:", train_dataset.features.shape[1])
    print("Number of features in test data:", test_dataset.features.shape[1])
    print("Number of features in training data:", val_dataset.features.shape[1])

    print("Creating dataloaders...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    input_size = train_dataset.features.shape[1]

    model = Forecaster(input_size=input_size,
                       hidden_size=128)
    
    return train_loader, test_loader, val_loader, model

"""
Takes the original DataFrame and separates  it into training, testing, and validation sets.
"""
def split_data(df, train_ratio, val_ratio):
    # Calculate split points
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    # Split the data
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]

    return train_df, test_df, val_df

"""
Class to define the neural network architecture.
"""
class Forecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        print(f"Initializing model with input_size={input_size}")
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.1)
        # Output layer
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x is the input to the neural network
        # print(f"Input shape: {x.shape}")
        output, _ = self.lstm(x)
        predictions = self.linear(output[:, -1, :])
        return predictions

def train_model(train_loader, val_loader, model, epochs=250):
    print("Beginning training...")

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')

    best_vloss = 1_000_000.

    for epoch in range(epochs):
        print(f"EPOCH {epoch}")
        model.train(True)

        running_loss = 0.
        avg_loss = 0.

        for i, data in enumerate(train_loader):
            inputs, labels = data
            # Clearing the stored gradient matrix for each batch
            optimizer.zero_grad()

            # Getting the model's current prediction
            prediction = model(inputs)
            # Computing the loss between the model's prediction and the actual true result
            loss = loss_fn(prediction, labels)
            # Computing gradidents of the loss
            loss.backward()
            # Gradient descent step to update the weights
            optimizer.step()

            running_loss += loss.item()
            # Printing some metrics every 100 rows processed
            if i % 100 == 99:
                avg_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, avg_loss))
                tb_x = epoch * len(train_loader) + i + 1
                writer.add_scalar('Loss/train', avg_loss, tb_x)
                running_loss = 0.

        # Evaluating the model's accuracy
        running_vloss = 0.0
        model.eval()

        # Disable gradient computation and reduce memory consumption
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

    return model

"""
Evaluating the model on the test set.
"""
def eval_model(test_loader, model):
    model.eval()
    loss_fn = torch.nn.MSELoss()
    test_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            test_loss += loss_fn(y_pred, y_batch).item()
            predictions.extend(y_pred.numpy())
            actuals.extend(y_batch.numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')
    
    return np.array(predictions), np.array(actuals)

def main():
    train_loader, test_loader, val_loader, model = prepare_data()
    model = train_model(train_loader, val_loader, model)

    # Getting the model's results on the test set
    predictions, actuals = eval_model(test_loader, model)

    r2 = r2_score(actuals, predictions)
    print(f"R2 Score: {r2}")    

if __name__ == "__main__":
    main()