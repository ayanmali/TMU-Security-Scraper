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
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# Defining the class to be used for the DataLoader
class IncidentDataset(torch.utils.data.Dataset):
    def __init__(self, df, sequence_length, prediction_length=1, scaler=None):
        # Creating a copy just in case
        copied_df = df.copy(deep=True)
        copied_df = copied_df.drop(columns=['id', 'page', 'otherincidenttype', 'detailsembed', 'locdetailsembed', 'locdescrembed', 'locationembed', 'descrembed'], axis=1)

        # For specifying the exact incident type for any incident type that is "Other"
        copied_df = replace_other_incident_type(copied_df)

        # For one hot encoding the locations
        copied_df = process_locations(copied_df)

        # For incident type
        copied_df = process_type(copied_df)

        # For the date/time of the incident
        copied_df, self.monthly_incidents = process_dates(copied_df)

        # text_features = {}
        # vectorizers = {}

        # # For incident details and suspect descriptions
        # for col in ('incidentdetails', 'description'):
        #     tfidf_df, vectorizers[col], _ = extract_text_features(copied_df, col=col)
        #     text_features[col] = scale_text_features(tfidf_df)

        # Dropping features that we don't need anymore
        copied_df = copied_df.drop(['incidentdetails', 'description'], axis=1)

        # Setting the target variable (# of incidents per time period)
        copied_df['target'] = copied_df.groupby('dateofincident').size()

        # Only scaling the day of week, day of month, hour, month, and target columns
        numerical_columns = ['target'] + [col for col in copied_df.columns 
                                            if col.endswith(('_sin', '_cos'))]
        
        # Ensuring that the scaler that was fitted to the training data is being used for the testing and validation data as well
        if (scaler is None):
            # Creating a scaler which will be fitted on the training set
            self.scaler = StandardScaler()
            copied_df[numerical_columns] = self.scaler.fit_transform(copied_df[numerical_columns])
        else:
            # Transforming the test/validation data with the training data's scaler
            self.scaler = scaler
            copied_df[numerical_columns] = self.scaler.transform(copied_df[numerical_columns])

        # Removing the incident date column since it's no longer needed
        copied_df = copied_df.drop('dateofincident', axis=1)

        self.features = copied_df.drop('target', axis=1).values
        self.target = copied_df['target'].values

        self.sequence_length = 10
        self.prediction_length = 1

        # Print debug information
        print("Feature names:", copied_df.columns.tolist())
        print("Number of features:", self.features.shape[1])
        
        # Sample a few rows to verify feature values
        print("Sample feature values:")
        print(copied_df.head())

        # Concatenate all features
        # result_df = pd.concat([copied_df] + list(text_features.values()), axis=1)

        # return copied_df, monthly_incidents

    def __len__(self):
        return len(self.features) - self.sequence_length - self.prediction_length + 1
    
    def __getitem__(self, idx):
        X = self.features[idx:idx + self.sequence_length]
        y = self.target[idx + self.sequence_length : idx + self.sequence_length + self.prediction_length]
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

    # Setting the target variable (# of incidents per time period)
    copied_df['target'] = copied_df.groupby('dateofincident').size()

    scaler = StandardScaler()
    # Only scaling the day of week, day of month, hour, month, and target columns
    numerical_columns = ['target'] + [col for col in copied_df.columns 
                                        if col.endswith(('_sin', '_cos'))]
    copied_df[numerical_columns] = scaler.fit_transform(copied_df[numerical_columns])

    # Removing the incident date column since it's no longer needed
    copied_df = copied_df.drop('dateofincident', axis=1)

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

"""
Creates the dataloader and model objects to use for training.
"""
def prepare_data(sequence_length=7, batch_size=12):
    # Loading the data
    print("Loading the data...")
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)

    print("Original DataFrame columns:", df.columns.tolist())

    print("Splitting the data...")
    # Sorting the DataFrame chronologically
    df['dateofincident'] = pd.to_datetime(df['dateofincident'])
    df = df.sort_values('dateofincident')

    train_df, test_df, val_df = split_data(df, TRAIN_RATIO, VAL_RATIO)

    print("Creating datasets...")
    train_dataset = IncidentDataset(train_df, sequence_length=sequence_length, scaler=None)
    # Reusing the training dataset scaler
    test_dataset = IncidentDataset(test_df, sequence_length, scaler=train_dataset.scaler)
    val_dataset = IncidentDataset(val_df, sequence_length, scaler=train_dataset.scaler)

    # monthly_incidents = dataset.get_monthly_incidents()
    print("Number of features in training data:", train_dataset.features.shape[1])

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
                       hidden_size=64)
    
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
        print(f"Input shape: {x.shape}")
        output, _ = self.lstm(x)
        predictions = self.linear(output[:, -1, :])
        return predictions

def train_model(train_loader, val_loader, model, epochs=100):
    print("Beginning training...")

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')

    best_vloss = 1_000_000.

    for epoch in range(epochs):
        print(f"EPOCH {epoch}")
        model.train(True)

        running_loss = 0
        avg_loss = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            # Clearing the stored gradient matrix for each iteration
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
            if i % 1000 == 999:
                avg_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, avg_loss))
                tb_x = epoch * len(train_loader) + i + 1
                writer.add_scalar('Loss/train', avg_loss, tb_x)
                running_loss = 0.

        # Evaluating the model's accuracy
        running_vloss = 0.0
        model.eval()

        # Disable gradient computation and reduce memory consumption.
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