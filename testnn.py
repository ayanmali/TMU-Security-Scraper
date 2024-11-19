"""
TODO: Implement SARIMA/RNN/Gradient boosted trees for time series analysis
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from sklearn.metrics import r2_score

import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')

TABLE_NAME = "incidents"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

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

    def __len__(self):
        return len(self.features) - self.sequence_length - self.prediction_length + 1
    
    def __getitem__(self, idx):
        X = self.features[idx:idx + self.sequence_length]
        y = self.target[idx + self.sequence_length : idx + self.sequence_length + self.prediction_length]
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def get_monthly_incidents(self):
        return self.monthly_incidents

"""
Creates the dataloader and model objects to use for training.
"""
def prepare_data(sequence_length=7, batch_size=1):
    # # Loading the data
    # print("Loading the data...")
    # engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')
    # df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)

    # df, monthly_incidents = load_and_transform_data(df)

    # # visualize_time_series(monthly_incidents)

    # print("New DataFrame columns:", df.columns.tolist())

    # print("Splitting the data...")
    # train_df, test_df, val_df = split_data(df, TRAIN_RATIO, VAL_RATIO)

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