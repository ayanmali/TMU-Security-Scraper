from locationclassifier import load_and_transform_data
from sqlalchemy import create_engine
import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import user, password, host, port, dbname
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

TABLE_NAME = "incidents"

print("Loading the data...")
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')
df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)

_, monthly_incidents, _ = load_and_transform_data(df)

def visualize_time_series(monthly_incidents):
    # Create the figure
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot the data

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

visualize_time_series(monthly_incidents)