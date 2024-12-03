"""
Generates a forecast of the total number of incidents over time.
"""

from locationclassifier import process_dates
from sqlalchemy import create_engine
import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import user, password, host, port, dbname
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose

TABLE_NAME = "incidents"
# number of months to forecast ahead
NUM_STEPS = 13

def visualize_time_series(incident_series):
    # Create the figure
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot the data

    # Calculate moving average (3-month window)
    moving_avg = incident_series.rolling(window=3).mean()

    # Plot both the actual data and moving average
    ax.plot(incident_series.index, incident_series.values, 
            label='Monthly Incidents', color='#2E86C1', alpha=0.6)
    ax.plot(moving_avg.index, moving_avg.values, 
            label='8-Week Moving Average', color='#E74C3C', linewidth=2)

    # Fill the area under the curve
    ax.fill_between(incident_series.index, incident_series.values, 
                    alpha=0.2, color='#2E86C1')

    # Customize the plot
    ax.set_title('Count of TMU Campus Security Incidents Per Month', fontsize=14, pad=15)
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

def decompose(weekly_incidents):
     # Breaking down the time series into seasonal patterns, using a season period of 52 (i.e. 52 weeks = 1 year)
     result = seasonal_decompose(weekly_incidents, model='additive', period=52)
     trend = result.trend.dropna()
     seasonal = result.seasonal.dropna()
     residual = result.resid.dropna()

     # Plot the decomposed components
     plt.figure(figsize=(6,6))

     plt.subplot(4, 1, 1)
     plt.plot(weekly_incidents, label='Original Series')
     plt.legend()

     plt.subplot(4, 1, 2)
     plt.plot(trend, label='Trend')
     plt.legend()

     plt.subplot(4, 1, 3)
     plt.plot(seasonal, label='Seasonal')
     plt.legend()

     plt.subplot(4, 1, 4)
     plt.plot(residual, label='Residuals')
     plt.legend()

     plt.tight_layout()
     plt.show()

def create_model(monthly_incidents):
     sarima = pm.auto_arima(monthly_incidents,
                           start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3, m=6,
                           start_P=0, seasonal=True,
                           d=None, D=1,
                           trace=False,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
     
     return sarima 

def forecast_incidents(monthly_incidents, sarima, forecast_steps=NUM_STEPS):
    # Generate forecast
    forecast, conf_int = sarima.predict(n_periods=forecast_steps, return_conf_int=True)
    
    # Create a date range for the forecast periods
    last_date = monthly_incidents.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=30), 
                                    periods=forecast_steps, 
                                    freq='M')
    
    # Convert forecast to a pandas Series with appropriate index
    forecast_series = pd.Series(forecast, index=forecast_dates)
    
    return forecast_series, conf_int

def plot_forecast(monthly_incidents, forecast_series, conf_int=None):
    # Create the figure
    plt.figure(figsize=(15, 7))
    
    # Plot the original time series
    plt.plot(monthly_incidents.index, monthly_incidents.values, 
             label='Historical Weekly Incidents', color='#2E86C1', alpha=0.7)
    
    # Plot the forecast
    plt.plot(forecast_series.index, forecast_series.values, 
             label='Forecasted Incidents', color='#E74C3C', linestyle='--')
    
    # Plot confidence intervals if provided
    if conf_int is not None:
        plt.fill_between(forecast_series.index, 
                         conf_int[:, 0], conf_int[:, 1], 
                         color='#E74C3C', alpha=0.2, 
                         label='95% Confidence Interval')
    
    # Customize the plot
    plt.title('TMU Campus Security Incidents: Historical and Forecast', fontsize=14, pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Incidents', fontsize=12)
    
    # Format y-axis to show whole numbers only
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend()
    
    # Add padding
    plt.tight_layout()
    
    # Show the plot
    plt.show()


def main():
        print("Loading the data...")
        engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)

        _, monthly_incidents, _ = process_dates(df)

        # visualize_time_series(weekly_incidents)

        # Subtracting a lagged version of the time series from itself (1 month lag)
        weekly_diff = monthly_incidents.diff(periods=4)
        weekly_diff.dropna(inplace=True)

        # visualize_time_series(weekly_diff)

        # print(weekly_incidents.index.month)

        # Create and fit the SARIMA model
        sarima_model = create_model(monthly_incidents)
        
        # Generate forecast
        forecast_series, conf_int = forecast_incidents(monthly_incidents, sarima_model)
        
        # Plot the forecast
        plot_forecast(monthly_incidents, forecast_series, conf_int)
        
        # Optional: print out the forecast values
        print(f"\nForecast for the next {NUM_STEPS} months:")
        print(forecast_series)

if __name__ == "__main__":
    main()