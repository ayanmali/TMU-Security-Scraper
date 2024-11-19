from times import visualize_time_series, load_and_transform_data
from sqlalchemy import create_engine
import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import user, password, host, port, dbname
import pandas as pd

TABLE_NAME = "incidents"

print("Loading the data...")
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')
df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)

_, monthly_incidents = load_and_transform_data(df)
visualize_time_series(monthly_incidents)