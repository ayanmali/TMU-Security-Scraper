# For manipulating and transforming the data
import pandas as pd

# For analytics and ML
from matplotlib import pyplot as plt
# import scikit-learn

# For the database connection
from sqlalchemy import create_engine

# Database credentials
import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import host, dbname, user, password, port



def main():
    # Setting up the database connection
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')

    # Loading the data into a DataFrame
    df = pd.read_sql("SELECT  * FROM incidents", engine)

if __name__ == "__main__":
    main()
