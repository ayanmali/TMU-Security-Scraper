# For manipulating and transforming the data
import pandas as pd

# For analytics and ML
from matplotlib import pyplot as plt
# import scikit-learn

# For the database connection
# from sqlalchemy import create_engine
import psycopg2
from psycopg2 import sql

from pgvector.psycopg2 import register_vector

# Database credentials
import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import db_params

# Defining constants
EMBED_COLUMN_NAME = "detailsembedding"
TABLE_NAME = "incidents"

"""
Creates the cursor and connection objects for interacting with the database.
"""
def setup_db():
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    return conn, cur

"""
- Vector embeddings
    - Search feature
    - Can use: OpenAI/Voyage Embeddings API, PGVector, TypeSense for search
    1. Generate embeddings
    2. Store embeddings in DB with PGVector
    3. Query from Python script
- Time Series Forecasting
    - likelihood of incidents occurring during specific times or days
- Clustering and Anomaly Detection
    - K-Means or DBSCAN to identify patterns in incident characteristics
    - Train an autoencoder or One Class SVM to identify incidents that deviate from typical patterns
- Incident Type Prediction (Classifier Model)
- Location Based Risk Assessment
    - Spatial Clustering to identify areas with high concentrations of incidents
    - Recommendation Algorithm to suggest patrol routes/areas to focus on based on historical data
- Topic Modelling Using Latent Dirichlet Allocation to find underlying themes in incident descriptions or suspect descriptions

"""

"""
Adds a vector embedding to each row of the table (i.e. each incident) based on it's incident details.
"""
def add_embeddings(cur, conn):
    num_rows = cur.query("SELECT MAX(id) FROM incidents")

    for row in range(1, num_rows):
        cur.query(f"UPDATE incidents SET detailsembedding = '[]' WHERE id = {row}")

def main():
    # # Setting up the database connection
    # engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')

    # # Loading the data into a DataFrame
    # df = pd.read_sql("SELECT  * FROM incidents", engine)
    # Setting up the database connection
    conn, cur = setup_db()
    register_vector(conn)

if __name__ == "__main__":
    main()