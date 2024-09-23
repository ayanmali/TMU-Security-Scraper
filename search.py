# For manipulating and transforming the data
import pandas as pd

# # For analytics and ML
# from matplotlib import pyplot as plt
# # import scikit-learn

# For the database connection
from sqlalchemy import create_engine
import psycopg2
from psycopg2 import sql
from pgvector.psycopg2 import register_vector

# For generating vector embeddings of incident details
from openai import OpenAI

# Database credentials
import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import db_params, user, password, host, port, dbname

import numpy as np
from numpy.linalg import norm

# Defining constants
# DETAILS_EMBED_COLUMN_NAME = "detailsembed"
LOCDETAILS_EMBED_COLUMN_NAME = "locdetailsembed"
LOCDESCR_EMBED_COLUMN_NAME = "locdescrembed"
TABLE_NAME = "incidents"
EMBED_MODEL = "text-embedding-3-small"
N_DIMS = 256 + 128
# COMBINED_N_DIMS = N_DIMS + 128

"""
- Search feature ✅
    - Can use: OpenAI/Voyage Embeddings API, PGVector, TypeSense for search
    1. Generate embeddings
    2. Store embeddings in DB with PGVector
    3. Query from Python script

- Similar Incident Recommendation System
    - Features such as incident type, suspect descriptions, date and time, incident details (bag of words, tf-idf), and location data (latitude, longitude, proximity to landmarks)
    - KNN, matrix factorization

- Time Series Forecasting
    - likelihood of incidents occurring during specific times or days
    - number of incidents expected to occur in future time periods (ARIMA, Prophet, SARIMA to account for trends and seasonality)

- Clustering and Anomaly Detection
    - K-Means or DBSCAN to identify patterns in incident characteristics
    - Train an autoencoder or One Class SVM to identify incidents that deviate from typical patterns
    - Look for anomalies in incident frequencies

- Location Based Risk Assessment
    - Spatial Clustering to identify areas with high concentrations of incidents/high risk areas
    - Recommendation Algorithm to suggest patrol routes/areas to focus on based on historical data

- Incident Type Prediction (Classifier Model)
- Topic Modelling Using Latent Dirichlet Allocation to find underlying themes in incident descriptions or suspect descriptions

"""

"""
Adds vector embeddings to each row of the table (i.e. each incident) based on its incident details and location + incident details, respectively
"""
def add_embeddings(cur, conn, client):
    # Use sql.Identifier to properly quote the table name
    query = sql.SQL("SELECT MAX(id) FROM {}").format(sql.Identifier(TABLE_NAME))
    cur.execute(query)
    num_rows = cur.fetchone()[0]
    print(f"Found {num_rows} rows")

    for row in range(1, num_rows+1):
        print(f"Adding embeddings for row {row}...")

        # Getting the location and incident details for the given row
        query = sql.SQL("SELECT location, incidentdetails, description FROM {} WHERE id = %s").format(sql.Identifier(TABLE_NAME))
        cur.execute(query, (row, ))
        result = cur.fetchone()
        location_string, details_string, description_string = result[0], result[1], result[2]

        # Creating a combined string that contains both the location and details of the incident
        locdetails_string = location_string + " " + details_string
        # Creating a combined string that contains both the location and suspect description of the incident
        locdescr_string = location_string + " " + description_string

        # Getting the embedding for the incident details string for the given row
        print(f"Generating embeddings for row {row}")
        locdetails_embedding = get_embedding(client, locdetails_string)
        locdescr_embedding = get_embedding(client, locdescr_string)

        # Storing both embeddings in the table in their appropriate columns
        query = sql.SQL("UPDATE {} SET {} = %s, {} = %s WHERE id = %s").format(sql.Identifier(TABLE_NAME), sql.Identifier(LOCDETAILS_EMBED_COLUMN_NAME), sql.Identifier(LOCDESCR_EMBED_COLUMN_NAME))
        cur.execute(query, (locdetails_embedding, locdescr_embedding, row))

        # Commiting changes to the DB
        conn.commit()

"""
Generates a vector embedding for the given string.
"""
def get_embedding(client, input_str):
    # Generating the embedding based on the input string
    response = client.embeddings.create(
        input=input_str,
        model=EMBED_MODEL,
        dimensions=N_DIMS
    )
    return response.data[0].embedding

"""
Returns the top n most relevant search results based on the given query.
"""
def get_search_results(cur, client, engine, search_query, vector_column, n=5):
    # Loading the data into a DataFrame
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)
    query_embedding = get_embedding(client, search_query)

    # Selecting the vector column in which to perform the search
    col = LOCDESCR_EMBED_COLUMN_NAME if vector_column == 1 else LOCDETAILS_EMBED_COLUMN_NAME

    # Creates a column to store the cosine similarity between each row's vector embedding and the search query's vector embedding
    df['Similarity'] = df[col].apply(lambda x: get_cos_similarity(x, query_embedding))

    # Returns the n rows with the greatest cosine similarity
    return df.sort_values("Similarity", ascending=False, ignore_index=True)[['id', 'incidentdetails', 'location', 'description']].head(n)

def get_cos_similarity(first, second):
    return np.dot(first, second)/(norm(first)*norm(second))

"""
Creates the cursor and connection objects for interacting with the database.
"""
def setup_db():
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    return conn, cur

def main():
    # Setting up the database connection
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')

    # Setting up the database connection
    conn, cur = setup_db()
    register_vector(conn)

    # Initializing the client to make OpenAI API requests
    client = OpenAI()

    # Adds embeddings for existing records in the database
    # add_embeddings(cur, conn, client)

    # Vector column 0 corresponds to location + incident details, 1 corresponds to location + suspect description
    print(get_search_results(cur, client, engine, "", vector_column=0, n=5))
    
    # Closing the database connection
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()