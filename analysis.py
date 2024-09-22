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
DETAILS_EMBED_COLUMN_NAME = "detailsembed"
COMBINED_EMBED_COLUMN_NAME = "combinedembed"
TABLE_NAME = "incidents"
EMBED_MODEL = "text-embedding-3-small"
DETAILS_N_DIMS = 256
COMBINED_N_DIMS = DETAILS_N_DIMS + 128

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
        # details_query = sql.SQL("""
        # SELECT incidentdetails FROM %s WHERE id = %s
        #     """)

        # Getting the location and incident details for the given row
        query = sql.SQL("SELECT location, incidentdetails FROM {} WHERE id = %s").format(sql.Identifier(TABLE_NAME))
        cur.execute(query, (row, ))
        result = cur.fetchone()
        location_string, details_string = result[0], result[1]

        # Creating a combined string that contains both the location and details of the incident
        combined_string = location_string + " " + details_string

        # Getting the embedding for the incident details string for the given row
        print(f"Generating embeddings for row {row}")
        details_embedding = get_embedding(client, details_string, combined=False)
        combined_embedding = get_embedding(client, combined_string, combined=True)

        # Storing both embeddings in the table in their appropriate columns
        query = sql.SQL("UPDATE {} SET {} = %s, {} = %s WHERE id = %s").format(sql.Identifier(TABLE_NAME), sql.Identifier(DETAILS_EMBED_COLUMN_NAME), sql.Identifier(COMBINED_EMBED_COLUMN_NAME))
        cur.execute(query, (details_embedding, combined_embedding, row))

        # Commiting changes to the DB
        conn.commit()

"""
Generates a vector embedding for the given string.
"""
def get_embedding(client, input_str, combined):
    # Generating the embedding based on the input string
    response = client.embeddings.create(
        input=input_str,
        model=EMBED_MODEL,
        dimensions=COMBINED_N_DIMS if combined else DETAILS_N_DIMS
    )
    return response.data[0].embedding

"""
Returns the top n most relevant search results based on the given query.
"""
def get_search_results(cur, client, engine, search_query, combined=True, n=5):
    # Loading the data into a DataFrame
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)
    query_embedding = get_embedding(client, search_query, combined=combined)

    # Creating a column to store the cosine similarity between that row's vector embedding and the search query's vector embedding
    if combined:
        df['Cosine Similarity'] = df[COMBINED_EMBED_COLUMN_NAME].apply(lambda x: get_cos_similarity(x, query_embedding))
    else:
        df['Cosine Similarity'] = df[DETAILS_EMBED_COLUMN_NAME].apply(lambda x: get_cos_similarity(x, query_embedding))

    # results = (
    #     df.sort_values("Cosine Similarity", ascending=False)
    #     .head(n)
    #     # .combined.str.replace("Title: ", "")
    #     # .str.replace("; Content:", ": ")
    # )

    # Returns the n rows with the greatest cosine similarity
    return df.sort_values("Cosine Similarity", ascending=False, ignore_index=True)[['id', 'incidentdetails', 'location']].head(n)

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

    print(get_search_results(cur, client, engine, "Assault near the library", combined=True, n=5))
    
    # Closing the database connection
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()