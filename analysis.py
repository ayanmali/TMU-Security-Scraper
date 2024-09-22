# # For manipulating and transforming the data
# import pandas as pd

# # For analytics and ML
# from matplotlib import pyplot as plt
# # import scikit-learn

# For the database connection
# from sqlalchemy import create_engine
import psycopg2
from psycopg2 import sql
from pgvector.psycopg2 import register_vector

# For generating vector embeddings of incident details
from openai import OpenAI

# Database credentials
import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import db_params

# Defining constants
EMBED_COLUMN_NAME = "detailsembed"
TABLE_NAME = "incidents"
EMBED_MODEL = "text-embedding-3-small"
N_DIMS = 256

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
Adds a vector embedding to each row of the table (i.e. each incident) based on its incident details.
"""
def add_embeddings(cur, conn, client):
    # Use sql.Identifier to properly quote the table name
    query = sql.SQL("SELECT MAX(id) FROM {}").format(sql.Identifier(TABLE_NAME))
    cur.execute(query)
    num_rows = cur.fetchone()[0]
    print(num_rows)

    for row in range(1, num_rows+1):
        print(f"Adding embedding for row {row}...")
        # details_query = sql.SQL("""
        # SELECT incidentdetails FROM %s WHERE id = %s
        #     """)
        query = sql.SQL("SELECT incidentdetails FROM {} WHERE id = %s").format(sql.Identifier(TABLE_NAME))
        cur.execute(query, (row, ))
        details_string = cur.fetchone()

        # Getting the embedding for the incident details string for the given row
        print(f"Generating embedding for row {row}")
        embedding = get_embedding(client, details_string)
        # Storing the embedding in the table in the appropriate column
        query = sql.SQL("UPDATE {} SET {} = %s WHERE id = %s").format(sql.Identifier(TABLE_NAME), sql.Identifier(EMBED_COLUMN_NAME))
        cur.execute(query, (embedding, row))

        # Commiting changes to the DB
        conn.commit()

"""
Generates a vector embedding for the given string.
"""
def get_embedding(client, input_str):
    response = client.embeddings.create(
        input=input_str,
        model=EMBED_MODEL,
        dimensions=256
    )
    return response.data[0].embedding

"""
Creates the cursor and connection objects for interacting with the database.
"""
def setup_db():
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    return conn, cur

def main():
    # # Setting up the database connection
    # engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')

    # # Loading the data into a DataFrame
    # df = pd.read_sql("SELECT  * FROM incidents", engine)

    # Setting up the database connection
    conn, cur = setup_db()
    register_vector(conn)

    client = OpenAI()

    add_embeddings(cur, conn, client)
    
    # Closing the database connection
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()