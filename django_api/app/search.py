# For manipulating and transforming the data
import pandas as pd

# # For analytics and ML
# from matplotlib import pyplot as plt
# # import scikit-learn

# For the database connection
from sqlalchemy import create_engine
import psycopg2
from psycopg2 import sql
# from pgvector.psycopg2 import register_vector

# For generating vector embeddings of incident details
from openai import OpenAI

# Database credentials
# import sys
# sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from .postgres_params import DB_PARAMS, USER, PASSWORD, HOST, PORT, DBNAME

import numpy as np
from numpy.linalg import norm

# Defining constants
DETAILS_EMBED_COLUMN_NAME = "detailsembed"
LOCATION_EMBED_COLUMN_NAME = "locationembed"
DESCRIPTION_EMBED_COLUMN_NAME = "descrembed"
LOCDETAILS_EMBED_COLUMN_NAME = "locdetailsembed"
LOCDESCR_EMBED_COLUMN_NAME = "locdescrembed"
ALL_EMBED_COLUMN_NAME = "allembed"
TABLE_NAME = "incidents"
EMBED_MODEL = "text-embedding-3-small"
N_DIMS = 256 + 128
# N_DIMS_ALL = 256 + 256 + 128

"""
Adds vector embeddings to each row of the table (i.e. each incident) based on its incident details and location + incident details and suspect description, respectively
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
        details_embedding = get_embedding(client, details_string, dims=N_DIMS-128)
        locdetails_embedding = get_embedding(client, locdetails_string)
        locdescr_embedding = get_embedding(client, locdescr_string)

        # Storing both embeddings in the table in their appropriate columns
        query = sql.SQL("UPDATE {} SET {} = %s, {} = %s, {} = %s WHERE id = %s").format(sql.Identifier(TABLE_NAME), sql.Identifier(DETAILS_EMBED_COLUMN_NAME), sql.Identifier(LOCDETAILS_EMBED_COLUMN_NAME), sql.Identifier(LOCDESCR_EMBED_COLUMN_NAME))
        cur.execute(query, (details_embedding, locdetails_embedding, locdescr_embedding, row))

        # Commiting changes to the DB
        conn.commit()

"""
Adds vector embeddings to each row of the table (i.e. each incident) based on its location and description respectively
"""
def add_loc_and_descr_embeddings(cur, conn, client):
    # Use sql.Identifier to properly quote the table name
    query = sql.SQL("SELECT MAX(id) FROM {}").format(sql.Identifier(TABLE_NAME))
    cur.execute(query)
    num_rows = cur.fetchone()[0]
    print(f"Found {num_rows} rows")

    for row in range(1, num_rows+1):
        print(f"Adding embeddings for row {row}...")

        # Getting the location and incident details for the given row
        query = sql.SQL("SELECT location, description FROM {} WHERE id = %s").format(sql.Identifier(TABLE_NAME))
        cur.execute(query, (row, ))
        result = cur.fetchone()
        location_string, description_string = result[0], result[1]

        # Getting the embedding for the incident details string for the given row
        print(f"Generating embeddings for row {row}")
        loc_embedding = get_embedding(client, location_string, dims=N_DIMS-256)
        descr_embedding = get_embedding(client, description_string, dims=N_DIMS-128)

        # Storing both embeddings in the table in their appropriate columns
        query = sql.SQL("UPDATE {} SET {} = %s, {} = %s WHERE id = %s").format(sql.Identifier(TABLE_NAME), sql.Identifier(LOCATION_EMBED_COLUMN_NAME), sql.Identifier(DESCRIPTION_EMBED_COLUMN_NAME))
        cur.execute(query, (loc_embedding, descr_embedding, row))

        # Commiting changes to the DB
        conn.commit()

"""
Generates a vector embedding for the given string.
"""
def get_embedding(client, input_str, dims=N_DIMS):
    # Generating the embedding based on the input string
    response = client.embeddings.create(
        input=input_str,
        model=EMBED_MODEL,
        dimensions=dims
    )
    return response.data[0].embedding

"""
Returns the top n most relevant search results based on the given query.
"""
def get_search_results(client, search_query, vector_column, df, n=5):
    query_embedding = get_embedding(client, search_query)

    # Selecting the vector column in which to perform the search
    col = LOCDETAILS_EMBED_COLUMN_NAME if vector_column == 0 else LOCDESCR_EMBED_COLUMN_NAME

    # Convert string representations of vectors to numpy arrays
    # _df[col] = _df[col].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)

    # Creates a column to store the cosine similarity between each row's vector embedding and the search query's vector embedding
    df['Similarity'] = df[col].apply(lambda x: get_cos_similarity(x, query_embedding))

    # Returns the n rows with the greatest cosine similarity
    return df.sort_values("Similarity", ascending=False, ignore_index=True)[['id', 'incidentdetails', 'location', 'description', 'incidenttype', 'dateofincident']].head(n)

def get_cos_similarity(first, second):
    # Ensure both inputs are numpy arrays
    first = np.array(first) if not isinstance(first, np.ndarray) else first
    second = np.array(second) if not isinstance(second, np.ndarray) else second
    return np.dot(first, second)/(norm(first)*norm(second))

"""
Creates the cursor and connection objects for interacting with the database.
"""
def setup_db():
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    # cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    return conn, cur

# For testing out the search functionality
def main():
    # Setting up the database connection
    engine = create_engine(f'postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}')

    # Setting up the database connection
    conn, cur = setup_db()
    # register_vector(conn)

    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)

    # Initializing the client to make OpenAI API requests
    client = OpenAI()

    # Adds embeddings for existing records in the database
    # add_embeddings(cur, conn, client)
    # add_loc_and_descr_embeddings(cur, conn, client)

    # Convert string representations of vectors to numpy arrays
    df[LOCDESCR_EMBED_COLUMN_NAME] = df[LOCDESCR_EMBED_COLUMN_NAME].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
    df[LOCDETAILS_EMBED_COLUMN_NAME] = df[LOCDETAILS_EMBED_COLUMN_NAME].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)

    # Search query
    query = "stabbing"
    # Vector column 0 corresponds to location + incident details, 1 corresponds to location + suspect description
    print(get_search_results(cur, client, engine, query, vector_column=0, df=df, n=5))
    
    # Closing the database connection
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()