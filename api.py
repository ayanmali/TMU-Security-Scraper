from typing import Annotated
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import psycopg2
from psycopg2 import sql
import datetime
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from openai import OpenAI

# from typing import List

from search import get_search_results, LOCDETAILS_EMBED_COLUMN_NAME, LOCDESCR_EMBED_COLUMN_NAME
from recommend_tfidf_algo import get_recommendations, load_and_transform_data, train_model, N_NEIGHBORS
# from preprocessing import format_url

import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import db_params, user, password, host, port, dbname

# import pickle

TABLE_NAME = "incidents"
DEFAULT_NUM_RETRIEVE = 5

"""
Creates the connection and cursor objects to interact with the database.
"""
def setup_db():
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    # cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    return conn, cur

# Setting up the database connection
conn, cur = setup_db()
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')
client = OpenAI()

# Setting up the DataFrame to use for searches/retrieving recent incidents
df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)
# Convert string representations of vectors to numpy arrays
df[LOCDESCR_EMBED_COLUMN_NAME] = df[LOCDESCR_EMBED_COLUMN_NAME].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
# Convert string representations of vectors to numpy arrays
df[LOCDETAILS_EMBED_COLUMN_NAME] = df[LOCDETAILS_EMBED_COLUMN_NAME].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)

# Creating the DataFrame to use for recommendations
recommend_df, _ = load_and_transform_data(df)
# Training the recommendation model
knn = train_model(recommend_df, N_NEIGHBORS)

# Creating a copy of the DataFrame to use for KNN recommendations
"""
Defining the Pydantic model to represent the data sent in endpoint responses.
"""
class Incident(BaseModel):
    id: int 
    incident_type: str 
    location: str 
    page_url: str
    incident_description: str 
    suspect_description: str
    date_of_incident: datetime.datetime
    date_posted: datetime.datetime
    date_reported: datetime.datetime
    other_incident_type: str

app = FastAPI()

# origins = [
#     "http://localhost.tiangolo.com",
#     "https://localhost.tiangolo.com",
#     "http://localhost",
#     "http://localhost:8080",
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message" : "Welcome to the TMU Security Incidents API"}

"""
Returns an incident based on its ID from the database.
"""
@app.get("/getone/{id}")
# Input validation for the ID
def get_incident_by_id(id: Annotated[int, Path(title="The ID of the item to get", ge=0)]) -> Incident:
    # SQL query to retrieve the desired record from the database
    query = sql.SQL("""SELECT
                    id, incidenttype, location, page, incidentdetails, description, dateofincident, dateposted, datereported, otherincidenttype
                    FROM {} WHERE id = %s""").format(sql.Identifier(TABLE_NAME))
    cur.execute(query, (id,))
    incident_raw = cur.fetchone()

    # Error handling
    if incident_raw is None:
        raise HTTPException(status_code=404, detail="Incident not found. The ID may not be valid.")
    
    # Stores the data from the DB into an Incident instance and returns it
    return Incident(**dict(zip(Incident.__fields__.keys(), incident_raw)))

"""
Gets the top N most recent incidents.
"""
@app.get("/getrecent/")
def get_recent_incidents(limit: Annotated[
        int | None, 
        Query(
            title="Number of incidents to retrieve",
            description="If not specified, returns default number of incidents",
            ge=1,
        )
    ] = None):
    if limit:
        to_retrieve = limit
    else:
        to_retrieve = DEFAULT_NUM_RETRIEVE

    # Query to retrieve the top N records with the most recent date of occurence
    query = sql.SQL("""SELECT
                    id, incidenttype, location, page, incidentdetails, description, dateofincident, dateposted, datereported, otherincidenttype
                    FROM {} ORDER BY dateofincident DESC LIMIT %s""").format(sql.Identifier(TABLE_NAME))
    cur.execute(query, (to_retrieve,))

    # List to store all incident records being retrieved
    incidents = []

    # Adds each incident record to the list
    for i in range(to_retrieve):
        incident_raw = cur.fetchone()

        # Object to store the incident data from the DB record
        incident = Incident(**dict(zip(Incident.__fields__.keys(), incident_raw)))
        # Stores the JSON representation of the object into the list
        incidents.append(jsonable_encoder(incident))

    # Returns the incident records and number of records as a JSON object
    return {
        "count": len(incidents),
        "incidents": incidents
    }

# Simulates a POST request to add an incident to the DB
# @app.post("/addincident/{incident}")
# def add_incident(incident: Incident):
#     return incident

"""
Returns matching incidents given a search query from the search model.
"""
@app.get("/search")
def search_results(query: Annotated[str, Query(min_length=1)], limit: Annotated[
        int | None, 
        Path(
            title="Number of incidents to retrieve",
            description="If not specified, returns default number of incidents",
            ge=1
        )
    ] = None):
    if limit:
        to_retrieve = limit 
    else:
        to_retrieve = DEFAULT_NUM_RETRIEVE

    # Gets the DataFrame containing the incident records with the highest match with the query
    results = get_search_results(cur, client, engine, search_query=query, vector_column=0, df=df, n=to_retrieve)
    
    # Returns the matching incidents in the response
    return {"results": results.to_dict(orient='records')}

"""
Returns similar incidents given an incident to reference from the recommendation model.
"""
@app.get("/recommend/")
def get_recommend(incident_url: str, limit: Annotated[
        int | None, 
        Path(
            title="Number of incidents to retrieve",
            description="If not specified, returns default number of incidents",
            ge=1
        )
    ] = None):
    # Importing the K-nearest neighbours model
    # with open('tfidf_recommend_model.pkl', 'rb') as file:
    #     model = pickle.load(file)
    if limit:
        to_retrieve = limit 
    else:
        to_retrieve = DEFAULT_NUM_RETRIEVE

    # Using the given incident to determine other recommended incidents
    # results = get_recommendations(id, recommend_df, knn, n_recommendations=to_retrieve)

    # Returns the matching incidents in the response
    return {"results": results.to_dict(orient='records')}