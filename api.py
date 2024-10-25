from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import psycopg2
from psycopg2 import sql
import datetime
from sqlalchemy import create_engine
from openai import OpenAI

# from typing import List

from search import get_search_results

import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import db_params, user, password, host, port, dbname

import pickle

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

@app.get("/")
def root():
    return {"message" : "Welcome to the TMU Security Incidents API"}

"""
Returns an incident based on its ID from the database.
"""
@app.get("/getone/{id}")
def get_incident_by_id(id: int) -> Incident:
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
def get_recent_incidents(limit: int | None = None):
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

"""
Returns matching incidents given a search query from the search model.
"""
@app.get("/search/")
def search_results(query: str, limit: int | None = None):
    if limit:
        to_retrieve = limit 
    else:
        to_retrieve = DEFAULT_NUM_RETRIEVE

    # Gets the DataFrame containing the incident records with the highest match with the query
    results = get_search_results(cur, client, engine, search_query=query, vector_column=0, n=to_retrieve)
    
    # Returns the matching incidents in the response
    return {"results": results.to_dict(orient='records')}

"""
Returns similar incidents given an incident to reference from the recommendation model.
"""
@app.get("/recommend")
def get_recommendations(incident: Incident, limit: int = 4):
    with open('tfidf_recommend_model.pkl', 'rb') as file:
        model = pickle.load(file)

    return