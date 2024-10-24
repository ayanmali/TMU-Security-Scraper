from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from psycopg2 import sql
import datetime
# from typing import List

import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import db_params#, user, password, host, port, dbname

import pickle

TABLE_NAME = "incidents"

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

app = FastAPI()

@app.get("/")
def root():
    return {"message" : "Welcome to the TMU Security Incidents API"}

"""
Returns an incident based on its ID from the database.
"""
@app.get("/getone")
def get_incident_by_id(id: int) -> Incident:
    # SQL query to retrieve the desired data from the database
    query = sql.SQL("""SELECT
                    id, incidenttype, location, page, incidentdetails, description, dateofincident, dateposted, datereported, otherincidenttype
                    FROM {} WHERE id = %s""").format(sql.Identifier(TABLE_NAME))
    cur.execute(query, (id,))
    incident_raw = cur.fetchone()

    # Error handling
    if incident_raw is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Stores the data from the DB into an Incident instance and returns it
    return Incident(**dict(zip(Incident.__fields__.keys(), incident_raw)))

"""
Gets the top N most recent incidents.
"""
@app.get("/getrecent")
def get_recent_incidents(limit: int = 5):
    query = sql.SQL("""SELECT
                    id, incidenttype, location, page, incidentdetails, description, dateofincident, dateposted, datereported, otherincidenttype
                    FROM {} ORDER BY dateofincident DESC LIMIT %s""")
    cur.execute(query, (limit,))

    total = []

    for i in range(limit):
        incident_raw = cur.fetchone()
        incident = Incident(**dict(zip(Incident.__fields__.keys(), incident_raw)))
        total.append(incident)

    return total

"""
Returns matching incidents given a search query from the search model.
"""
@app.get("/search")
def get_search_results(query: str, limit: int = 4):
    pass

"""
Returns similar incidents given an incident to reference from the recommendation model.
"""
@app.get("/recommend")
def get_recommendations(incident: Incident, limit: int = 4):
    with open('tfidf_recommend_model.pkl', 'rb') as file:
        model = pickle.load(file)

    return