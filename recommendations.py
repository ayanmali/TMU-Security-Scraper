# For converting text features into vectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import NearestNeighbors

# Can try either using TFIDFVectorizer or embeddings to see which works better
# from openai import OpenAI

import pandas as pd
# For the database connection
from sqlalchemy import create_engine
import psycopg2
from psycopg2 import sql
from pgvector.psycopg2 import register_vector

# Database credentials
import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import db_params, user, password, host, port, dbname

TABLE_NAME = "incidents"

def load_and_transform_data(engine):
    # Loading the data into a DataFrame
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY id", engine)

    # For incident type
    one_hot_encoding(df)
    # For the date/time of the incident
    get_dates(df)

    text_feature_names = {}
    vectorizers = {}

    # For incident details, locations, and suspect descriptions
    for col in ('incidentdetails', 'description', 'location'):
        df, text_feature_names[col], vectorizers[col] = extract_text_features(df, col=col)

    df = scale_text_features(df, text_feature_names)

    return df, vectorizers

def one_hot_encoding(df):
    df['incidenttype_cleaned'] = df['incidenttype'].replace({": Suspect Arrested" : ""}, regex=True)
    df = pd.concat(df, pd.get_dummies(df['incidenttype_cleaned'], prefix='incidenttype'), axis=1)

def get_dates(df):
    df['day_of_week'] = df['dateofincident'].dt.dayofweek
    df['month'] = df['dateofincident'].dt.month
    df['hour'] = pd.to_datetime(df['time']).dt.hour

    # One hot encoding the new date/time columns
    day_dummies = pd.get_dummies(df['day_of_week'], prefix='day')
    month_dummies = pd.get_dummies(df['month'], prefix='month')

    # Renaming the columns to actual day names
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_dummies.columns = [f'is_{day}' for day in day_names]
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_dummies.columns = [f'is_{month}' for month in month_names]

    # Adding the new columns to the original DataFrame
    df = pd.concat([df, day_dummies, month_dummies], axis=1)

"""
Extracts features from the incident details, location, and description columns.
"""
def extract_text_features(df, col):
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(df[col])
    array = matrix.to_array()

    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(array, columns=feature_names)

    df = pd.concat([df, tfidf_df], axis=1)

    return df, feature_names, vectorizer

"""
Scales the columns in the DataFrame corresponding to text feature data using a StandardScaler.
"""
def scale_text_features(df, text_feature_names):
    for col in text_feature_names.keys():
        scaler = StandardScaler()
        scaled_text_features = scaler.fit_transform(df[text_feature_names[col]])
        df = pd.concat(df, scaled_text_features)

    return df

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

    conn, cur = setup_db()
    register_vector(conn)

    df, vectorizers = load_and_transform_data(engine)
    # print(df['dateofincident'])


if __name__ == "__main__":
    main()