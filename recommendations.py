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
N_NEIGHBORS = 5

def load_and_transform_data(engine):
    # Loading the data into a DataFrame
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY id", engine)

    # For incident type
    one_hot_encoding(df)
    # For the date/time of the incident
    get_dates(df)

    text_features = {}
    vectorizers = {}

    # For incident details, locations, and suspect descriptions
    for col in ('incidentdetails', 'description', 'location'):
        tfidf_df, _, vectorizers[col] = extract_text_features(df, col=col)
        text_features[col] = scale_text_features(tfidf_df)

    # Concatenate all features
    result_df = pd.concat([df] + list(text_features.values()), axis=1)

    return result_df, vectorizers

def one_hot_encoding(df):
    df['incidenttype_cleaned'] = df['incidenttype'].replace({": Suspect Arrested" : ""}, regex=True)
    df = pd.concat([df, pd.get_dummies(df['incidenttype_cleaned'], prefix='incidenttype')], axis=1)

"""
Extracts the month, day of week, and hour of each incident's date.
"""
def get_dates(df):
    df['day_of_week'] = df['dateofincident'].dt.dayofweek
    df['month'] = df['dateofincident'].dt.month
    df['hour'] = df['dateofincident'].dt.hour

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
Extracts features from a given text column (incident details, location, or description).
"""
def extract_text_features(df, col):
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(df[col])
    array = matrix.toarray()

    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(array, columns=[f"{col}_{name}" for name in feature_names])

    # df = pd.concat([df, tfidf_df], axis=1)

    return tfidf_df, feature_names, vectorizer

"""
Scales the columns in the DataFrame corresponding to text feature data using a StandardScaler.
"""
def scale_text_features(tfidf_feature_cols):
    # Initializing a new scaler object
    scaler = StandardScaler()
    # Creating a new DataFrame that contains the scaled text data
    scaled_features = scaler.fit_transform(tfidf_feature_cols)
    # Adding the scaled data to the original DataFrame
    return pd.DataFrame(scaled_features, columns=tfidf_feature_cols.columns)

# Function to get recommendations
def get_recommendations(df, knn, incident_id, n_recommendations=5):
    incident_vector = df[incident_id]
    distances, indices = knn.kneighbors(incident_vector.reshape(1, -1), n_neighbors=n_recommendations+1)
    
    # Exclude the incident itself
    similar_incidents = df.iloc[indices[0][1:]]
    return similar_incidents

def train_model(X):
    knn = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric='cosine')
    knn.fit(X)
    return knn

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

    # df.to_csv("recommendations.csv")
    # print(df['dateofincident'])

    knn = train_model()

if __name__ == "__main__":
    main()