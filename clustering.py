"""
Uses incident type, date of incident, as well as location, incident details, and suspect descriptions to suggest recommendations.
"""

# For converting text features into vectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Can try either using TFIDFVectorizer or embeddings to see which works better
# from openai import OpenAI

import pandas as pd
# For the database connection
from sqlalchemy import create_engine
# import psycopg2
# from psycopg2 import sql
# from pgvector.psycopg2 import register_vector

# Database credentials
import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import user, password, host, port, dbname #, db_params

TABLE_NAME = "incidents"
N_CLUSTERS = 5

def load_and_transform_data(engine):
    # Loading the data into a DataFrame
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY id", engine)
    # Storing a duplicate of the DataFrame for reference when recommendations are suggested
    copied_df = df.copy(deep=True)
    df = df.drop(columns=['page', 'otherincidenttype', 'detailsembed', 'locdetailsembed', 'locdescrembed', 'locationembed', 'descrembed'], axis=1)

    # For incident type
    df = one_hot_encoding(df)
    # For the date/time of the incident
    df = get_dates(df)

    text_features = {}
    vectorizers = {}

    # For incident details, locations, and suspect descriptions
    for col in ('incidentdetails', 'description', 'location'):
        tfidf_df, vectorizers[col], _ = extract_text_features(df, col=col)
        text_features[col] = scale_text_features(tfidf_df)

    df = df.drop(['incidentdetails', 'description', 'location'], axis=1)

    # Concatenate all features
    result_df = pd.concat([df] + list(text_features.values()), axis=1)

    return result_df, copied_df, vectorizers

def one_hot_encoding(df):
    df['incidenttype_cleaned'] = df['incidenttype'].replace({": Suspect Arrested" : ""}, regex=True)
    df = pd.concat([df, pd.get_dummies(df['incidenttype_cleaned'], prefix='incidenttype', dtype=int)], axis=1)
    df = df.drop(columns=['incidenttype'], axis=1)

    return df

"""
Extracts the month, day of week, and hour of each incident's date.
"""
def get_dates(df):
    df['day_of_week'] = df['dateofincident'].dt.dayofweek
    df['month'] = df['dateofincident'].dt.month
    df['hour'] = df['dateofincident'].dt.hour

    # One hot encoding the new date/time columns
    day_dummies = pd.get_dummies(df['day_of_week'], prefix='day', dtype=int)
    month_dummies = pd.get_dummies(df['month'], prefix='month', dtype=int)
    hour_dummies = pd.get_dummies(df['hour'], prefix='hour', dtype=int)

    # Renaming the columns to actual day names
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_dummies.columns = [f'is_{day}' for day in day_names]
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_dummies.columns = [f'is_{month}' for month in month_names]
    hour_names = [str(x) for x in list(range(24))]
    hour_dummies.columns = [f'is_{hour}' for hour in hour_names]

    # Adding the new columns to the original DataFrame
    df = pd.concat([df, day_dummies, month_dummies, hour_dummies], axis=1)
    df = df.drop(columns=['dateofincident', 'dateposted', 'datereported', 'day_of_week', 'month', 'hour'])
    return df

"""
Extracts features from a given text column (incident details, location, or description).
"""
def extract_text_features(df, col):
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix = vectorizer.fit_transform(df[col])
    array = matrix.toarray()

    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(array, columns=[f"{col}_{name}" for name in feature_names])

    return tfidf_df, vectorizer, feature_names

"""
Scales the columns in the DataFrame corresponding to text feature data using a StandardScaler.
"""
def scale_text_features(tfidf_feature_df):
    # Initializing a new scaler object
    scaler = StandardScaler()
    # Creating a new DataFrame that contains the scaled text data
    scaled_features = scaler.fit_transform(tfidf_feature_df)
    # Adding the scaled data to the original DataFrame
    return pd.DataFrame(scaled_features, columns=tfidf_feature_df.columns)

def train_model(df):
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    kmeans.fit(df.drop('incidenttype_cleaned', axis=1))
    labels = kmeans.labels_
    return kmeans, labels

"""
Creates the cursor and connection objects for interacting with the database.
"""
# def setup_db():
#     conn = psycopg2.connect(**db_params)
#     cur = conn.cursor()
#     cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
#     return conn, cur

def main():
    # Setting up the database connection
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')

    # conn, cur = setup_db()
    # register_vector(conn)

    df, copied_df, vectorizers = load_and_transform_data(engine)

    kmeans, labels = train_model(df)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(df.drop('incidenttype_cleaned', axis=1))

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Security Incidents Clusters')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.show()

    # Analyze clusters
    for cluster in range(N_CLUSTERS):
        print(f"Cluster {cluster}:")
        cluster_data = df[labels == cluster]
        print(cluster_data['incidenttype_cleaned'].value_counts(normalize=True))
        print("\n")

if __name__ == "__main__":
    main()