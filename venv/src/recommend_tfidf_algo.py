"""
Uses incident type, date of incident, as well as location, incident details, and suspect descriptions to suggest recommendations.
"""

# For converting text features into vectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import NearestNeighbors

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

from streets import secondary, landmarks
from details_keywords import primary_keywords, secondary_keywords

TABLE_NAME = "incidents"
N_NEIGHBORS = 5

"""
Loads and preprocesses the data for training.
"""
def load_and_transform_data(engine):
    # Loading the data into a DataFrame
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY id", engine)
    # Storing a duplicate of the DataFrame for reference when recommendations are suggested
    copied_df = df.copy(deep=True)
    df = df.drop(columns=['page', 'otherincidenttype', 'detailsembed', 'locdetailsembed', 'locdescrembed', 'locationembed', 'descrembed'], axis=1)

    # For specifying the exact incident type for any incident type that is "Other"
    df = replace_other_incident_type(df)

    # For one hot encoding the locations
    df = process_locations(df)

    # For incident type
    df = process_type(df)
    # For the date/time of the incident
    df = get_dates(df)

    text_features = {}
    vectorizers = {}

    # For incident details, locations, and suspect descriptions
    for col in ('incidentdetails', 'description'):
        tfidf_df, vectorizers[col], _ = extract_text_features(df, col=col)
        text_features[col] = scale_text_features(tfidf_df)

    # Dropping features that we don't need anymore
    df = df.drop(['incidentdetails', 'description'], axis=1)

    # Concatenate all features
    result_df = pd.concat([df] + list(text_features.values()), axis=1)

    return result_df, copied_df, vectorizers

"""
Uses the keyword dictionaries to map any rows with "Other" as their incident type to an appropriate keyword (i.e. mapping it to a new category).
"""
def format_type(row, primary_keywords, secondary_keywords):
    if row['incidenttype'] == "Other":
        details_lower = row['incidentdetails'].lower()

        for key, value in primary_keywords.items():
            if key in details_lower:
                return value
            
        for key, value in secondary_keywords.items():
            if key in details_lower:
                return value

        return "Suspicious Behaviour"

    return row['incidenttype']

"""
Replaces any values with an incident type of "Other" with an appropriate replacement based on the keywords of the incident details.
"""
def replace_other_incident_type(df):
    df['incidenttype'] = df.apply(lambda row: format_type(row, primary_keywords, secondary_keywords), axis=1)
    return df

"""
Maps any location that is listed as a landmark to the closest street intersection.
"""
def format_landmarks(location):
    for key, value in landmarks.items():
        if key in location:
            return value
    return location.strip()

"""
Remove unnecessary text from location names, and format them so that the street that
runs west to east always comes first, and the street that runs north to south always comes second.
"""
def format_street_names(location):
    loc = location.replace(" East", "")
    loc = loc.replace(" West", "")
    loc = loc.replace("Laneway", "Lane")
    loc = loc.replace(" area", "")
    loc = loc.replace("Bond and", "Bond Street and")
    loc = loc.replace("Wak", "Walk")
    loc = loc.replace("O’Keefe Lane", "O'Keefe Lane")
    loc = loc.replace("Gold", "Gould")
    loc = loc.replace("the", "")
    loc = loc.strip()

    splitted = loc.split(" and ")
    if len(splitted) == 2:
        if splitted[0].strip() in secondary:
            return splitted[1].strip() + " and " + splitted[0].strip()
        return splitted[0].strip() + " and " + splitted[1].strip()
    return loc

"""
Formatting and one hot encoding the locations.
"""
def process_locations(df):
    df['location'] = df['location'].apply(format_landmarks)
    df['location'] = df['location'].apply(format_street_names)

    df[['Primary Street', 'Secondary Street']] = df['location'].str.split(' and ', expand=True)

    primary_st_dummies = pd.get_dummies(df['Primary Street'], prefix='Primary_Street', dtype=int)
    secondary_st_dummies = pd.get_dummies(df['Secondary Street'], prefix='Secondary_Street', dtype=int)

    df = pd.concat([df, primary_st_dummies, secondary_st_dummies], axis=1)
    df = df.drop(columns=['location', 'Primary Street', 'Secondary Street'], axis=1)

    return df

"""
One hot encodes the incident type.
"""
def process_type(df):
    # Removing any extra unnecessary phrases from the incident type
    df['incidenttype_cleaned'] = df['incidenttype'].replace({": Suspect Arrested" : "", ": Update" : ""}, regex=True)
    df = pd.concat([df, pd.get_dummies(df['incidenttype_cleaned'], prefix='incidenttype', dtype=int)], axis=1)
    df = df.drop(columns=['incidenttype', 'incidenttype_cleaned'], axis=1)

    return df

"""
Extracts the month, day of week, and hour of each incident's date.
"""
def get_dates(df):
    # Extracting the day of the week, month, and hour from the datetime column
    df['day_of_week'] = df['dateofincident'].dt.dayofweek
    df['month'] = df['dateofincident'].dt.month
    df['hour'] = df['dateofincident'].dt.hour

    # One hot encoding the new date/time columns
    # day_dummies = pd.get_dummies(df['day_of_week'], prefix='day', dtype=int)
    # month_dummies = pd.get_dummies(df['month'], prefix='month', dtype=int)
    # hour_dummies = pd.get_dummies(df['hour'], prefix='hour', dtype=int)

    # # Renaming the columns to actual day names
    # day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # day_dummies.columns = [f'is_{day}' for day in day_names]
    # month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    # month_dummies.columns = [f'is_{month}' for month in month_names]
    # hour_names = [str(x) for x in list(range(24))]
    # hour_dummies.columns = [f'is_{hour}' for hour in hour_names]

    # # Adding the new columns to the original DataFrame
    # df = pd.concat([df, day_dummies, month_dummies, hour_dummies], axis=1)
    df = df.drop(columns=['dateofincident', 'dateposted', 'datereported'])
    return df

"""
Extracts features from a given text column (incident details, location, or description).
"""
def extract_text_features(df, col):
    # Using the TF-IDF of the words for the given text feature to create a matrix that numerically represents the text data
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

"""
Suggests potentially related incidents given an incident ID (from the database).
"""
def get_recommendations(id, df, model, n_recommendations=5):
    # Find the index of the given id in the DataFrame
    try:
        idx = df.index[df['id'] == id].tolist()[0]
    except IndexError:
        print(f"ID {id} not found in the dataset.")
        return None

    # Get the feature vector for the given id
    incident_vector = df.iloc[idx].values.reshape(1, -1)

    # Find the nearest neighbors
    distances, indices = model.kneighbors(incident_vector, n_neighbors=n_recommendations+1)

    # The first result will be the incident itself, so we exclude it
    similar_incidents = df.iloc[indices[0][1:]]

    return similar_incidents

"""
Trains a K-nearest neighbors model on the dataset and returns the trained model.
"""
def train_model(df, n_neighbors):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(df)
    return knn

def main():
    # Setting up the database connection
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')

    # Preprocessing the data
    df, copied_df, vectorizers = load_and_transform_data(engine)

    # Training the model
    knn = train_model(df, N_NEIGHBORS)

    # The ID in the database of the incident to check
    id_to_check = 420
    
    # Gets a list of IDs of the recommended incidents
    recommendations = get_recommendations(id_to_check, df, knn)

    if recommendations is not None:
        print(f"Recommendations for incident with ID {id_to_check}:")
        print(recommendations[['id']])

if __name__ == "__main__":
    main()