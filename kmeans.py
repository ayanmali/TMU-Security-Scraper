import numpy as np

# For converting text features into vectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, scale
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score

# For visualizations
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# For manipulating and transforming the data
import pandas as pd
# For the database connection
from sqlalchemy import create_engine

# Database credentials
import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import user, password, host, port, dbname #, db_params

from streets import secondary, landmarks
from details_keywords import primary_keywords, secondary_keywords

# Declaring constants
TABLE_NAME = "incidents"
# Obtained from silhouette analysis and the elbow plot
N_CLUSTERS = 3
# Testing a range of possible cluster counts
RANGE_N_CLUSTERS = range(2, 21)
# FEATURES_TO_ANALYZE = ['id', 'incidenttype_cleaned', 'location']

"""
Loads and preprocesses the data for training.
"""
def load_and_transform_data(engine):
    # Loading the data into a DataFrame
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY id", engine)
    # Storing a duplicate of the DataFrame for reference
    copied_df = df.copy(deep=True)
    # Dropping unneccessary columns
    df = df.drop(columns=['page', 'otherincidenttype', 'detailsembed', 'locdetailsembed', 'locdescrembed', 'locationembed', 'descrembed'], axis=1)

    # For specifying the exact incident type for any incident type that is "Other"
    df = replace_other_incident_type(df)
    # For one hot encoding each street name of the intersection
    df = process_locations(df)
    # For one hot encoding incident types
    df = process_types(df)
    # For the date/time of the incident
    df = get_dates(df)

    # Dropping text columns since they are not needed for clustering
    # df = df.drop(['incidentdetails', 'description'], axis=1)
    # text_features = {}
    # vectorizers = {}

    # # For incident details, locations, and suspect descriptions
    # for col in ('incidentdetails', 'description'):
    #     tfidf_df, vectorizers[col], _ = extract_text_features(df, col=col)
    #     text_features[col] = scale_text_features(tfidf_df)

    # # Dropping features that we don't need anymore
    # df = df.drop(['incidentdetails', 'description'], axis=1)

    # # Concatenate all features
    # result_df = pd.concat([df] + list(text_features.values()), axis=1)

    # return result_df, copied_df, vectorizers

    # vectorizers = {}
    
    return df, copied_df

"""
Uses the keyword dictionaries to map any rows with "Other" as their incident type to an appropriate keyword (i.e. mapping it to a new category).
"""
def format_type(row, primary_keywords, secondary_keywords):
    if row['incidenttype'] == "Other":
        details_lower = row['incidentdetails'].lower()

        # More than one possible keyword can be used in an incident description, so the primary ones are to be checked first, and if there are no matches, then the secondary ones
        # This is to avoid some cases being misclassified - ex. a sexual assault case labeled as "Assault" because it has the string "assault" in the description.
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
    df = df.drop(columns=['Primary Street', 'Secondary Street'], axis=1)

    return df

"""
One hot encodes the incident type.
"""
def process_types(df):
    # Removing any extra unnecessary phrases from the incident type
    df['incidenttype_cleaned'] = df['incidenttype'].replace({": Suspect Arrested" : "", ": Update" : ""}, regex=True)
    df = pd.concat([df, pd.get_dummies(df['incidenttype_cleaned'], prefix='incidenttype', dtype=int)], axis=1)
    df = df.drop(columns=['incidenttype'], axis=1)

    return df

"""
Extracts the month, day of week, and hour of each incident's date and one hot encodes each one.
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
    df = df.drop(columns=['dateposted', 'datereported', 'dateofincident'])
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
Trains a K-means clustering model on the dataset and returns the trained model as well as the labels for each cluster.
"""
def train_model(X):
    random_state = 42
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=random_state)
    kmeans.fit(X)
    labels = kmeans.labels_
    return kmeans, labels

"""
Creates a plot showing the change in variation across a given range of K values to use for clustering.
"""
def elbow_plot(X):
    # Reducing the data down to two dimensions so it can be visualized and clustered more easily
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced = tsne.fit_transform(X)

    # The y-values for the plot
    inertias = []

    # Obtaining the inertia value for each possible K value
    for n_clusters in RANGE_N_CLUSTERS:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_reduced)
        inertias.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(RANGE_N_CLUSTERS, inertias, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.show()

"""
For a given number of clusters n in a particular range, a silhouette plot is created for each of the n clusters as well as a visualization of the clustered data.
"""
def silhouette_analysis(X):
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced = tsne.fit_transform(X)

    for n_clusters in RANGE_N_CLUSTERS:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10, n_init='auto')
        cluster_labels = clusterer.fit_predict(X_reduced)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X_reduced[:, 0], X_reduced[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()

"""
Creates a scatter plot showing the clustered data points and provides a breakdown of the data in each cluster.
"""
def analyze_clusters(X, df, labels):
    # Reduces dimensionality of the dataset down to just two features so the data and the clusters can be easily visualized
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(X)

    # Creates a scatter plot of the two features of the new reduced dataset
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Security Incidents Clusters')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.show()

    # Show a breakdown of the number of different values for each feature in each cluster
    for cluster in range(N_CLUSTERS):
        print(f"Cluster {cluster}:")
        cluster_data = df[labels == cluster]

        print(cluster_data['id'].value_counts(normalize=False))
        print("\n")
        print(cluster_data['location'].value_counts(normalize=False))
        print("\n")
        print(cluster_data['incidenttype_cleaned'].value_counts(normalize=False))
        print("\n")
        print(cluster_data['day_of_week'].value_counts(normalize=False))
        print("\n")
        print(cluster_data['month'].value_counts(normalize=False))
        print("\n")
        print(cluster_data['hour'].value_counts(normalize=False))
        print("\n")
        print(cluster_data['incidentdetails'].value_counts(normalize=False))
        print("\n")
        print(cluster_data['description'].value_counts(normalize=False))
        print("\n")

def main():
    # Setting up the database connection
    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')

    # Preprocessing the data
    df, copied_df = load_and_transform_data(engine)

    # When preprocessing the data, certain columns are left out so that they can be indexed again for analysis; when using the DataFrame for clustering, these features are to be dropped as they are not in a usable format
    X = df.drop(columns=['id', 'incidenttype_cleaned', 'location', 'incidentdetails', 'description'], axis=1)
    # X = StandardScaler().fit_transform(X)
    X = scale(X, axis=1)
    # Displays the silhouette plots
    # silhouette_analysis(X)

    # Using the elbow method to find the point where the variation drops off the most (the "elbow" of the plot)
    elbow_plot(X)

    # Training the model
    _, labels = train_model(X)

    # Analyzing characteristics of each cluster
    analyze_clusters(X, df, labels)

    # predict(kmeans)

if __name__ == "__main__":
    main()