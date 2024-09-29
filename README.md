# TMU-Security-Incidents
Applying machine learning to TMU security incident data to help improve campus security.

## Part 1. Scraping

The first step was loading all the security data into a database. Using the API from the security incidents page, a list of all security incidents from 2018 to present were scraped and added to a PostgreSQL database. For each security incident, the incident's details and suspect descriptions were scraped via HTML parsing.

## Part 2. Search

To make it easier to search for incidents, a search system is implemented using vector embeddings. For each incident, a string containing the location as well as the incident details is created. This string is then used to generate a vector embedding to represent this body of text in a high-dimensional space (in this case, 384 dimensions - 256 for the details, 128 for the location). Once these embeddings are generated for every incident, generating matches to the given query is as easy as generating a vector embedding for the query text, then getting the top N rows that have the highest cosine similarity with respect to the query text embedding.

Note that an alternative search system is also built out that uses the location and the suspect description for the embeddings, allowing users to search for a suspect description, and optionally, a location as well if desired.

## Part 3. Recommendations

Two recommendation algorithms are built here to suggest potentially related incidents to investigators. The first algorithm uses the incident date and incident type, along with location, incident details, and suspect descriptions each represented as vector embeddings, then reduced their dimensionality using PCA.
The second algorithm uses the incident date and type like before, but this algorithm uses TF-IDF (term frequency-inverse document frequency) for the three text columns instead of vector embeddings. TF-IDF helps to evaluate the importance of various words in a sentence, which makes it useful for finding incidents that may be related to another.

NOTE: vector embeddings are stored in the table using PGVector.

## TODO
- Search feature ✅
    - Can use: OpenAI/Voyage Embeddings API, PGVector for search
    1. Generate embeddings✅
    2. Store embeddings in DB with PGVector✅
    3. Query from Python script✅

- Similar Incident Recommendation System ✅
    - Features such as incident type, suspect descriptions, date and time, incident details (bag of words, tf-idf), and location data (latitude, longitude, proximity to landmarks) ✅
    - Matrix factorization ⭕
    - Time feature relevance - incidents closer in date to the given incident are more relevant ⭕
    - Use streets.py for adjusting formatting of location (one hot encoding?) ✅

- Clustering
    - K-Means to identify patterns in incident characteristics ✅
        - Figure out how to use silhouette analysis to determine the number of clusters to use ✅
    - Try out different kinds of algorithms (DBSCAN, HDBSCAN) ✅
    - Format the analysis better (ex. charts and graphs) ⭕
    - Use streets.py for adjusting formatting of location (one hot encoding?) ✅

- Incident Type Prediction (Classifier Model) ⭕
    - Predict incident type given location, description, time of year, day of week

- Predict incident location (Classifier) ⭕
    - given the type, suspect description, day of week, time of year

- Time Series Forecasting
    - likelihood of number of incidents occurring during specific times or days
    - likelihood of types of incidents occuring during specific times or days
    - number of incidents expected to occur in future time periods (ARIMA, Prophet, SARIMA to account for trends and seasonality)
    - Predict incident hotspots based on recent activity

- Train an autoencoder or One Class SVM to identify incidents that deviate from typical patterns
    - Look for anomalies in incident frequencies

- Location Based Risk Assessment
    - Spatial Clustering to identify areas with high concentrations of incidents/high risk areas
    - Recommendation Algorithm to suggest patrol routes/areas to focus on based on historical data

- Topic Modelling Using Latent Dirichlet Allocation to find underlying themes in incident descriptions or suspect descriptions

"""
