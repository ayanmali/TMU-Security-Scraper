# For HTTP requests
import requests

# Proxy to mask IP address when making requests
import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
# from proxies import proxies

# For integrating the database
import psycopg2
from psycopg2 import sql
from postgres_params import db_params

# For parsing HTML from the incident details page
from bs4 import BeautifulSoup

from openai import OpenAI

from search import get_embedding, TABLE_NAME

# For pausing in between requests to avoid rate limits
import time

# For formatting dates
from datetime import datetime

# For storing and manipulating data
# import pandas as pd

# To disable warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
START_YEAR = 2018
CURRENT_YEAR = datetime.now().year
MONTHS = ["Jan", "Feb", "March", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]

def create_table_if_not_exists(cur, conn):
    # Note: make sure that 'page' is a unique column, and make sure client_encoding is the same as server_encoding
    create_table_query = sql.SQL("""
    CREATE TABLE IF NOT EXISTS Incidents (
        id SERIAL PRIMARY KEY,
        page TEXT,
        incidentType TEXT,
        datePosted TIMESTAMP WITH TIME ZONE,
        dateReported TIMESTAMP WITH TIME ZONE,
        dateOfIncident TIMESTAMP WITH TIME ZONE,
        location TEXT,
        otherIncidentType TEXT,
        incidentDetails TEXT,
        description TEXT,
    )
    """)
    
    cur.execute(create_table_query)
    conn.commit()
    print("Table 'Incidents' created or already exists.")

def process_response(cur, conn, client, response, previous_response_data, month, year, p):
    # If the response is valid
    if response.status_code == 200:
        data = response.json().get('data', [])
        # If the next page contains no new data
        if response.json().get('totalMatches', 0) <= 0 or not data or data == previous_response_data:
            print(f"No incidents for {MONTHS[month-1]} {year} for page {p}")
            return False
        # The next page does contain new data, which is stored in the database
        else:
            print(f"Storing data from page {p}...")
            insert_data(cur, conn, data, client)
            return True
    # Response is not valid
    return None

"""
Scrapes security incidents that have occured in the current year to date.
"""
def scrape_recent_incidents(cur, conn, client):
    payload = ""

    headers = {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "accept-language": "en-US,en;q=0.7",
        "cache-control": "no-cache",
        # "cookie": "cswebPSJSESSIONID=ZHoRIFOXqODUru79GBo4fvU0rWwB0tDY\u0021-736116267; PS_TokenSite=https://sis.torontomu.ca/psp/csprd/?cswebPSJSESSIONID; SignOnDefault=; lcsrftoken=RQDgSSv/PAc4oTtULg3iQef9+KObdwCncADvQXISYEs=",
        "pragma": "no-cache",
        "priority": "u=1, i",
        "referer": "https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents/",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"99\", \"Brave\";v=\"128\", \"Chromium\";v=\"128\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "sec-gpc": "1",
        "user-agent": USER_AGENT,
        "x-requested-with": "XMLHttpRequest"
    }

    print(f"Getting incidents for {CURRENT_YEAR}...")

    # Checking each page of results for data until there's no more left
    p = 1
    while True:
        print(f"Scraping page {p}...")

        url = f"https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents/jcr:content/content/restwocoltwoone/c1/ressecuritystack.data.{p}.json"

        # Getting the response and storing it
        response = requests.request("GET", url, #proxies=proxies,
                                    verify=False, data=payload, headers=headers)
        
        if response.status_code == 200:
            if response.json().get('data') != []:
                print(f"Storing data from page {p}...")
                # Storing the data associated with the 'data' key of the response object
                insert_data(cur, conn, response.json()['data'], client=client)
            else:
                # No more pages left to scrape
                print(f"Endpoint for page {p} contains no data")
                break

        else:
            print(f"Failed to retrieve data from {url}: Status code {response.status_code}")
            break
        
        # Incrementing the counter to get the next page of results
        p += 1
        
        # Pausing to space out requests and avoid hitting limits
        time.sleep(10)
    
    print("Completed scraping recent incidents.")

"""
Scrapes archived security incidents from previous years.
"""
def scrape_archived_incidents(cur, conn, client):
    # Creating the session object
    s = requests.Session()
    s.verify = False

    payload = ""

    # Defining request headers
    headers = {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.7",
        "cache-control": "no-cache",
        # "cookie": "cswebPSJSESSIONID=ZHoRIFOXqODUru79GBo4fvU0rWwB0tDY\u0021-736116267; PS_TokenSite=https://sis.torontomu.ca/psp/csprd/?cswebPSJSESSIONID; SignOnDefault=; lcsrftoken=RQDgSSv/PAc4oTtULg3iQef9+KObdwCncADvQXISYEs=",
        "pragma": "no-cache",
        "priority": "u=1, i",
        # This header gets updated for each year and month being checked
        "referer": "",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"99\", \"Brave\";v=\"128\", \"Chromium\";v=\"128\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "sec-gpc": "1",
        "user-agent": USER_AGENT,
        "x-requested-with": "XMLHttpRequest"
    }

    # Setting the session headers
    s.headers.update(headers)
    
    # Getting data for each month from 2020-2023
    for year in range(START_YEAR, CURRENT_YEAR):
        for month in range(1, 13):
            print(f"Getting incidents for {MONTHS[month-1]} {year}...")
            # Formatting the month string
            str_month = f"0{month}" if month < 10 else str(month)

            # Adjusting the referer header URL for the year and month
            s.headers.update({
                "referer": f"https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents/{year}/{str_month}/"
            })

            p = 1
            previous_response_data = None
            while True:
                print(f"Scraping page {p}...")
                # Trying two URLs, since some months use one endpoint and some use the other
                main_url = f"https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents/{year}/{str_month}/jcr:content/content/restwocoltwoone/c1/ressecuritystack.data.{p}.json"
                alt_url = f"https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents/{year}/{str_month}/jcr:content/content/restwocoltwoone_copy/c1/ressecuritystack.data.{p}.json"
                
                for url in [main_url, alt_url]:
                    try:
                        # Getting the response and storing it
                        response = s.get(url, data=payload)
                        #response = requests.request("GET", url, #proxies=proxies, 
                        #                            verify=False, data=payload, headers=headers)

                        # Evaluating whether the response for the current month, year, and page contains new data (true), no new data (false), or is invalid (None)
                        result = process_response(cur, conn, client, response, previous_response_data, month, year, p)
                        # If the response was valid and went through properly, then there's no need to check the other URL
                        if result is not None:
                            break
                    
                    # TooManyRedirects exception is thrown when the main URL receives a request instead of the alt URL
                    except requests.exceptions.TooManyRedirects:
                        print(f"Too many redirects for URL: {url}")
                        if url == main_url:
                            print("Trying alternative URL...")
                            continue
                        else:
                            print("Both URLs failed due to too many redirects.")
                            result = None
                            break

                # If one of the request URLs went through successfully and there is no new data left to parse, then move on to the next month
                if result is False:
                    # No more incidents for this month
                    break
                # If both URLs fail
                elif result is None:
                    # Exit the function if both URLs fail
                    print(f"Couldn't retrieve data from either URL for {MONTHS[month-1]} {year}, page {p}")
                    return
                
                # Updating the previous response data
                previous_response_data = response.json()['data']

                # Incrementing the counter to get data from the next page
                p += 1

                # Spacing out requests to avoid hitting rate limits
                time.sleep(10)

    print(f"Completed scraping archived incidents from {START_YEAR} to {CURRENT_YEAR-1}")

def insert_data(cur, conn, data, client):
    # Prepares the first part of the query that declares the columns for which values are being added
    # ON CONFLICT portion ensures duplicates aren't being added in
    insert_query = sql.SQL("""
    INSERT INTO {} (page, incidentType, datePosted, dateReported, dateOfIncident, location, otherIncidentType, incidentDetails, description, detailsembed, locdetailsembed, locdescrembed)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (page) DO NOTHING
    """).format(sql.Identifier(TABLE_NAME))

    # Adding the data from each incident returned in the response
    for item in data:
        # Cleaning the page URL
        page = item['page'].replace("/content/ryerson/", "https://www.torontomu.ca/")

        # Adding other incident type if it was present in the response, NA otherwise   
        if "otherIncidentType" in item.keys():
            otherIncidentTypeValue = item['otherIncidentType']
        else:
            otherIncidentTypeValue = "NA"

        if "dateReported" in item.keys():
            dateReportedValue = item['dateReported']
        else:
            dateReportedValue = None

        # Getting incident and description information
        incident_details, description = get_details(item['page'])
        time.sleep(7)

        # Creating a string that contains both the location and details of the incident
        locdetails_string = item['location'] + " " + incident_details
        locdescr_string = item['location'] + " " + description

        # Adding vector embeddings of the location + incident details and location + suspect description if we're scraping new incidents from the current year
        details_embed = get_embedding(client, incident_details, dims=256)
        locdetails_embedding = get_embedding(client, locdetails_string)
        locdescr_embedding = get_embedding(client, locdescr_string)

        # Executing the query and passing in the values to add to the table
        cur.execute(insert_query, (
            page,
            item['incidentType'],
            item['date'],
            dateReportedValue,
            item['dateOfIncident'],
            item['location'],
            otherIncidentTypeValue,
            incident_details,
            description,
            details_embed,
            locdetails_embedding,
            locdescr_embedding
        ))

    # Committing changes
    conn.commit()

"""
Scrapes incident details and the description for a given security incident.
"""
def get_details(page):
    # Getting the part of each row in 'page' that contains the directory that points to the incident's details page
    directory = page.split("list-of-security-incidents")[1]
    # Getting the URL of the incident details page
    url = f"https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents{directory}/"

    # Getting the HTML content of the page
    html = get_html_content(url)
    if html:
        print(f"HTML content for {page} retrieved successfully.")
    else:
        print("Failed to retrieve HTML content.")

    # Creating the soup object to extract data from the HTML
    soup = BeautifulSoup(html, 'html.parser')
    return get_incident_details(soup), get_description_details(soup)

def get_incident_details(soup):
    # Find the div with class "incidentdetails"
    incident_div = soup.find('div', class_='incidentdetails')

    # If the div with the specified class has been found
    if incident_div:
        # Find the paragraph within this div
        paragraph = incident_div.find('p')
        
        if paragraph:
            return paragraph.text.strip()
        else:
            return "Paragraph not found within the incident details div."
    else:
        return "Incident details div not found."

def get_description_details(soup):
    # Find the div with class "description"
    description_div = soup.find('div', class_='description')

    # If the div with the specified class has been found
    if description_div:
        # Find the paragraphs within this div
        paragraphs = description_div.find_all('p')

        if paragraphs:
            # Get a string containing each paragraph of the description
            return " ".join([p.text.strip() for p in paragraphs])
        else:
            return "No paragraphs found within the description div."

    else:
        return "Description div not found."

def get_html_content(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url, #proxies=proxies, 
                                verify=False)
        
        # Raise an exception for bad status codes
        response.raise_for_status()
        
        # Return the HTML content
        return response.text
    
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def main():
    # Setting up the database connection
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    client = OpenAI()
    # Creating the table if it doesn't already exist
    create_table_if_not_exists(cur, conn)

    # Getting and storing incident data
    scrape_recent_incidents(cur, conn, client)
    scrape_archived_incidents(cur, conn)

    # Closing the database connection
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()