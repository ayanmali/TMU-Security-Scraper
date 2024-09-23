"""
This script is to be used when scraping a few newly added incidents to the website.
"""

import requests
import psycopg2
from psycopg2 import sql
from openai import OpenAI 
import sys 
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from postgres_params import db_params
import time
from bs4 import BeautifulSoup
from search import get_embedding

TABLE_NAME = "incidents"

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

def insert_data(cur, conn, data, client):
    # Prepares the first part of the query that declares the columns for which values are being added
    # ON CONFLICT portion ensures duplicates aren't being added in
    insert_query = sql.SQL("""
    INSERT INTO {} (page, incidentType, datePosted, dateReported, dateOfIncident, location, otherIncidentType, incidentDetails, description, locdetailsembed, locdescrembed)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (page) DO NOTHING
    """).format(sql.Identifier(TABLE_NAME))

    # Adding the data from each incident returned in the response
    for item in data:
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
        locdetails_embedding = get_embedding(client, locdetails_string)
        locdescr_embedding = get_embedding(client, locdescr_string)

        # Executing the query and passing in the values to add to the table
        cur.execute(insert_query, (
            item['page'],
            item['incidentType'],
            item['date'],
            dateReportedValue,
            item['dateOfIncident'],
            item['location'],
            otherIncidentTypeValue,
            incident_details,
            description,
            locdetails_embedding,
            locdescr_embedding
        ))

        break

    # Committing changes
    conn.commit()
    print("Added new incident data successfully.")

# Setting up the database connection
conn = psycopg2.connect(**db_params)
cur = conn.cursor()
client = OpenAI()

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
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "x-requested-with": "XMLHttpRequest"
}

print("Scraping page 1...")

url = "https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents/jcr:content/content/restwocoltwoone/c1/ressecuritystack.data.1.json"

# Getting the response and storing it
response = requests.request("GET", url, #proxies=proxies,
                            verify=False, data=payload, headers=headers)

insert_data(cur, conn, response.json()['data'], client=client)