# For HTTP requests
import requests

# Proxy to mask IP address when making requests
import sys
sys.path.insert(1, 'c:/Users/ayan_/Desktop/Desktop/Coding/Cursor Workspace/Scrapers')
from proxies import proxies

# For integrating the database
import psycopg2
from psycopg2 import sql
from postgres_params import db_params

# For parsing HTML from the incident details page
from bs4 import BeautifulSoup

# For pausing in between requests to avoid rate limits
import time

# For formatting dates
from datetime import datetime

# For storing and manipulating data
# import pandas as pd

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
START_YEAR = 2020
CURRENT_YEAR = datetime.now().year
MONTHS = ["Jan", "Feb", "March", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]

"""
Scrapes security incidents that have occured in the current year to date.
"""
def scrape_recent_incidents(cur, conn):
    payload = ""

    headers = {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "accept-language": "en-US,en;q=0.7",
        "cache-control": "no-cache",
        # "cookie": "cswebPSJSESSIONID=ZHoRIFOXqODUru79GBo4fvU0rWwB0tDY\u0021-736116267; PS_TokenSite=https://sis.torontomu.ca/psp/csprd/?cswebPSJSESSIONID; SignOnDefault=; lcsrftoken=RQDgSSv/PAc4oTtULg3iQef9+KObdwCncADvQXISYEs=",
        "pragma": "no-cache",
        "priority": "u=1, i",
        "referer": "https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents/",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"99\", \"Brave\";v=\"127\", \"Chromium\";v=\"127\"",
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
        #page = 1
        url = f"https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents/jcr:content/content/restwocoltwoone/c1/ressecuritystack.data.{p}.json"

        # Getting the response and storing it
        response = requests.request("GET", url, proxies=proxies, verify=False, data=payload, headers=headers)
        if response.status_code == 200:
            if response.json().get('data') != []:
                print(f"Storing data from page {p}...")
                # Storing the data associated with the 'data' key of the response object
                # all_data.extend(response.json()['data'])
                # Inserting the response data into the table
                insert_data(cur, conn, response.json()['data'])
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
    
    # return all_data

"""
Scrapes archived security incidents from previous years.
"""
def scrape_archived_incidents(cur, conn):
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
        "sec-ch-ua": "\"Not)A;Brand\";v=\"99\", \"Brave\";v=\"127\", \"Chromium\";v=\"127\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "sec-gpc": "1",
        "user-agent": USER_AGENT,
        "x-requested-with": "XMLHttpRequest"
    }
    
    # Getting data for each month from 2020-2023
    for year in range(CURRENT_YEAR, CURRENT_YEAR+1):
        for month in range(1, 13):
            print(f"Getting incidents for {MONTHS[month-1]} {year}...")
            # Formatting the month string
            if month < 10:
                str_month = "0" + str(month)
            else:
                str_month = str(month)

            # Adjusting the referer header URL for the year and month
            headers['referer'] = f"https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents/{year}/{str_month}/"

            p = 1
            while True:
                print(f"Scraping page {p}...")
                # Adjusting the request URL based on the year and month
                # url = f"https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents/{year}/{str_month}/jcr:content/content/restwocoltwoone/c1/ressecuritystack.data.{p}.json"
                url = f"https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents/{year}/{str_month}/jcr:content/content/restwocoltwoone_copy/c1/ressecuritystack.data.{p}.json"
                # Getting the response and storing it
                response = requests.request("GET", url, proxies=proxies, verify=False, data=payload, headers=headers)
                if response.status_code == 200:
                    # Parsing the JSON response and checking to see if there are no valid results, in which case the program proceeds to the next month. Otherwise store the data and check the next page of results
                    if response.json().get('totalMatches') <= 0 or response.json().get('data') == []:
                        print(f"No incidents for {MONTHS[month-1]} {year} for page {p}")
                        break
                    else:
                        print(f"Storing data from page {p}...")
                        # all_data.extend(response.json()['data'])
                        # Inserting response data into the table
                        insert_data(cur, conn, response.json()['data'])
                else:
                    print(f"Failed to retrieve data from {url}: Status code {response.status_code}")
                    break
                
                # Incrementing the counter to check the next page
                p += 1
                # Pausing to space out requests and avoid hitting limits
                time.sleep(10)

    # return all_data

def insert_data(cur, conn, data):
    # Prepares the first part of the query that declares the columns for which values are being added
    insert_query = sql.SQL("""
    INSERT INTO Incidents (page, incidentType, datePosted, dateReported, dateOfIncident, location, otherIncidentType, incidentDetails, description)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """)

    # Adding the data from each incident returned in the response
    for item in data:
        # Adding other incident type if it was present in the response, N/A otherwise   
        if "otherIncidentType" in item.keys():
            otherIncidentTypeValue = item['otherIncidentType']
        else:
            otherIncidentTypeValue = "N/A"

        # Getting incident and description information
        incident_details, description = get_details(data['page'])

        # Executing the query and passing in the values to add to the table
        cur.execute(insert_query, (
            item['page'],
            item['incidentType'],
            item['date'],
            item['dateReported'],
            item['dateOfIncident'],
            item['location'],
            otherIncidentTypeValue,
            incident_details,
            description
        ))
    
    # Committing changes
    conn.commit()

"""
Scrapes incident details and the description for each security incident.
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

# def get_details(df):
#     # Getting the 'page' columns
#     pages = df['page']
#     incident_details = []
#     description_details = []

#     # Navigating to each individual incident's details page
#     for idx, page in pages.items():
#         # Getting the part of each row in 'page' that contains the directory that points to the incident's details page
#         directory = page.split("list-of-security-incidents")[1]
#         # Getting the URL of the incident details page
#         url = f"https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents{directory}/"

#         # Getting the HTML content of the page
#         html = get_html_content(url)
#         if html:
#             print(f"HTML content for page {idx} retrieved successfully.")
#         else:
#             print("Failed to retrieve HTML content.")

#         # Creating the soup object to extract data from the HTML
#         soup = BeautifulSoup(html, 'html.parser')

#         incident_details.append(get_incident_details(soup))

#         description_details.append(get_description_details(soup))

#     return incident_details, description_details

def get_incident_details(soup):
    # Find the div with class "incidentdetails"
    incident_div = soup.find('div', class_='incidentdetails')

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
        response = requests.get(url, proxies=proxies, verify=False)
        
        # Raise an exception for bad status codes
        response.raise_for_status()
        
        # Return the HTML content
        return response.text
    
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def main():
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()

    # List to store all JSON response data
    # all_data = []

    # Getting and storing incident data
    scrape_recent_incidents(cur, conn)
    scrape_archived_incidents(cur, conn)

    # Creating a DataFrame to store the data
    # df = pd.DataFrame(all_data)
    # print(f"DataFrame for incidents from {CURRENT_YEAR}")
    # print(df)
    # print(f"Total rows: {len(df)}")

    # Adding two extra columns for incident and description details
    # df["Incident Details"], df["Description Details"] = get_details(df)
    # print(f"Final DataFrame for incidents from {CURRENT_YEAR} (including incident and description details)")
    # print(df)

    # Exporting the DataFrame as a .csv file
    #df.to_csv(f"TMU-Security-Incidents-{START_YEAR}-{CURRENT_YEAR}.csv")
    # df.to_csv(f"TMU-Security-Incidents-{CURRENT_YEAR}.csv")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()