import requests
import time
from datetime import datetime
import pandas as pd

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
START_YEAR = 2020
CURRENT_YEAR = datetime.now().year

# List to store all JSON response data
all_data = []

"""
Scrapes security incidents that have occured in the current year.
"""
def scrape_recent_incidents():
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

    # Getting data from each page of results
    for p in range(1, 7):
        try:
            page = 1
            url = f"https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents/jcr:content/content/restwocoltwoone/c1/ressecuritystack.data.{page}.json"

            # Getting the response and storing it
            response = requests.request("GET", url, data=payload, headers=headers)
            if response.status_code == 200:
                all_data.extend(response.json())
            else:
                print(f"Failed to retrieve data from {url}: Status code {response.status_code}")
            
            # Pausing to space out requests and avoid hitting limits
            time.sleep(10)

        # If the attempted page number doesn't exist, get an error
        except Exception:
            print(f"Index {p} is out of bounds")

"""
Scrapes archived security incidents from previous years.
"""
def scrape_archived_incidents():
    payload = ""

    # Defining request headers
    headers = {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "accept-language": "en-US,en;q=0.7",
        "cache-control": "no-cache",
        # "cookie": "cswebPSJSESSIONID=ZHoRIFOXqODUru79GBo4fvU0rWwB0tDY\u0021-736116267; PS_TokenSite=https://sis.torontomu.ca/psp/csprd/?cswebPSJSESSIONID; SignOnDefault=; lcsrftoken=RQDgSSv/PAc4oTtULg3iQef9+KObdwCncADvQXISYEs=",
        "pragma": "no-cache",
        "priority": "u=1, i",
        "referer": "https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents/2023/01/",
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
    for year in range(START_YEAR, CURRENT_YEAR):
        for month in range(1, 12):
            # Formatting the month string
            if month < 10:
                month = "0" + str(month)
            else:
                month = str(month)

            # Adjusting the referer header URL for the year and month
            headers['referer'] = f"https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents/{year}/{month}/"

            # Adjusting the request URL based on the year and month
            url = f"https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents/{year}/{month}/jcr:content/content/restwocoltwoone/c1/ressecuritystack.data.1.json"
            
            # Getting the response and storing it
            response = requests.request("GET", url, data=payload, headers=headers)
            if response.status_code == 200:
                # Parsing the JSON response and making sure there is at least one incident for the given month and year
                if response.json().get('totalMatches', 0) > 0:
                    all_data.extend(response.json())
                else:
                    print(f"No incidents for {month} {year}")
            else:
                print(f"Failed to retrieve data from {url}: Status code {response.status_code}")
            
            # Pausing to space out requests and avoid hitting limits
            time.sleep(10)

def get_incident_details(df):
    # Getting the 'page' columns
    pages = df['page']
    # Navigating to each individual incident's details page
    for page in pages:
        # Getting the part of each row in 'page' that contains the directory that points to the incident's details page
        directory = page.split("list-of-security-incidents")[1]
        url = f"https://www.torontomu.ca/community-safety-security/security-incidents/list-of-security-incidents{directory}/"



def main():
    # Getting and storing incident data
    scrape_recent_incidents()
    scrape_archived_incidents()

    # Creating a DataFrame to store the data
    df = pd.DataFrame(all_data)
    print(df)
    print(f"Total rows: {len(df)}")

    # Exporting the DataFrame as a .csv file
    df.to_csv(f"TMU-Security-Incidents-{START_YEAR}-{CURRENT_YEAR}.csv")

    get_incident_details(df)

if __name__ == "__main__":
    main()