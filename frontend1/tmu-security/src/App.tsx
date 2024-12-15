import React from 'react';
import { useState } from 'react';
import moment from 'moment';
import IncidentList from './components/IncidentList'
import Navbar from './components/Navbar'
import SearchResults from './components/SearchTab'
import RecommendResults from './components/RecommendTab'
// import Analytics from './components/Analytics'
import TimeSeriesForecastChart from './components/Forecast'
import './App.css'
// import IncidentForecastVisualization from './components/Visuals';

export const API_BASE_URL = process.env.REACT_APP_API_BASE_URL
export const ITEMS_PER_PAGE: number = 20;
export const AUTH_TOKEN = process.env.REACT_APP_AUTH_TOKEN;
export const HOST = process.env.REACT_APP_HOST;

// Error handling
if(!API_BASE_URL) throw new Error("API_BASE_URL is undefined");
if(!AUTH_TOKEN) throw new Error("AUTH_TOKEN is undefined");
if(!HOST) throw new Error("HOST is undefined");

const headers = new Headers();
headers.append('Authorization', AUTH_TOKEN)
headers.append('Access-Control-Allow-Origin', HOST);
headers.append('Access-Control-Allow-Credentials', 'true');

let currentPage = 1;
let totalPages = 1;

// Defining an interface for an Incident record
export interface Incident {
  id: number;
  incident_type?: string;
  incidenttype?: string,
  location: string;
  date_of_incident?: string;
  dateofincident?: string;
  date_posted?: string;
  dateposted?: string,
  incident_description?: string;
  incidentdetails?: string;
  suspect_description?: string;
  description?: string;
  page_url?: string;
  page?: string;
}

// interface APIResponse {
//   total: number;
//   incidents: Incident[];
// }

// import logo from './logo.svg';
// import './App.css';

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.tsx</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// }

/*
Generates the component for the card that provides an overview of the incident data.
*/
export function createIncidentCard(incident: Incident) {
  return (
    <div className="incident-card">
      <h3>{incident.incident_type || incident.incidenttype}</h3>
      <p><strong>Location: </strong>{incident.location}</p>
      <p><strong>Incident Date: </strong>{moment.utc(incident.date_of_incident || incident.dateofincident).format('MMMM D, YYYY')}</p>
      <p><strong>Time of Occurrence: </strong>{moment.utc(incident.date_of_incident || incident.dateofincident).format('hh:mm A')}</p>
      <p><strong>Incident Description: </strong>{incident.incident_description || incident.incidentdetails}</p>
      <p><strong>Suspect Description: </strong>{incident.suspect_description || incident.description}</p>
      {/* {incident.incident_description || incident.page && (
        <p><strong>Details: {incident.incident_description?.includes('male') ? "Male" : "Female"}</strong></p>)} */}

      {incident.page_url || incident.page ? (
        <p><a href={incident.page_url || incident.page} target="_blank" rel="noopener noreferrer">View Details</a></p>
      ) : ('')}
    </div>
  );
}


// async function fetchAllIncidents() {
//   const offset = (currentPage - 1) * ITEMS_PER_PAGE;
//   const url = `${API_BASE_URL}/getincidents/?offset=${offset}&limit=${ITEMS_PER_PAGE}`;

//   // Making the API request to the server
//   try {
//     const response = await fetch(url, {
//       method: 'GET',
//       headers: headers
//   });
//   if (!response.ok) throw new Error('Failed to fetch incidents');
  
//   const data = await response.json();
//   totalPages = data.pagination.total_pages;
//   // Update pagination controls here

//   } catch(err) {
//       //if (err instanceof Error)
//   } finally {

//   }

// }
function App() {
  const [currentTab, setCurrentTab] = useState('all');

  return (
    <div className="app">
      <div className="top-screen">
        <h1>TMU Security Incidents Dashboard</h1>
        {/* <div className="tabs">
            <button className="tab active" data-tab="all">All Incidents</button>
            <button className="tab" data-tab="search">Search</button>
            <button className="tab" data-tab="recommendations">Recommendations</button>
        </div> */}
        {/* <img>https://cdn-icons-png.flaticon.com/512/25/25231.png</img> */}
        
        <a href='https://www.github.com/ayanmali'><img className='github-icon' src='https://cdn-icons-png.flaticon.com/512/25/25231.png'
        width='40' height='40'></img></a>
      </div>
        <Navbar onTabChange={setCurrentTab} />
        {currentTab === "all" && <IncidentList />}
        {currentTab === "search" && <SearchResults />}
        {currentTab === "recommend" && <RecommendResults />}
        {currentTab === "analytics" && <TimeSeriesForecastChart />}
    </div>
  );
}

export default App;