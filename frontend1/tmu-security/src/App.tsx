import React from 'react';
import moment from 'moment';
import IncidentList from './components/IncidentList'
import './App.css'

const API_BASE_URL = 'http://127.0.0.1:8000/app';
const ITEMS_PER_PAGE: number = 20;
const AUTH_TOKEN: string = "Token a3030df88e83c018e5e9ab64dd27e6da63edac21"
const HOST: string = "http://127.0.0.1:5500"
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

interface APIResponse {
  total: number;
  incidents: Incident[];
}

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
      <p><strong>Incident Description: </strong>{incident.incident_description || incident.incidentdetails}</p>
      <p><strong>Suspect Description: </strong>{incident.suspect_description || incident.description}</p>
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
  return (
    <div className="app">
        <h1>TMU Security Incidents Dashboard</h1>
        <IncidentList />
    </div>
  );
}

export default App;