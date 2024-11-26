import * as moment from 'moment';
const API_BASE_URL = 'http://127.0.0.1:8000/app'; // Update this with your API URL
const ITEMS_PER_PAGE = 20;
const AUTH_TOKEN = "Token 3b2d5ac00aa108d2ab58e68f14886d28db859680";
let currentPage = 1;
let totalPages = 1;
// interface SearchResponse {
//     count: number;
//     results: Incident[];
// }
function switchTab(tabName) {
    // Hide all tabs
    document.getElementById('all-tab')?.style.setProperty('display', 'none');
    document.getElementById('search-tab')?.style.setProperty('display', 'none');
    document.getElementById('recommendations-tab')?.style.setProperty('display', 'none');
    // Show selected tab
    document.getElementById(`${tabName}-tab`)?.style.setProperty('display', 'block');
    // Update active tab styling
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelector(`[onclick="switchTab('${tabName}')"]`)?.classList.add('active');
    // Clear previous results
    clearResults();
    if (tabName == 'all') {
        fetchAllIncidents();
    }
}
function updatePaginationControls() {
    const pageElement = document.getElementById('current-page');
    const prevButton = document.getElementById('prev-page');
    const nextButton = document.getElementById('next-page');
    if (pageElement)
        pageElement.textContent = currentPage.toString();
    if (prevButton)
        prevButton.disabled = currentPage === 1;
    if (nextButton)
        nextButton.disabled = currentPage === totalPages;
}
async function changePage(dir) {
    if (dir == 'prev' && currentPage > 1) {
        currentPage--;
    }
    else if (dir == 'next' && currentPage < totalPages) {
        currentPage++;
    }
    await fetchAllIncidents();
}
function showLoading() {
    document.getElementById('loading')?.style.setProperty('display', 'block');
    document.getElementById('error')?.style.setProperty('display', 'none');
    const containerElement = document.getElementById('incidents-container');
    if (containerElement) {
        containerElement.innerHTML = '';
    }
}
function hideLoading() {
    document.getElementById('loading')?.style.setProperty('display', 'none');
}
function showError(message) {
    const errorElement = document.getElementById('error');
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.style.display = 'block';
    }
}
function clearResults() {
    const containerElement = document.getElementById('incidents-container');
    if (containerElement) {
        containerElement.innerHTML = '';
    }
    const errorElement = document.getElementById('error');
    if (errorElement) {
        errorElement.style.display = 'none';
    }
}
function createIncidentCard(incident) {
    return `
        <div class="incident-card">
            <h3>${incident.incident_type || incident.incidenttype}</h3>
            <p><strong>Location:</strong> ${incident.location}</p>
            <p><strong>Date of Incident:</strong> ${moment(incident.date_of_incident || incident.dateofincident).format('MMMM D, YYYY')}</p>
            <p><strong>Description:</strong> ${incident.incident_description || incident.incidentdetails}</p>
            <p><strong>Suspect Description:</strong> ${incident.suspect_description || incident.description}</p>
            <p><strong>ID:</strong> ${incident.id}</p>
            ${incident.page_url || incident.page ? `<p><a href="${incident.page_url || incident.page}" target="_blank">View Details</a></p>` : ''}
        </div>
    `;
}
async function fetchAllIncidents() {
    showLoading();
    const offset = (currentPage - 1) * ITEMS_PER_PAGE;
    const headers = new Headers();
    headers.append('Authorization', AUTH_TOKEN);
    const url = `${API_BASE_URL}/getincidents/?offset=${offset}&limit=${ITEMS_PER_PAGE}`;
    try {
        const response = await fetch(url, {
            method: 'GET',
            headers: headers
        });
        if (!response.ok)
            throw new Error('Failed to fetch incidents');
        const data = await response.json();
        totalPages = Math.ceil(data.total / ITEMS_PER_PAGE);
        updatePaginationControls();
        const container = document.getElementById('incidents-container');
        if (container) {
            container.innerHTML = data.incidents.map(createIncidentCard).join('');
        }
    }
    catch (error) {
        if (error instanceof Error) {
            showError(error.message);
        }
    }
    finally {
        hideLoading();
    }
}
async function searchIncidents() {
    showLoading();
    const queryElement = document.getElementById('search-query');
    const limitElement = document.getElementById('search-limit');
    if (!queryElement?.value) {
        showError('Please enter a search query');
        hideLoading();
        return;
    }
    const url = `${API_BASE_URL}/search/?query=${encodeURIComponent(queryElement.value)}${limitElement.value ? '&limit=' + limitElement.value : ''}`;
    const headers = new Headers();
    headers.append('Authorization', AUTH_TOKEN);
    try {
        const response = await fetch(url, {
            method: 'GET',
            headers: headers
        });
        if (!response.ok)
            throw new Error('Search failed');
        const data = await response.json();
        const container = document.getElementById('incidents-container');
        if (container) {
            container.innerHTML = data.incidents.map(createIncidentCard).join('');
        }
    }
    catch (error) {
        if (error instanceof Error) {
            showError(error.message);
        }
    }
    finally {
        hideLoading();
    }
}
async function getRecommendations() {
    showLoading();
    const idElement = document.getElementById('incident-id');
    const limitElement = document.getElementById('recommend-limit');
    if (!idElement?.value) {
        showError('Please enter an incident ID');
        hideLoading();
        return;
    }
    const url = `${API_BASE_URL}/recommend/${idElement.value}${limitElement.value ? '?limit=' + limitElement.value : ''}`;
    const headers = new Headers();
    headers.append('Authorization', AUTH_TOKEN);
    try {
        const response = await fetch(url, {
            method: 'GET',
            headers: headers
        });
        if (!response.ok)
            throw new Error('Failed to get recommendations');
        const data = await response.json();
        const container = document.getElementById('incidents-container');
        if (container) {
            container.innerHTML = data.incidents.map(createIncidentCard).join('');
        }
    }
    catch (error) {
        if (error instanceof Error) {
            showError(error.message);
        }
    }
    finally {
        hideLoading();
    }
}
// Load recent incidents by default
fetchAllIncidents();
