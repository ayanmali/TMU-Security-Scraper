import * as moment from 'moment';

const API_BASE_URL = 'http://127.0.0.1:8000';  // Update this with your API URL
const ITEMS_PER_PAGE = 20;
let currentPage = 1;
let totalPages = 1;

// Defining an interface for an Incident record
interface Incident {
    id: number;
    incident_type?: string;
    incidenttype?: string,
    location: string;
    date_of_incident?: string;
    dateofincident?: string;
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

// interface SearchResponse {
//     count: number;
//     results: Incident[];
// }

function switchTab(tabName: string): void {
    // Hide all tabs
    document.getElementById('all-tab')?.style.setProperty('display', 'none');
    document.getElementById('search-tab')?.style.setProperty('display', 'none')
    document.getElementById('recommendations-tab')?.style.setProperty('display', 'none')
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`)?.style.setProperty('display', 'block')
    
    // Update active tab styling
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelector(`[onclick="switchTab('${tabName}')"]`)?.classList.add('active');
    
    // Clear previous results
    clearResults();

    if (tabName == 'all') {
        fetchAllIncidents();
    }
}

function updatePaginationControls(): void {
    const pageElement = document.getElementById('current-page');
    const prevButton = document.getElementById('prev-page') as HTMLButtonElement;
    const nextButton = document.getElementById('next-page') as HTMLButtonElement;

    if (pageElement) pageElement.textContent = currentPage.toString();
    if (prevButton) prevButton.disabled = currentPage === 1;
    if (nextButton) nextButton.disabled = currentPage === totalPages;
}

async function changePage(dir): Promise<void> {
    if (dir == 'prev' && currentPage > 1) {
        currentPage--;
    }
    else if (dir == 'next' && currentPage < totalPages) {
        currentPage++;
    }
    await fetchAllIncidents();
}

function showLoading(): void {
    document.getElementById('loading')?.style.setProperty('display', 'block');
    document.getElementById('error')?.style.setProperty('display', 'none');
    const containerElement = document.getElementById('incidents-container');
    if (containerElement) { containerElement.innerHTML = ''; }
}

function hideLoading(): void {
    document.getElementById('loading')?.style.setProperty('display', 'none');
}

function showError(message: string): void {
    const errorElement = document.getElementById('error');
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.style.display = 'block';
    }
}

function clearResults(): void {
    const containerElement = document.getElementById('incidents-container');
    if (containerElement) { containerElement.innerHTML = ''; }

    const errorElement = document.getElementById('error');
    if (errorElement) { errorElement.style.display = 'none'; }
}

function createIncidentCard(incident: Incident): string {
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

async function fetchAllIncidents(): Promise<void> {
    showLoading();
    const offset = (currentPage - 1) * ITEMS_PER_PAGE;
    const url = `${API_BASE_URL}/getincidents?offset=${offset}&limit=${ITEMS_PER_PAGE}`;
    
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to fetch incidents');
        const data = await response.json();
        
        totalPages = Math.ceil(data.total / ITEMS_PER_PAGE);
        updatePaginationControls();

        const container = document.getElementById('incidents-container');
        if (container) { container.innerHTML = data.incidents.map(createIncidentCard).join(''); }
    } catch (error) {  
        if (error instanceof Error) { showError(error.message); }
    } finally {
        hideLoading();
    }
}

async function searchIncidents(): Promise<void> {
    showLoading();
    const queryElement = document.getElementById('search-query') as HTMLInputElement;
    const limitElement = document.getElementById('search-limit') as HTMLInputElement;
    
    if (!queryElement?.value) {
        showError('Please enter a search query');
        hideLoading();
        return;
    }

    const url = `${API_BASE_URL}/search?query=${encodeURIComponent(queryElement.value)}${limitElement.value ? '&limit=' + limitElement.value : ''}`;
    
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error('Search failed');
        const data: APIResponse = await response.json();
        
        const container = document.getElementById('incidents-container');
        if (container) { container.innerHTML = data.incidents.map(createIncidentCard).join(''); }
    } catch (error) {
        if (error instanceof Error) { showError(error.message); }
    } finally {
        hideLoading();
    }
}

async function getRecommendations(): Promise<void> {
    showLoading();
    const idElement = document.getElementById('incident-id') as HTMLInputElement;
    const limitElement = document.getElementById('recommend-limit') as HTMLInputElement;
    
    if (!idElement?.value) {
        showError('Please enter an incident ID');
        hideLoading();
        return;
    }

    const url = `${API_BASE_URL}/recommend/${idElement.value}${limitElement.value ? '?limit=' + limitElement.value : ''}`;
    
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to get recommendations');
        const data: APIResponse = await response.json();
        
        const container = document.getElementById('incidents-container');
        if (container) { container.innerHTML = data.incidents.map(createIncidentCard).join(''); }
    } catch (error) {
        if (error instanceof Error) { showError(error.message); }
    } finally {
        hideLoading();
    }
}

// Load recent incidents by default
fetchAllIncidents();