// Old JavaScript file (not being used)

const API_BASE_URL = 'http://127.0.0.1:8000';  // Update this with your API URL
const ITEMS_PER_PAGE = 20;
let currentPage = 1;
let totalPages = 1;

function switchTab(tabName) {
    // Hide all tabs
    document.getElementById('all-tab').style.display = 'none';
    document.getElementById('search-tab').style.display = 'none';
    document.getElementById('recommendations-tab').style.display = 'none';
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).style.display = 'block';
    
    // Update active tab styling
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.querySelector(`[onclick="switchTab('${tabName}')"]`).classList.add('active');
    
    // Clear previous results
    clearResults();

    if (tabName == 'all') {
        fetchAllIncidents();
    }
}

function updatePaginationControls() {
    document.getElementById('current-page').textContent = currentPage;
    document.getElementById('prev-page').disabled = currentPage === 1;
    document.getElementById('next-page').disabled = currentPage === totalPages;
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
    document.getElementById('loading').style.display = 'block';
    document.getElementById('error').style.display = 'none';
    document.getElementById('incidents-container').innerHTML = '';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function showError(message) {
    const errorElement = document.getElementById('error');
    errorElement.textContent = message;
    errorElement.style.display = 'block';
}

function clearResults() {
    document.getElementById('incidents-container').innerHTML = '';
    document.getElementById('error').style.display = 'none';
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
    const url = `${API_BASE_URL}/getincidents?offset=${offset}&limit=${ITEMS_PER_PAGE}`;
    
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to fetch incidents');
        const data = await response.json();
        
        totalPages = Math.ceil(data.total / ITEMS_PER_PAGE);
        updatePaginationControls();

        const container = document.getElementById('incidents-container');
        container.innerHTML = data.incidents.map(createIncidentCard).join('');
    } catch (error) {  
        showError(error.message);
    } finally {
        hideLoading();
    }
}

async function searchIncidents() {
    showLoading();
    const query = document.getElementById('search-query').value;
    const limit = document.getElementById('search-limit').value;
    
    if (!query) {
        showError('Please enter a search query');
        hideLoading();
        return;
    }

    const url = `${API_BASE_URL}/search?query=${encodeURIComponent(query)}${limit ? '&limit=' + limit : ''}`;
    
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error('Search failed');
        const data = await response.json();
        
        const container = document.getElementById('incidents-container');
        container.innerHTML = data.results.map(createIncidentCard).join('');
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

async function getRecommendations() {
    showLoading();
    const id = document.getElementById('incident-id').value;
    const limit = document.getElementById('recommend-limit').value;
    
    if (!id) {
        showError('Please enter an incident ID');
        hideLoading();
        return;
    }

    const url = `${API_BASE_URL}/recommend/${id}${limit ? '?limit=' + limit : ''}`;
    
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to get recommendations');
        const data = await response.json();
        
        const container = document.getElementById('incidents-container');
        container.innerHTML = data.results.map(createIncidentCard).join('');
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// Load recent incidents by default
fetchAllIncidents();