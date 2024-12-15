import React, { useState, useEffect } from 'react';
import { createIncidentCard, Incident, API_BASE_URL, ITEMS_PER_PAGE, AUTH_TOKEN, HOST } from '../App';
import '../App.css'

// Creating a function component to represent the list of incidents
const IncidentList: React.FC = () => {
    // state to store the current list of incidents
    const [incidents, setIncidents] = useState<Incident[]>([]);
    const [currentPage, setCurrentPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Defining the async function to make the HTTP request
    const fetchIncidents = async () => {
        setIsLoading(true);
        setError(null);

        const offset = (currentPage - 1) * ITEMS_PER_PAGE;
        // Error handling
        if(!API_BASE_URL) throw new Error("API_BASE_URL is undefined");
        if(!AUTH_TOKEN) throw new Error("AUTH_TOKEN is undefined");
        if(!HOST) throw new Error("HOST is undefined");
        const headers = new Headers( {
            'Authorization': AUTH_TOKEN,
            'Access-Control-Allow-Origin': HOST,
            'Access-Control-Allow-Credentials': 'true'
        } )

        // Attempt the HTTP request
        try {
            const response = await fetch(
                `${API_BASE_URL}/getincidents/?offset=${offset}&limit=${ITEMS_PER_PAGE}`,
                { method: 'GET', headers } );
            if (!response.ok) throw new Error('Failed to fetch incidents.');

            const data = await response.json();
            setIncidents(data.results);
            setTotalPages(data.pagination.total_pages);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred when sending the request.');
        } finally {
            setIsLoading(false);
        }

    };

    // Fetching data when the component mounts or when the page changes
    useEffect(() => {
        fetchIncidents();
    }, [currentPage]); // fetches when page changes

    return (
        <div className="incidents-list">
            {/** Error msg*/}
            {error && (
                <div className="error-message">
                    {error}
                </div>
            )}

            {/*Loading spinner*/}
            {/* If its not loading, then show the incidents */}
            {isLoading ? (
                <div className="loading-spinner">Loading...</div>
            ) : (
                <>
                    {/* Pagination controls */}
                    <div className="pagination">
                        {/* Previous page button */}
                        <button onClick={() => setCurrentPage(p => p - 1)}
                                disabled={currentPage === 1}>
                                        Previous
                        </button>

                        {/* Indicates the current page number */}
                        <span>
                            Page {currentPage} of {totalPages}
                        </span>

                        {/* Next page button */}
                        <button onClick={() => setCurrentPage(p => p + 1)}
                                disabled={currentPage === totalPages}>
                                        Next
                        </button>
                    </div>

                    {/* Incidents grid */}
                    <div className="incidents-container">
                        {/* Mapping each retrieved incident to a card */}
                        {incidents.map((incident) => (
                            <div key={incident.id}>
                                {createIncidentCard(incident)}
                            </div>
                        ))}
                    </div>

                </>
            )}
        </div>
    );

};

export default IncidentList;