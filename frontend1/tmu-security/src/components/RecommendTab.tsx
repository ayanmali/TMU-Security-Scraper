import React, { useState, useEffect, useRef } from 'react';
import { createIncidentCard, Incident, API_BASE_URL, ITEMS_PER_PAGE, AUTH_TOKEN, HOST } from '../App';
import '../App.css'

const RecommendResults: React.FC = () => {
    // State variable to track the "Get Recommendations" button press
    const [ showResults, setShowResults ] = useState(false);

    // State to track the search results to display
    const [searchResults, setSearchResults] = useState<Incident[]>([]);

    // Managing loading and error states
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Ref to track the incident identifier provided in the input field
    const identifierRef = useRef<HTMLInputElement>(null);
    
    // Ref to store the limit (i.e. the number of records to retrieve).
    const limitRef = useRef<HTMLInputElement>(null);

    async function handleClick() {
        // Obtaining the current value of the ref and using it as the search query string
        let identifier: string = "";
        if (identifierRef.current) {
            identifier = identifierRef.current.value;
        }
        // Obtaining the current value of the limit ref and using it as the limit query parameter 
        let limit: string = '5';
        if (limitRef.current) {
            limit = limitRef.current.value;
        }

        // Adjusting states once the button has been clicked
        setError(null);
        setSearchResults([]);
        setIsLoading(true);

        // Setting the request headers
        const headers = new Headers( {
            'Authorization': AUTH_TOKEN,
            'Access-Control-Allow-Origin': HOST,
            'Access-Control-Allow-Credentials': 'true'
        } )

        // Attempting the HTTP request
        try {
            const response = await fetch(
                `${API_BASE_URL}/recommend/${encodeURIComponent(identifier)}?limit=${limit}`, {
                method: 'GET', headers: headers
            });
            if (!response.ok) throw new Error('Failed to fetch incidents');
            const data = await response.json();
            setSearchResults(data.results);
            setShowResults(true);
        } catch (err) {
            // Handling and logging any errors that occur while fetching
            setError('An error occurred when searching. Ensure the search query contains at least four characters.');
            console.error(err);
        } finally {
            // Adjusting states after the response has been received
            setIsLoading(false);
        }
    }

    // The component being returned
    return (
        // The div that contains all of the search elements
        <div id="recommendations-tab">
            {/* Contains the input fields for the search query and number of results to retrieve */}
            <div className="search-container">
                <div className="tooltip-container">
                    <input ref={identifierRef} type="text" id="incident-id" placeholder="Enter the date of an incident (YYYY-MM-DD)..."></input>
                    {/* Adding a tooltip icon to make the instructions clearer */}
                    <span className="tooltip-icon">?</span>
                    {/* Tooltip text */}
                    <div className="tooltip">
                        If there was more than one incident on a particular date, you can indicate the specific incident by adding a number at the end (i.e. YYYY-MM-DD-N).
                    </div>
                </div>

                <input ref={limitRef} type="number" id="recommend-limit" placeholder="Number of recommendations (default is 5)" min="1"></input>
                <button onClick={handleClick} disabled={isLoading}>Get Recommendations</button>
            </div>

            {/* Displaying any errors */}
            {error && (
                <div className="error">
                    {error}
                </div>
            )}

            {/* Indicating that the page is loading */}
            {isLoading && (
            <div className="loading">
                Loading results...
            </div>
            )}

            {/* Displaying the cards for all of the retrieved incidents */}
            {showResults &&
                // Mapping each retrieved incident to a card
                searchResults.map((incident) => (
                    <div key={incident.id}>
                        {createIncidentCard(incident)}
                    </div>
                ))
            }

        </div>

    );
}

export default RecommendResults;