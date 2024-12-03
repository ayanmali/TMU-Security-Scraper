import React, { useState, useEffect, useRef } from 'react';
import { createIncidentCard, Incident, API_BASE_URL, ITEMS_PER_PAGE, AUTH_TOKEN, HOST } from '../App';

const SearchResults: React.FC = () => {
    // State variable to track the "Search" button press
    const [ showResults, setShowResults ] = useState(false);

    // State to track the search results (i.e. the actual data) to display
    const [searchResults, setSearchResults] = useState<Incident[]>([]);

    // Managing loading and error states
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Ref to track the input fields
    const searchInputRef = useRef<HTMLInputElement>(null);
    
    // Ref to store the limit (i.e. the number of records to retrieve).
    const limitRef = useRef<HTMLInputElement>(null);

    async function handleSearch() {
        // Obtaining the current value of the ref and using it as the search query string
        let searchQuery: string = "";
        if (searchInputRef.current) {
            searchQuery = searchInputRef.current.value;
        }
        // Obtaining the current value of the limit ref and using it as the limit query parameter 
        let limit: string = '5';
        if (limitRef.current) {
            limit = limitRef.current.value;
        }

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
                `${API_BASE_URL}/search/?query=${encodeURIComponent(searchQuery)}&limit=${limit}`, {
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
        <div id="search-tab">
            {/* Contains the input fields for the search query and number of results to retrieve */}
            <div className="search-container">
                <input ref={searchInputRef} type="text" id="search-query" placeholder="Enter a search query..."></input>
                <input ref={limitRef} type="number" id="search-limit" placeholder="Number of results (default 5)" min="1"></input>
                <button onClick={handleSearch} disabled={isLoading}>
                    {isLoading ? 'Searching...' : 'Search'}
                </button>
                {/* <button onClick={() => handleClick(regInput.current?.value, document.getElementById('search-limit').value)}>Search</button> */}
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

export default SearchResults;