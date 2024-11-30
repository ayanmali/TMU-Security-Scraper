import React, { useState, useEffect } from 'react';
import { createIncidentCard } from '../App';
import { Incident } from '../App';
import '../App.css'

const RecommendResults: React.FC = () => {

    // The component being returned
    return (
        // The div that contains all of the search elements
        <div id="recommendations-tab">
            {/* Contains the input fields for the search query and number of results to retrieve */}
            <div className="search-container">
                <div className="tooltip-container">
                    <input type="text" id="incident-id" placeholder="Enter the date of an incident (YYYY-MM-DD)..."></input>
                    {/* Adding a tooltip icon to make the instructions clearer */}
                    <span className="tooltip-icon">?</span>
                    {/* Tooltip text */}
                    <div className="tooltip">
                        If there was more than one incident on a particular date, you can indicate the specific incident by adding a number at the end (i.e. YYYY-MM-DD-N).
                    </div>
                </div>

                <input type="number" id="recommend-limit" placeholder="Number of recommendations (default is 5)" min="1"></input>
                <button>Get Recommendations</button>
            </div>
        </div>
    );
}

export default RecommendResults;