/*
Component to define the Analytics tab.
*/

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { API_BASE_URL, AUTH_TOKEN, HOST } from '../App';

// import { createIncidentCard, Incident, API_BASE_URL, ITEMS_PER_PAGE, AUTH_TOKEN, HOST } from '../App';
import '../App.css'

// Define interfaces for data types
interface TimeSeriesData {
    index: string[];
    weekly_incident_counts: number[];
    forecast_series: number[];
}

interface PlotData {
    date: Date;
    value: number;
    type: 'actual' | 'forecast';
}

// const sampleData = {
//     index: [
//       "2018-07-01T00:00:00Z",
//       "2018-07-08T00:00:00Z",
//       "2018-07-15T00:00:00Z",
//       "2018-07-22T00:00:00Z",
//       "2018-07-29T00:00:00Z",
//       "2018-08-05T00:00:00Z",
//       "2018-08-12T00:00:00Z"
//     ],
//     weekly_incident_counts: [1, 1, 3, 4, 2, 5, 3],
//     forecast_series: [4, 2, 3, 4, 5, 6, 7]
// };

const TimeSeriesForecastChart: React.FC = () => {
    const svgRef = useRef(null);
    const [ countData, setCountData ] = useState<TimeSeriesData>(
        {
            index: [],
            weekly_incident_counts: [],
            forecast_series: []
        }
    );
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    async function getWeeklyData() {
        setIsLoading(true);
        setError(null);

        // Error handling
        if(!API_BASE_URL) throw new Error("API_BASE_URL is undefined");
        if(!AUTH_TOKEN) throw new Error("AUTH_TOKEN is undefined");
        if(!HOST) throw new Error("HOST is undefined");
        // Setting the headers for the HTTP request
        const headers = new Headers( {
            'Authorization': AUTH_TOKEN,
            'Access-Control-Allow-Origin': HOST,
            'Access-Control-Allow-Credentials': 'true'
        } )
        
        try {
            const response = await fetch(
                `${API_BASE_URL}/weekly-forecast/`, {
                    method: 'GET', headers: headers
                }
            )
            
            if (!response.ok) { throw new Error('An error occurred when retrieving the forecast data.') }
            const data = await response.json();
            setCountData(
                {
                    index: data.index,
                    weekly_incident_counts: data.weekly_incident_counts,
                    forecast_series: data.forecast_series
                }
            );
        } catch(err) {
            setError(err instanceof Error ? err.message : 'An error occurred when sending the request.');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    }
  
    useEffect(() => {
      getWeeklyData()
      // Adding a guard clause to stop the code from running if the data is empty
      if (!svgRef.current || countData.index.length === 0) return;
  
      // Clear any existing SVG content
      d3.select(svgRef.current).selectAll("*").remove();
  
      // Defines the function for parsing dates and combine actual and forecast data
      const parseDate = d3.timeParse("%Y-%m-%dT%H:%M:%SZ");
      
      const actualData: PlotData[] = countData.index.map((date, i) => {
        const parsedDate = parseDate(date);
        if (!parsedDate) { throw new Error(`Unable to parse date: ${date}`) }
        
        return {
            date: parsedDate,
            value: countData.weekly_incident_counts[i],
            type: 'actual'
        };
    });
      
      const lastActualDate = actualData[actualData.length - 1].date;
      // Checking to ensure the final date in the array is not null
      if (lastActualDate == null) {
        throw Error('The final date in the index is null.');
      }

      const forecastData: PlotData[] = countData.forecast_series.map((value, i) => ({
        date: new Date(lastActualDate.getTime() + (i + 1) * 7 * 24 * 60 * 60 * 1000), // Add weeks
        value: value,
        type: 'forecast'
      }));
  
      const combinedData: PlotData[] = [...actualData, ...forecastData];
      
      // Type assertion
      const dateExtent = d3.extent(combinedData, d => d.date) as [Date, Date];

      // Set up dimensions
      const margin = { top: 20, right: 30, bottom: 30, left: 40 };
      const width = 1200 - margin.left - margin.right;
      const height = 400 - margin.top - margin.bottom;
  
      // Create SVG
      const svg = d3.select(svgRef.current)
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);
  
      // X Scale
      const x = d3.scaleTime()
        .domain(dateExtent)
        .range([0, width]);
  
      // Y Scale
      const maxData = d3.max(combinedData, d => d.value);
      const y = d3.scaleLinear()
        .domain([0, maxData ? maxData * 1.2 : 10])
        .range([height, 0]);
  
      // Create line generator
      const line = d3.line<PlotData>()
        .x(d => x(d.date))
        .y(d => y(d.value));
  
      // Actual data line
      svg.append("path")
        .datum(actualData)
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-width", 2)
        .attr("d", line);
  
      // Forecast data line
      svg.append("path")
        .datum(forecastData)
        .attr("fill", "none")
        .attr("stroke", "red")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5,5")
        .attr("d", line);
  
      // X Axis
      svg.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(x));
  
      // Y Axis
      svg.append("g")
        .call(d3.axisLeft(y));
  
      // Vertical line to separate actual and forecast
      svg.append("line")
        .attr("x1", x(lastActualDate))
        .attr("x2", x(lastActualDate))
        .attr("y1", 0)
        .attr("y2", height)
        .attr("stroke", "gray")
        .attr("stroke-dasharray", "3,3");
  
      // Legend
      const legend = svg.append("g")
      // Adjusting the position of the box
        .attr("transform", `translate(${width - 150},0)`);
  
      legend.append("rect")
        .attr("width", 150)
        .attr("height", 60)
        .attr("fill", "white")
        .attr("stroke", "gray");
  
      legend.append("line")
        .attr("x1", 10)
        .attr("x2", 30)
        .attr("y1", 20)
        .attr("y2", 20)
        .attr("stroke", "steelblue")
        .attr("stroke-width", 2);
  
      legend.append("text")
        .attr("x", 40)
        .attr("y", 20)
        .attr("alignment-baseline", "middle")
        .text("Actual Data");
  
      legend.append("line")
        .attr("x1", 10)
        .attr("x2", 30)
        .attr("y1", 40)
        .attr("y2", 40)
        .attr("stroke", "red")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "5,5");
  
      legend.append("text")
        .attr("x", 40)
        .attr("y", 40)
        .attr("alignment-baseline", "middle")
        .text("Forecast");
  
    }, [countData]);
  
    return (
      <div className="incident-forecast">
        <h2>Count of TMU Campus Security Incidents per Week + Forecast</h2>
        <svg ref={svgRef}></svg>
      </div>
    );
  };
  
  export default TimeSeriesForecastChart;