import React from 'react';
import { useState } from 'react';

interface NavbarProps {
    onTabChange: (tab: string) => void;
  }
  
const Navbar: React.FC<NavbarProps> = ({ onTabChange }) => {
    const [activeTab, setActiveTab] = useState('all');

    function handleClick(tab: string) {
        setActiveTab(tab);
        onTabChange(tab);
    }

    return (
    <div className="tabs">
        <button className={activeTab == "all"  ? 'tab active' : 'tab'} data-tab="all" onClick={() => handleClick('all')}>All Incidents</button>
        <button className={activeTab == "search"  ? 'tab active' : 'tab'} data-tab="search" onClick={() => handleClick('search')}>Search</button>
        <button className={activeTab == "recommend"  ? 'tab active' : 'tab'} data-tab="recommend" onClick={() => handleClick('recommend')}>Recommendations</button>
        <button className={activeTab == "analytics"  ? 'tab active' : 'tab'} data-tab="analytics" onClick={() => handleClick('analytics')}>Analytics</button>
    </div>
    );
}

export default Navbar;