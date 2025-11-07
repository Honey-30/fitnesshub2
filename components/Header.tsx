import { Activity, Calculator, Dumbbell, GlassWater, Settings, Utensils } from 'lucide-react';
import React from 'react';
import { Page } from '../types';

interface HeaderProps {
    currentPage: Page;
    onNavigate: (page: Page) => void;
    hasResults: boolean;
}

const NavButton: React.FC<{
    page: Page;
    currentPage: Page;
    onNavigate: (page: Page) => void;
    disabled?: boolean;
    icon: React.ReactNode;
    label: string;
    ariaLabel: string;
}> = ({ page, currentPage, onNavigate, disabled = false, icon, label, ariaLabel }) => {
    const isActive = currentPage === page || (currentPage === 'results' && page === 'calculator');
    return (
        <button
            onClick={() => onNavigate(page)}
            disabled={disabled}
            aria-label={ariaLabel}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-semibold transition-all duration-300 transform active:scale-95 ${
                isActive
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-500/30'
                    : 'text-slate-500 hover:bg-slate-200/70 hover:text-slate-800'
            } ${disabled ? 'opacity-40 cursor-not-allowed hover:bg-transparent' : ''}`}
        >
            {icon}
            <span className="hidden sm:inline">{label}</span>
        </button>
    );
};


const Header: React.FC<HeaderProps> = ({ currentPage, onNavigate, hasResults }) => {
  return (
    <header className="glass-card sticky top-0 z-50 border-b border-slate-200/80">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <button onClick={() => onNavigate('landing')} aria-label="Go to landing page" className="flex items-center gap-2 text-slate-800 transition-transform transform hover:scale-105">
            <Activity className="w-8 h-8 text-blue-600" />
            <span className="font-spartan font-bold text-xl tracking-tighter hidden md:block">AI Fitness Hub</span>
          </button>

          {/* Navigation Links */}
          <div className="flex items-center gap-2">
            <NavButton
                page="calculator"
                currentPage={currentPage}
                onNavigate={onNavigate}
                icon={<Calculator size={18} />}
                label="My Plan"
                ariaLabel="Go to My Plan and Calculator page"
            />
             <NavButton
                page="workout"
                currentPage={currentPage}
                onNavigate={onNavigate}
                icon={<Dumbbell size={18} />}
                label="Workout"
                ariaLabel="Go to Workout page"
            />
            <NavButton
                page="food-tracker"
                currentPage={currentPage}
                onNavigate={onNavigate}
                disabled={!hasResults}
                icon={<Utensils size={18} />}
                label="Food Tracker"
                ariaLabel="Go to Food Tracker page"
            />
            <NavButton
                page="water-tracker"
                currentPage={currentPage}
                onNavigate={onNavigate}
                disabled={!hasResults}
                icon={<GlassWater size={18} />}
                label="Water Tracker"
                ariaLabel="Go to Water Tracker page"
            />
            {/* Settings Button */}
            <NavButton
                page="api-settings"
                currentPage={currentPage}
                onNavigate={onNavigate}
                icon={<Settings size={18} />}
                label="Settings"
                ariaLabel="Go to API Settings page"
            />
          </div>
        </div>
      </nav>
    </header>
  );
};

export default Header;