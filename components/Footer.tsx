import React from 'react';
import { Activity } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="glass-card border-t border-slate-200/80 mt-auto">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8 text-center text-slate-500">
        <div className="flex justify-center items-center gap-2 mb-2">
           <Activity className="w-5 h-5 text-slate-400" />
           <p className="font-spartan font-bold text-slate-600">AI Fitness Hub</p>
        </div>
        <p className="text-sm">&copy; {new Date().getFullYear()} All Rights Reserved. Your Journey to a Healthier You.</p>
      </div>
    </footer>
  );
};

export default Footer;