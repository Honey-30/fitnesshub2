import React from 'react';
import { CalculationResults, Page } from '../types';
import { Droplets, Beef, Wheat, Leaf, BrainCircuit, RotateCcw, Activity, Utensils, GlassWater } from 'lucide-react';
import BmiGauge from './BmiGauge';

interface ResultsDashboardProps {
  results: CalculationResults;
  aiTips: string;
  isLoadingAiTips: boolean;
  onNavigate: (page: Page) => void;
  error: string | null;
}

const StatCard: React.FC<{ icon: React.ReactNode; title: string; value: string; subtitle: string; }> = ({ icon, title, value, subtitle }) => (
  <div className="glass-card p-6 rounded-xl text-center flex flex-col items-center justify-center">
    <div className="w-12 h-12 mx-auto mb-3 flex items-center justify-center">
      {icon}
    </div>
    <p className="text-3xl font-bold text-slate-800">{value}</p>
    <p className="text-slate-600 text-sm">{title}</p>
    <p className="text-slate-400 text-xs">{subtitle}</p>
  </div>
);

const LoadingSpinner: React.FC = () => (
    <div className="flex items-center justify-center space-x-2">
        <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></div>
        <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse [animation-delay:0.2s]"></div>
        <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse [animation-delay:0.4s]"></div>
        <p className="text-blue-500">Atlas is generating your plan...</p>
    </div>
);

const AiTipsCard: React.FC<{ tips: string, isLoading: boolean, error: string | null }> = ({ tips, isLoading, error }) => {
    const formatTips = (text: string) => {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong class="text-blue-600">$1</strong>')
            .replace(/-\s(.*?)(?:\n|$)/g, '<li class="ml-4 mb-2 pl-2 border-l-2 border-blue-500/30">$1</li>')
            .replace(/(<li.*?>.*?<\/li>)/gs, '<ul>$1</ul>')
            .replace(/<\/ul>\s*<ul>/gs, '');
    };
    
    return (
        <div className="glass-card p-6 rounded-2xl border-blue-500/30 border col-span-1 md:col-span-3">
            <h3 className="text-xl font-bold font-spartan text-slate-900 mb-4 flex items-center"><BrainCircuit className="w-6 h-6 mr-3 text-blue-600" /> Your AI Coach Insights</h3>
            {isLoading && <LoadingSpinner />}
            {error && <p className="text-red-600 bg-red-100/50 p-3 rounded-md">{error}</p>}
            {!isLoading && !error && tips && (
                 <div className="prose text-slate-700 prose-strong:text-blue-600" dangerouslySetInnerHTML={{ __html: formatTips(tips) }} />
            )}
        </div>
    );
};

const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ results, aiTips, isLoadingAiTips, onNavigate, error }) => {
  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-fade-in-up">
        <div className="glass-card p-6 sm:p-8 rounded-2xl">
             <BmiGauge bmiValue={results.bmi.value} bmiCategory={results.bmi.category} />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <StatCard icon={<Activity size={28} className="text-orange-500"/>} title="Daily Calories" value={`${results.idealCalories}`} subtitle="kcal / day" />
            <StatCard icon={<Droplets size={28} className="text-sky-500"/>} title="Water Intake" value={`${(results.waterIntake / 1000).toFixed(1)} L`} subtitle="per day" />
            <StatCard icon={<Beef size={28} className="text-purple-500"/>} title="Protein" value={`${results.macros.protein} g`} subtitle="per day"/>
        </div>

        <AiTipsCard tips={aiTips} isLoading={isLoadingAiTips} error={error} />
        
        <div className="flex flex-col sm:flex-row justify-center items-center gap-4 pt-4">
             <button onClick={() => onNavigate('calculator')} aria-label="Recalculate your plan" className="w-full sm:w-auto inline-flex items-center justify-center bg-white hover:bg-slate-100 text-slate-700 font-semibold py-2 px-6 rounded-md transition duration-300 border border-slate-300">
                <RotateCcw className="mr-2 h-4 w-4" />
                Start Over
            </button>
            <button onClick={() => onNavigate('food-tracker')} aria-label="Go to nutrition tracker" className="w-full sm:w-auto inline-flex items-center justify-center bg-gradient-to-r from-blue-600 to-sky-500 hover:from-blue-500 hover:to-sky-400 text-white font-bold py-2 px-6 rounded-md transition duration-300 shadow-lg shadow-blue-500/20 hover:shadow-2xl hover:shadow-blue-500/30">
                <Utensils className="mr-2 h-4 w-4" />
                Track Nutrition
            </button>
        </div>
    </div>
  );
};

export default ResultsDashboard;