import React, { useCallback, useState } from 'react';
import BmiCalculatorForm from './components/BmiCalculatorForm';
import Footer from './components/Footer';
import Header from './components/Header';
import ResultsDashboard from './components/ResultsDashboard';
import ApiKeySettings from './pages/ApiKeySettings';
import LandingPage from './pages/LandingPage';
import FoodTrackerPage from './pages/MacrosTrackerPage';
import WaterTrackerPage from './pages/WaterTrackerPage';
import WorkoutPage from './pages/WorkoutPage';
import { getFitnessTips } from './services/geminiService';
import { CalculationResults, Page, UserData } from './types';
import { calculateFitnessPlan } from './utils/calculator';

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<Page>('landing');
  const [results, setResults] = useState<CalculationResults | null>(null);
  const [aiTips, setAiTips] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleCalculate = useCallback(async (data: UserData) => {
    setIsLoading(true);
    setError(null);
    setResults(null);
    setAiTips('');

    try {
      const calculatedResults = calculateFitnessPlan(data);
      setResults(calculatedResults);
      setCurrentPage('results');

      const tips = await getFitnessTips(data, calculatedResults);
      setAiTips(tips);
    } catch (e) {
      console.error(e);
      const errorMessage = e instanceof Error ? e.message : 'An unexpected error occurred.';
      setError(`Failed to get AI tips. ${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  const getPageTitleAndDescription = () => {
    switch (currentPage) {
      case 'food-tracker':
        return { title: 'Food & Macros Tracker', description: 'Log your meals and water to visualize your progress against your daily targets.' };
      case 'water-tracker':
        return { title: 'Water Intake Tracker', description: 'Log your daily water intake to stay hydrated and hit your goals.' };
       case 'workout':
        return { title: 'Real-Time Workout Coach', description: 'Select an exercise and get live AI feedback on your form.' };
      case 'api-settings':
        return { title: 'API Key Settings', description: 'Configure your Gemini API key to enable AI-powered features.' };
      case 'results':
         return { title: 'Your Personalized Plan', description: 'Here are your calculated results and AI-powered insights.' };
      case 'calculator':
        return { title: 'Get Your Personalized Plan', description: 'Answer a few questions to generate your custom-tailored fitness and nutrition plan.' };
      case 'landing':
      default:
        return { title: null, description: null };
    }
  };

  const renderContent = () => {
    switch (currentPage) {
      case 'landing':
        return <LandingPage onNavigate={setCurrentPage} />;
      case 'food-tracker':
        return results && <FoodTrackerPage targets={results} />;
      case 'water-tracker':
        return results && <WaterTrackerPage targets={results} />;
       case 'workout':
        return <WorkoutPage userData={results ? { weight: results.bmi.value } : { weight: 75 }} />;
      case 'api-settings':
        return <ApiKeySettings />;
      case 'results':
        return results && (
          <ResultsDashboard 
            results={results} 
            aiTips={aiTips} 
            isLoadingAiTips={isLoading && !aiTips}
            onNavigate={setCurrentPage}
            error={error}
          />
        );
      case 'calculator':
      default:
        return <BmiCalculatorForm onCalculate={handleCalculate} isLoading={isLoading} />;
    }
  };
  
  const { title, description } = getPageTitleAndDescription();

  return (
    <div className="min-h-screen text-slate-800 font-sans flex flex-col">
      <Header currentPage={currentPage} onNavigate={setCurrentPage} hasResults={!!results}/>
      
      <main className="flex-grow pb-8">
        <div className={`p-4 sm:p-6 lg:p-8 ${!title ? 'pt-0 sm:pt-0 lg:pt-0' : ''}`}>
          <div className="max-w-7xl mx-auto">
            {title && (
              <header className="text-center mb-8 md:mb-12 animate-fade-in-up">
                <h1 className="text-4xl sm:text-5xl lg:text-6xl font-spartan font-extrabold tracking-tighter text-slate-900 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-sky-500">
                  {title}
                </h1>
                {description && <p className="text-slate-600 max-w-2xl mx-auto text-base sm:text-lg mt-3">
                  {description}
                </p>}
              </header>
            )}
            {renderContent()}
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
};

export default App;