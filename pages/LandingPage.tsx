import { ArrowRight, BrainCircuit, Dumbbell, Target, Utensils } from 'lucide-react';
import React from 'react';
import { Page } from '../types';

interface LandingPageProps {
  onNavigate: (page: Page) => void;
}

const FeatureCard: React.FC<{ icon: React.ReactNode; title: string; description: string; delay: string; }> = ({ icon, title, description, delay }) => (
    <div className="glass-card p-6 rounded-2xl text-center animate-fade-in-scale transition-all duration-500 hover:-translate-y-2 hover:shadow-2xl backdrop-blur-sm border border-slate-200/50" style={{ animationDelay: delay }}>
        <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-slate-900 to-slate-700 rounded-2xl flex items-center justify-center transform transition-all duration-500 hover:scale-110 hover:rotate-6 shadow-lg">
            {icon}
        </div>
        <h3 className="text-xl font-bold font-spartan text-slate-900 mb-2 transition-colors duration-300 hover:text-blue-600">{title}</h3>
        <p className="text-slate-600 text-sm leading-relaxed">{description}</p>
    </div>
);

const HowItWorksStep: React.FC<{ number: string; title: string; description: string; delay: string; }> = ({ number, title, description, delay }) => (
    <div className="text-center animate-fade-in-scale group" style={{ animationDelay: delay }}>
        <div className="w-16 h-16 mx-auto mb-4 border-2 border-slate-300 rounded-full flex items-center justify-center font-spartan text-2xl font-bold text-slate-900 transition-all duration-500 group-hover:border-blue-600 group-hover:bg-blue-50 group-hover:scale-110 group-hover:text-blue-600">
            {number}
        </div>
        <h3 className="text-xl font-bold font-spartan text-slate-900 mb-2 transition-colors duration-300 group-hover:text-blue-600">{title}</h3>
        <p className="text-slate-600 text-sm leading-relaxed">{description}</p>
    </div>
);


const LandingPage: React.FC<LandingPageProps> = ({ onNavigate }) => {
  return (
    <div className="text-slate-800">
      {/* Hero Section */}
      <section className="text-center py-20 sm:py-28">
        <div className="max-w-4xl mx-auto px-4">
            <h1 className="text-5xl sm:text-7xl lg:text-8xl font-extrabold font-spartan tracking-tighter animate-fade-in-scale text-slate-900">
                Your <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-sky-500">AI-Powered</span> Fitness Coach
            </h1>
            <p className="text-lg sm:text-xl text-slate-600 max-w-2xl mx-auto mt-6 animate-fade-in-scale leading-relaxed" style={{ animationDelay: '0.2s' }}>
                Stop guessing, start progressing. Get a personalized nutrition and workout plan based on your unique data, and achieve your goals faster than ever.
            </p>
            <button 
                onClick={() => onNavigate('calculator')} 
                className="mt-10 inline-flex items-center justify-center bg-slate-900 hover:bg-blue-600 text-white font-bold py-4 px-8 rounded-xl transition-all duration-500 transform hover:scale-105 hover:-translate-y-1 text-lg animate-fade-in-scale shadow-xl hover:shadow-2xl" 
                style={{ animationDelay: '0.4s' }}
            >
                Get Your Free Plan <ArrowRight className="ml-2 h-5 w-5" />
            </button>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4">
            <h2 className="text-4xl font-bold font-spartan text-center mb-12 animate-fade-in-scale text-slate-900">Everything You Need to Succeed</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
                <FeatureCard 
                    icon={<Target size={32} className="text-white" />} 
                    title="Personalized Plans"
                    description="Get calorie, macro, and water targets calculated specifically for your body and goals."
                    delay="0.2s"
                />
                <FeatureCard 
                    icon={<BrainCircuit size={32} className="text-white" />} 
                    title="AI Coach Insights"
                    description="Receive actionable tips and motivation from our advanced AI to keep you on track."
                    delay="0.4s"
                />
                <FeatureCard 
                    icon={<Dumbbell size={32} className="text-white" />} 
                    title="Real-Time Workout Coach"
                    description="Use your camera to get live feedback on your exercise form, count clean reps, and prevent injuries."
                    delay="0.6s"
                />
                <FeatureCard 
                    icon={<Utensils size={32} className="text-white" />} 
                    title="Nutrition Tracking"
                    description="Easily log your meals and water intake to monitor your progress against your daily targets."
                    delay="0.8s"
                />
            </div>
        </div>
      </section>
      
      {/* How It Works Section */}
        <section className="py-20">
            <div className="max-w-5xl mx-auto px-4">
                <h2 className="text-4xl font-bold font-spartan text-center mb-12 animate-fade-in-up text-slate-900">Get Started in 3 Simple Steps</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
                    <HowItWorksStep
                        number="01"
                        title="Enter Your Data"
                        description="Complete our simple, interactive form with your stats, activity level, and goals."
                        delay="0.2s"
                    />
                    <HowItWorksStep
                        number="02"
                        title="Get Your Plan"
                        description="Instantly receive your personalized plan with detailed targets and AI-powered insights."
                        delay="0.4s"
                    />
                     <HowItWorksStep
                        number="03"
                        title="Track & Improve"
                        description="Use the trackers and live workout coach to stay consistent and achieve your fitness goals."
                        delay="0.6s"
                    />
                </div>
            </div>
        </section>
    </div>
  );
};

export default LandingPage;