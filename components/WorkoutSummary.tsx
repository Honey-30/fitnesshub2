import { Activity, Award, BarChart3, Bot, BrainCircuit, Flame, Loader2, RefreshCw, Send, ShieldCheck, Target, Timer } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import { chatWithAI, getWorkoutAnalysis } from '../services/geminiService';
import { WorkoutSession } from '../types';

interface WorkoutSummaryProps {
  session: WorkoutSession;
  onRestart: () => void;
}

const StatCard: React.FC<{ icon: React.ReactNode; title: string; value: string; subtitle?: string; }> = ({ icon, title, value, subtitle }) => (
    <div className="glass-card p-4 rounded-xl text-center">
        <div className="w-12 h-12 mx-auto mb-2 flex items-center justify-center">
            {icon}
        </div>
        <p className="text-2xl font-bold text-slate-800">{value}</p>
        <p className="text-sm text-slate-500">{title}</p>
        {subtitle && <p className="text-xs text-slate-400 mt-1">{subtitle}</p>}
    </div>
);

const TypingIndicator: React.FC = () => (
    <div className="flex items-center gap-3 animate-chat-message">
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-sky-500 flex-shrink-0 flex items-center justify-center text-white"><Bot size={20} /></div>
        <div className="max-w-xs md:max-w-md p-3 rounded-2xl bg-slate-100 text-slate-800 rounded-bl-none flex items-center space-x-1">
            <span className="w-2 h-2 bg-slate-400 rounded-full dot-pulse"></span>
            <span className="w-2 h-2 bg-slate-400 rounded-full dot-pulse" style={{ animationDelay: '0.2s' }}></span>
            <span className="w-2 h-2 bg-slate-400 rounded-full dot-pulse" style={{ animationDelay: '0.4s' }}></span>
        </div>
    </div>
);

const ChatMessage: React.FC<{ message: { role: string; parts: string } }> = ({ message }) => {
    const isUser = message.role === 'user';
    return (
        <div className={`flex items-start gap-3 animate-chat-message ${isUser ? 'justify-end' : ''}`}>
            {!isUser && <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-sky-500 flex-shrink-0 flex items-center justify-center text-white"><Bot size={20} /></div>}
            <div className={`max-w-xs md:max-w-md p-3 rounded-2xl ${isUser ? 'bg-blue-600 text-white rounded-br-none' : 'bg-slate-100 text-slate-800 rounded-bl-none'}`}>
                <p className="text-sm">{message.parts}</p>
            </div>
        </div>
    );
};


const WorkoutSummary: React.FC<WorkoutSummaryProps> = ({ session, onRestart }) => {
    const [aiAnalysis, setAiAnalysis] = useState('');
    const [isLoadingAnalysis, setIsLoadingAnalysis] = useState(true);
    const [chatHistory, setChatHistory] = useState<{ role: string; parts: string }[]>([]);
    const [userMessage, setUserMessage] = useState('');
    const [isChatting, setIsChatting] = useState(false);
    const chatContainerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const fetchAnalysis = async () => {
            try {
                const analysis = await getWorkoutAnalysis(session);
                setAiAnalysis(analysis);
            } catch (error) {
                setAiAnalysis('Could not load AI analysis. Please try again later.');
            } finally {
                setIsLoadingAnalysis(false);
            }
        };
        fetchAnalysis();
    }, [session]);

    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [chatHistory, isChatting]);

    const handleChatSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!userMessage.trim() || isChatting) return;

        const newUserMessage = { role: 'user', parts: userMessage };
        setChatHistory(prev => [...prev, newUserMessage]);
        setUserMessage('');
        setIsChatting(true);

        try {
            const aiResponse = await chatWithAI(session, chatHistory, userMessage);
            const newAiMessage = { role: 'model', parts: aiResponse };
            setChatHistory(prev => [...prev, newAiMessage]);
        } catch (error) {
            const errorMessage = { role: 'model', parts: 'Sorry, I am unable to respond right now.' };
            setChatHistory(prev => [...prev, errorMessage]);
        } finally {
            setIsChatting(false);
        }
    };
    
    const mostCommonMistake = Object.entries(session.mistakes).sort(([, a], [, b]) => (b as number) - (a as number))[0]?.[0] || 'None';

  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-fade-in-up">
        <div className="text-center">
             <h2 className="text-4xl lg:text-5xl font-spartan font-bold text-slate-900 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-sky-500">Workout Complete!</h2>
             <p className="text-slate-600 mt-2">Great job pushing yourself. Here's your advanced performance analysis.</p>
        </div>

        {/* Primary Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {session.exercise.type === 'reps' ? (
                 <StatCard 
                    icon={<Award size={24} className="text-green-500"/>} 
                    title="Clean Reps" 
                    value={`${session.cleanReps}`} 
                />
            ) : (
                <StatCard 
                    icon={<Timer size={24} className="text-sky-500"/>} 
                    title="Time Held" 
                    value={`${session.timeHeldSeconds}s`} 
                />
            )}
            <StatCard 
                icon={<ShieldCheck size={24} className="text-purple-500"/>} 
                title="Avg. Form" 
                value={`${session.avgFormScore}`}
                subtitle="AI Ensemble"
            />
            <StatCard 
                icon={<Target size={24} className="text-yellow-500"/>} 
                title="Top Mistake" 
                value={mostCommonMistake} 
            />
            <StatCard 
                icon={<Flame size={24} className="text-orange-500"/>} 
                title="Calories" 
                value={`${session.caloriesBurned}`} 
            />
        </div>

        {/* Advanced Metrics Grid */}
        {(session.consistencyScore || session.biomechanicalEfficiency) && (
            <div className="glass-card p-6 rounded-2xl">
                <h3 className="text-xl font-bold font-spartan text-slate-900 mb-4 flex items-center">
                    <Activity className="w-6 h-6 mr-3 text-blue-600" /> Advanced Performance Metrics
                </h3>
                <div className="grid grid-cols-2 gap-4">
                    {session.consistencyScore !== undefined && (
                        <StatCard
                            icon={<BarChart3 size={20} className="text-purple-500"/>}
                            title="Consistency"
                            value={`${session.consistencyScore}%`}
                            subtitle="Rep-to-rep"
                        />
                    )}
                    {session.biomechanicalEfficiency !== undefined && (
                        <StatCard
                            icon={<Activity size={20} className="text-green-500"/>}
                            title="Bio-Efficiency"
                            value={`${session.biomechanicalEfficiency}%`}
                            subtitle="Movement quality"
                        />
                    )}
                </div>
            </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="glass-card p-6 rounded-2xl">
                <h3 className="text-xl font-bold font-spartan text-slate-900 mb-4 flex items-center"><BrainCircuit className="w-6 h-6 mr-3 text-blue-600" /> AI Coach Analysis</h3>
                {isLoadingAnalysis ? (
                    <div className="flex items-center justify-center h-24"><Loader2 className="w-8 h-8 animate-spin text-blue-500" /></div>
                ) : (
                    <div className="prose text-slate-700" dangerouslySetInnerHTML={{ __html: aiAnalysis.replace(/\*\*(.*?)\*\*/g, '<strong class="text-blue-600">$1</strong>') }} />
                )}
            </div>
             <div className="glass-card p-6 rounded-2xl flex flex-col">
                <h3 className="text-xl font-bold font-spartan text-slate-900 mb-4 flex items-center"><Bot className="w-6 h-6 mr-3 text-sky-600" /> Ask AI a Question</h3>
                <div 
                    ref={chatContainerRef} 
                    className="flex-grow space-y-4 overflow-y-auto h-48 pr-2 mb-4 bg-slate-100/50 rounded-lg p-2"
                    aria-live="polite"
                    aria-atomic="false"
                >
                   {chatHistory.length === 0 && !isChatting ? (
                       <div className="flex items-center justify-center h-full"><p className="text-slate-500">Ask about your performance...</p></div>
                   ) : (
                       chatHistory.map((msg, index) => <ChatMessage key={index} message={msg} />)
                   )}
                   {isChatting && <TypingIndicator />}
                </div>
                <form onSubmit={handleChatSubmit} className="flex gap-2">
                    <input 
                        type="text"
                        value={userMessage}
                        onChange={(e) => setUserMessage(e.target.value)}
                        placeholder="e.g., How can I improve my squat depth?"
                        aria-label="Ask AI a question about your workout"
                        className="w-full bg-white border border-slate-300 rounded-md p-3 text-slate-800 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition"
                    />
                    <button type="submit" disabled={isChatting} aria-label="Send message" className="bg-blue-600 hover:bg-blue-700 p-3 rounded-md text-white disabled:bg-slate-400">
                        {isChatting ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
                    </button>
                </form>
             </div>
        </div>

        <div className="text-center pt-4">
            <button onClick={onRestart} className="inline-flex items-center justify-center bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-lg transition duration-300 shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40">
                <RefreshCw className="mr-2 h-4 w-4" /> Start a New Workout
            </button>
        </div>
    </div>
  );
};

export default WorkoutSummary;