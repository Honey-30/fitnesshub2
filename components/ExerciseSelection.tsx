import { Activity, Target, TrendingUp, Zap } from 'lucide-react';
import React from 'react';
import { Exercise } from '../types';

interface ExerciseSelectionProps {
  exercises: Exercise[];
  onSelect: (exercise: Exercise) => void;
}

const difficultyColors: Record<string, string> = {
  beginner: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  intermediate: 'bg-amber-50 text-amber-700 border-amber-200',
  advanced: 'bg-rose-50 text-rose-700 border-rose-200',
};

const ExerciseSelection: React.FC<ExerciseSelectionProps> = ({ exercises, onSelect }) => {
  return (
    <div className="relative">
      <div className="relative text-center mb-12 animate-fade-in-up">
        <h2 className="text-4xl font-bold font-spartan text-slate-900 mb-3">
          Select Your Exercise
        </h2>
        <p className="text-slate-600 text-lg">
          Choose an exercise to begin your AI-guided workout session
        </p>
      </div>
      
      <div className="relative grid grid-cols-1 md:grid-cols-2 gap-6 max-w-5xl mx-auto">
        {exercises.map((exercise, index) => (
          <div
            key={exercise.id}
            className="group glass-card p-7 rounded-2xl transition-all duration-500 hover:-translate-y-1 hover:shadow-2xl cursor-pointer border border-slate-200/50 animate-fade-in-up backdrop-blur-sm"
            style={{ animationDelay: `${index * 0.1}s` }}
            onClick={() => onSelect(exercise)}
          >
            <div className="flex items-start gap-5">
              <div className="relative flex-shrink-0">
                <div className="w-16 h-16 bg-gradient-to-br from-slate-900 to-slate-700 rounded-2xl flex items-center justify-center transform transition-all duration-500 group-hover:scale-105 group-hover:rotate-3 shadow-lg">
                  <Activity className="w-8 h-8 text-white" />
                </div>
              </div>
              
              <div className="flex-1 text-left">
                <div className="flex items-start justify-between mb-2">
                  <h3 className="text-2xl font-bold font-spartan text-slate-900 transition-colors duration-300 group-hover:text-blue-600">
                    {exercise.name}
                  </h3>
                  <span className={`px-3 py-1 rounded-full border text-xs font-semibold ${difficultyColors[exercise.difficulty]} transition-all duration-300 whitespace-nowrap ml-2`}>
                    {exercise.difficulty}
                  </span>
                </div>
                
                <p className="text-sm text-slate-600 leading-relaxed mb-4">
                  {exercise.description}
                </p>
                
                <div className="flex flex-wrap gap-2 mb-4">
                  <div className="flex items-center gap-1 text-xs text-slate-500">
                    <Target className="w-4 h-4 text-blue-600" />
                    <span className="font-medium">Targets:</span>
                  </div>
                  {exercise.muscleGroups.map((muscle) => (
                    <span
                      key={muscle}
                      className="text-xs px-2.5 py-1 bg-blue-50/80 text-blue-700 rounded-lg border border-blue-100/50 transition-all duration-300 group-hover:bg-blue-100 font-medium animate-slide-in-left"
                    >
                      {muscle}
                    </span>
                  ))}
                </div>
                
                <div className="flex items-center gap-6 text-sm text-slate-600 pb-4 border-b border-slate-200/50">
                  <span className="flex items-center gap-1.5 transition-colors duration-300 group-hover:text-blue-600">
                    <Zap className="w-4 h-4" />
                    <span className="font-medium">{exercise.metValue}</span>
                    <span className="text-xs">MET</span>
                  </span>
                  <span className="flex items-center gap-1.5 transition-colors duration-300 group-hover:text-blue-600">
                    <TrendingUp className="w-4 h-4" />
                    <span className="font-medium capitalize">{exercise.type}</span>
                  </span>
                </div>
                
                <button
                  onClick={() => onSelect(exercise)}
                  className="w-full mt-4 bg-slate-900 hover:bg-blue-600 text-white font-semibold py-3.5 px-6 rounded-xl transition-all duration-500 transform group-hover:scale-[1.02] shadow-sm hover:shadow-lg"
                >
                  Start Exercise
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ExerciseSelection;