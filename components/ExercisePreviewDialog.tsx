import { BookOpen, Play, X } from 'lucide-react';
import React from 'react';
import { Exercise } from '../types';

interface ExercisePreviewDialogProps {
  exercise: Exercise;
  onViewGuide: () => void;
  onStartDirectly: () => void;
  onClose: () => void;
}

const ExercisePreviewDialog: React.FC<ExercisePreviewDialogProps> = ({
  exercise,
  onViewGuide,
  onStartDirectly,
  onClose
}) => {
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fade-in">
      <div className="bg-white rounded-2xl max-w-md w-full shadow-2xl animate-pop-in">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-sky-500 text-white p-6 rounded-t-2xl relative">
          <button
            onClick={onClose}
            className="absolute top-4 right-4 w-8 h-8 rounded-full bg-white/20 hover:bg-white/30 flex items-center justify-center transition-colors"
            aria-label="Close dialog"
          >
            <X size={20} />
          </button>
          <h2 className="text-2xl font-bold font-spartan">{exercise.name}</h2>
          <p className="text-blue-100 mt-1 text-sm">{exercise.description}</p>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          <div className="text-center">
            <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-blue-100 flex items-center justify-center">
              <BookOpen className="w-10 h-10 text-blue-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-900 mb-2">
              Would you like to learn how to perform this exercise?
            </h3>
            <p className="text-slate-600 text-sm">
              Our guide will show you proper form, common mistakes to avoid, and pro tips for better results.
            </p>
          </div>

          {/* Benefits */}
          <div className="bg-blue-50 rounded-xl p-4 space-y-2">
            <p className="font-semibold text-slate-900 text-sm">The guide includes:</p>
            <ul className="space-y-1 text-sm text-slate-700">
              <li className="flex items-center gap-2">
                <span className="text-blue-600">✓</span>
                <span>Step-by-step instructions</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="text-blue-600">✓</span>
                <span>Common mistakes to avoid</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="text-blue-600">✓</span>
                <span>Expert tips for perfect form</span>
              </li>
            </ul>
          </div>

          {/* Action Buttons */}
          <div className="space-y-3">
            <button
              onClick={onViewGuide}
              className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-sky-500 hover:from-blue-500 hover:to-sky-400 text-white font-bold rounded-xl transition-all shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40 flex items-center justify-center gap-2"
            >
              <BookOpen size={20} />
              View Exercise Guide
            </button>
            <button
              onClick={onStartDirectly}
              className="w-full px-6 py-3 bg-white border-2 border-slate-300 text-slate-700 font-semibold rounded-xl hover:bg-slate-50 transition-colors flex items-center justify-center gap-2"
            >
              <Play size={20} />
              Start Without Guide
            </button>
          </div>

          <p className="text-xs text-center text-slate-500">
            💡 Tip: Watching the guide helps prevent injuries and improves results
          </p>
        </div>
      </div>
    </div>
  );
};

export default ExercisePreviewDialog;
