import { AlertCircle, CheckCircle, Play, Target, X } from 'lucide-react';
import React from 'react';
import { Exercise } from '../types';

interface ExerciseGuideModalProps {
  exercise: Exercise;
  onStart: () => void;
  onClose: () => void;
}

const exerciseGuides: Record<string, {
  steps: string[];
  commonMistakes: string[];
  tips: string[];
}> = {
  squats: {
    steps: [
      'Stand with feet shoulder-width apart, toes slightly pointed out',
      'Keep your chest up and core engaged throughout the movement',
      'Lower your body by bending knees and hips, as if sitting in a chair',
      'Go down until thighs are parallel to the ground (or lower if flexible)',
      'Push through your heels to return to starting position',
      'Keep knees aligned with toes, don\'t let them cave inward'
    ],
    commonMistakes: [
      'Knees caving inward',
      'Heels lifting off the ground',
      'Leaning too far forward',
      'Not going deep enough',
      'Rounding the lower back'
    ],
    tips: [
      'Imagine sitting back into a chair',
      'Keep weight on your heels',
      'Look straight ahead, not down',
      'Breathe in as you lower, breathe out as you rise'
    ]
  },
  pushups: {
    steps: [
      'Start in plank position with hands slightly wider than shoulders',
      'Keep your body in a straight line from head to heels',
      'Engage your core and glutes throughout the movement',
      'Lower your body by bending elbows until chest nearly touches ground',
      'Keep elbows at 45-degree angle to your body',
      'Push back up to starting position, fully extending arms'
    ],
    commonMistakes: [
      'Sagging hips or raising buttocks',
      'Flaring elbows out too wide',
      'Not going down far enough',
      'Holding breath',
      'Neck not aligned with spine'
    ],
    tips: [
      'Engage your core like doing a plank',
      'Think about pushing the floor away',
      'Keep a neutral spine throughout',
      'Modify on knees if needed for proper form'
    ]
  },
  lunges: {
    steps: [
      'Stand tall with feet hip-width apart',
      'Take a large step forward with one leg',
      'Lower your hips until both knees are bent at 90-degree angles',
      'Front knee should be directly above ankle, not pushed out past toes',
      'Back knee should hover just above the ground',
      'Push through front heel to return to starting position'
    ],
    commonMistakes: [
      'Front knee extending past toes',
      'Leaning forward too much',
      'Not lowering back knee enough',
      'Unstable balance',
      'Feet too close together'
    ],
    tips: [
      'Keep torso upright throughout',
      'Engage core for balance',
      'Push through front heel, not toes',
      'Take a bigger step if knee passes toes'
    ]
  },
  plank: {
    steps: [
      'Start on forearms and toes, elbows under shoulders',
      'Keep body in straight line from head to heels',
      'Engage core by pulling belly button toward spine',
      'Squeeze glutes to prevent sagging hips',
      'Keep neck neutral, looking at floor slightly ahead',
      'Breathe steadily and hold the position'
    ],
    commonMistakes: [
      'Hips sagging or piking up',
      'Head dropping or looking up',
      'Not engaging core',
      'Holding breath',
      'Shoulders hunched toward ears'
    ],
    tips: [
      'Think about creating a straight line',
      'Engage everything - core, glutes, quads',
      'If too hard, modify on knees',
      'Quality over duration - maintain perfect form'
    ]
  }
};

const ExerciseGuideModal: React.FC<ExerciseGuideModalProps> = ({ exercise, onStart, onClose }) => {
  const guide = exerciseGuides[exercise.id] || exerciseGuides.squats;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fade-in">
      <div className="bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl animate-pop-in">
        {/* Header */}
        <div className="sticky top-0 bg-gradient-to-r from-blue-600 to-sky-500 text-white p-6 rounded-t-2xl">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-3xl font-bold font-spartan">{exercise.name}</h2>
              <p className="text-blue-100 mt-1">Learn the proper form before you begin</p>
            </div>
            <button
              onClick={onClose}
              className="w-10 h-10 rounded-full bg-white/20 hover:bg-white/30 flex items-center justify-center transition-colors"
              aria-label="Close guide"
            >
              <X size={24} />
            </button>
          </div>
        </div>

        <div className="p-6 space-y-8">
          {/* Step-by-Step Instructions */}
          <div className="glass-card p-6 rounded-2xl">
            <h3 className="text-xl font-bold font-spartan text-slate-900 mb-4 flex items-center gap-2">
              <CheckCircle className="text-green-600" size={24} />
              Step-by-Step Guide
            </h3>
            <ol className="space-y-3">
              {guide.steps.map((step, index) => (
                <li key={index} className="flex gap-3">
                  <span className="flex-shrink-0 w-7 h-7 rounded-full bg-blue-600 text-white flex items-center justify-center text-sm font-bold">
                    {index + 1}
                  </span>
                  <span className="text-slate-700 pt-0.5">{step}</span>
                </li>
              ))}
            </ol>
          </div>

          {/* Common Mistakes & Tips Grid */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* Common Mistakes */}
            <div className="glass-card p-6 rounded-2xl border-2 border-orange-200">
              <h3 className="text-lg font-bold font-spartan text-slate-900 mb-4 flex items-center gap-2">
                <AlertCircle className="text-orange-600" size={24} />
                Common Mistakes to Avoid
              </h3>
              <ul className="space-y-2">
                {guide.commonMistakes.map((mistake, index) => (
                  <li key={index} className="flex gap-2 text-sm text-slate-700">
                    <span className="text-orange-600">✗</span>
                    <span>{mistake}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Pro Tips */}
            <div className="glass-card p-6 rounded-2xl border-2 border-green-200">
              <h3 className="text-lg font-bold font-spartan text-slate-900 mb-4 flex items-center gap-2">
                <Target className="text-green-600" size={24} />
                Pro Tips
              </h3>
              <ul className="space-y-2">
                {guide.tips.map((tip, index) => (
                  <li key={index} className="flex gap-2 text-sm text-slate-700">
                    <span className="text-green-600">✓</span>
                    <span>{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4">
            <button
              onClick={onClose}
              className="flex-1 px-6 py-3 bg-white border-2 border-slate-300 text-slate-700 font-semibold rounded-xl hover:bg-slate-50 transition-colors"
            >
              Review Again
            </button>
            <button
              onClick={onStart}
              className="flex-1 px-6 py-3 bg-gradient-to-r from-blue-600 to-sky-500 hover:from-blue-500 hover:to-sky-400 text-white font-bold rounded-xl transition-all shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40 flex items-center justify-center gap-2"
            >
              <Play size={20} />
              Start Workout
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExerciseGuideModal;
