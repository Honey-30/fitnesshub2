import React from 'react';
import { LoggedFoodItem } from '../types';
import { Trash2 } from 'lucide-react';

interface LoggedFoodListProps {
  foods: LoggedFoodItem[];
  onRemove: (logId: string) => void;
}

const macroDotColors: Record<string, string> = {
  protein: 'bg-purple-500',
  carbs: 'bg-yellow-500',
  fat: 'bg-green-500',
};

const getDominantMacro = (food: LoggedFoodItem): 'protein' | 'carbs' | 'fat' => {
  const pCals = food.protein * 4;
  const cCals = food.carbs * 4;
  const fCals = food.fat * 9;
  
  if (pCals >= cCals && pCals >= fCals) return 'protein';
  if (cCals >= pCals && cCals >= fCals) return 'carbs';
  return 'fat';
};

const LoggedFoodList: React.FC<LoggedFoodListProps> = ({ foods, onRemove }) => {
  const getMacros = (food: LoggedFoodItem) => {
    const multiplier = food.grams / 100;
    return {
      calories: Math.round(food.calories * multiplier),
      protein: Math.round(food.protein * multiplier),
      carbs: Math.round(food.carbs * multiplier),
      fat: Math.round(food.fat * multiplier),
    }
  }

  return (
    <div className="glass-card p-6 rounded-2xl flex flex-col">
      <h3 className="text-xl font-bold font-spartan text-slate-900 mb-4 flex-shrink-0">Today's Log</h3>
      {foods.length === 0 ? (
        <div className="flex-grow flex items-center justify-center min-h-[200px]">
          <p className="text-slate-500 text-center py-4">No food logged yet.</p>
        </div>
      ) : (
        <ul className="space-y-3 overflow-y-auto pr-2 flex-grow max-h-96">
          {foods.map(food => {
            const macros = getMacros(food);
            const dominantMacro = getDominantMacro(food);
            return (
              <li key={food.logId} className="bg-white/80 p-3 rounded-lg animate-fade-in-up border border-slate-200">
                <div className="flex items-start justify-between gap-4">
                    <div className="flex items-start gap-3 min-w-0">
                        <span className={`w-2 h-2 rounded-full ${macroDotColors[dominantMacro]} flex-shrink-0 mt-1.5`}></span>
                        <div className="min-w-0">
                            <p className="font-semibold text-slate-800 leading-tight truncate">{food.name}</p>
                            <p className="text-sm text-slate-500">{food.grams}g &bull; {macros.calories} kcal</p>
                        </div>
                    </div>
                     <button onClick={() => onRemove(food.logId)} className="text-slate-400 hover:text-red-500 transition-colors p-1 flex-shrink-0">
                          <Trash2 size={16} />
                      </button>
                </div>
                 <div className="grid grid-cols-3 gap-2 text-center text-xs text-slate-600 mt-2 pt-2 border-t border-slate-200">
                      <span className="font-mono"><strong className="text-purple-600">P:</strong> {macros.protein}g</span>
                      <span className="font-mono"><strong className="text-yellow-600">C:</strong> {macros.carbs}g</span>
                      <span className="font-mono"><strong className="text-green-600">F:</strong> {macros.fat}g</span>
                  </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};

export default LoggedFoodList;