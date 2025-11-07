import React, { useMemo, useState } from 'react';
import FoodSearchAndAdd from '../components/FoodSearchAndAdd';
import LoggedFoodList from '../components/LoggedFoodList';
import MacroDistributionChart from '../components/MacroDistributionChart';
import { CalculationResults, FoodItem, LoggedFoodItem } from '../types';

interface FoodTrackerPageProps {
  targets: CalculationResults;
}

const FoodTrackerPage: React.FC<FoodTrackerPageProps> = ({ targets }) => {
  const [loggedFoods, setLoggedFoods] = useState<LoggedFoodItem[]>([]);
  const [recentFoods, setRecentFoods] = useState<FoodItem[]>([]);

  const consumed = useMemo(() => {
    return loggedFoods.reduce((acc, item) => {
        const multiplier = item.grams / 100;
        acc.calories += item.calories * multiplier;
        acc.macros.protein += item.protein * multiplier;
        acc.macros.carbs += item.carbs * multiplier;
        acc.macros.fat += item.fat * multiplier;
        return acc;
    }, { calories: 0, macros: { protein: 0, carbs: 0, fat: 0 }});
  }, [loggedFoods]);

  const handleAddFood = (food: FoodItem, grams: number) => {
    const newLoggedItem: LoggedFoodItem = {
      ...food,
      grams,
      logId: `${Date.now()}-${food.id}`,
    };
    setLoggedFoods(prev => [newLoggedItem, ...prev]);

    setRecentFoods(prevRecents => {
      const filteredRecents = prevRecents.filter(item => item.id !== food.id);
      const updatedRecents = [food, ...filteredRecents];
      return updatedRecents.slice(0, 5);
    });
  };

  const handleRemoveFood = (logId: string) => {
    setLoggedFoods(prev => prev.filter(item => item.logId !== logId));
  };
  
  const calorieProgress = targets.idealCalories > 0 ? Math.min((consumed.calories / targets.idealCalories) * 100, 100) : 0;
  const proteinProgress = targets.macros.protein > 0 ? Math.min((consumed.macros.protein / targets.macros.protein) * 100, 100) : 0;
  const carbsProgress = targets.macros.carbs > 0 ? Math.min((consumed.macros.carbs / targets.macros.carbs) * 100, 100) : 0;
  const fatProgress = targets.macros.fat > 0 ? Math.min((consumed.macros.fat / targets.macros.fat) * 100, 100) : 0;


  return (
    <div className="space-y-8 pb-12">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column: Search first, then Log */}
        <div className="space-y-6 animate-slide-in-left">
          <FoodSearchAndAdd onAddFood={handleAddFood} recentFoods={recentFoods} />
          <LoggedFoodList foods={loggedFoods} onRemove={handleRemoveFood} />
        </div>
        
        {/* Right Column: Dashboard Visualizations */}
        <div className="space-y-6 animate-slide-in-right">
            <div className="glass-card p-6 rounded-2xl space-y-6 transition-all duration-500 hover:shadow-xl">
                <h3 className="text-xl font-bold font-spartan text-slate-900">Daily Summary</h3>

                {/* Calories */}
                <div className="space-y-2 animate-fade-in-scale" style={{ animationDelay: '0.1s' }}>
                    <div className="flex flex-col items-start sm:flex-row sm:items-baseline sm:justify-between">
                    <span className="font-semibold text-blue-600">Calories</span>
                    <span className="text-lg font-bold text-slate-800">{Math.round(consumed.calories)} / {targets.idealCalories} <span className="text-sm font-normal text-slate-500">kcal</span></span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2.5 overflow-hidden">
                    <div className="bg-gradient-to-r from-blue-500 to-sky-500 h-2.5 rounded-full transition-all duration-700 ease-out" style={{ width: `${calorieProgress}%` }}></div>
                    </div>
                </div>

                {/* Macros with Mini-Progress Bars */}
                <div className="space-y-4 pt-2">
                    {/* Protein */}
                    <div className="space-y-1 animate-fade-in-scale" style={{ animationDelay: '0.2s' }}>
                        <div className="flex flex-col items-start sm:flex-row sm:items-baseline sm:justify-between text-sm">
                            <p className="font-semibold text-purple-600">Protein</p>
                            <p className="text-slate-800 font-medium">{Math.round(consumed.macros.protein)}g <span className="text-slate-500 font-normal">/ {targets.macros.protein}g</span></p>
                        </div>
                        <div className="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
                            <div className="bg-gradient-to-r from-purple-500 to-pink-500 h-1.5 rounded-full transition-all duration-700 ease-out" style={{ width: `${proteinProgress}%` }}></div>
                        </div>
                    </div>
                    {/* Carbs */}
                    <div className="space-y-1 animate-fade-in-scale" style={{ animationDelay: '0.3s' }}>
                        <div className="flex flex-col items-start sm:flex-row sm:items-baseline sm:justify-between text-sm">
                            <p className="font-semibold text-yellow-600">Carbs</p>
                            <p className="text-slate-800 font-medium">{Math.round(consumed.macros.carbs)}g <span className="text-slate-500 font-normal">/ {targets.macros.carbs}g</span></p>
                        </div>
                        <div className="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
                            <div className="bg-gradient-to-r from-yellow-500 to-amber-500 h-1.5 rounded-full transition-all duration-700 ease-out" style={{ width: `${carbsProgress}%` }}></div>
                        </div>
                    </div>
                    {/* Fat */}
                    <div className="space-y-1 animate-fade-in-scale" style={{ animationDelay: '0.4s' }}>
                        <div className="flex flex-col items-start sm:flex-row sm:items-baseline sm:justify-between text-sm">
                            <p className="font-semibold text-green-600">Fat</p>
                            <p className="text-slate-800 font-medium">{Math.round(consumed.macros.fat)}g <span className="text-slate-500 font-normal">/ {targets.macros.fat}g</span></p>
                        </div>
                        <div className="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
                            <div className="bg-gradient-to-r from-green-500 to-emerald-500 h-1.5 rounded-full transition-all duration-700 ease-out" style={{ width: `${fatProgress}%` }}></div>
                        </div>
                    </div>
                </div>
            </div>
            <div className="animate-fade-in-scale" style={{ animationDelay: '0.5s' }}>
              <MacroDistributionChart consumedMacros={consumed.macros} targetMacros={targets.macros} />
            </div>
        </div>
      </div>
    </div>
  );
};

export default FoodTrackerPage;