import { History, PlusCircle, Scale, Search } from 'lucide-react';
import React, { useMemo, useState } from 'react';
import { foodDatabase } from '../data/foodDatabase';
import { FoodItem } from '../types';

interface FoodSearchAndAddProps {
  onAddFood: (food: FoodItem, grams: number) => void;
  recentFoods: FoodItem[];
}

const FoodSearchAndAdd: React.FC<FoodSearchAndAddProps> = ({ onAddFood, recentFoods }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedFood, setSelectedFood] = useState<FoodItem | null>(null);
  const [grams, setGrams] = useState(100);

  const searchResults = useMemo(() => {
    if (!searchTerm.trim()) return [];
    const lowerCaseSearchTerm = searchTerm.trim().toLowerCase();
    return foodDatabase
      .map(food => {
          const lowerCaseFoodName = food.name.toLowerCase();
          let score = 0;
          if (lowerCaseFoodName.startsWith(lowerCaseSearchTerm)) {
              score = 100;
          } else if (lowerCaseFoodName.includes(lowerCaseSearchTerm)) {
              score = 50;
          }
          
          const highlightedName = food.name.replace(
              new RegExp(`(${searchTerm.trim()})`, 'gi'),
              '<strong class="font-bold text-blue-600">$1</strong>'
          );

          return { ...food, score, highlightedName };
      })
      .filter(item => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 5);
  }, [searchTerm]);

  const handleSelectFood = (food: FoodItem) => {
    setSelectedFood(food);
    setSearchTerm(''); 
    setGrams(100);
  };
  
  const handleAddClick = () => {
    if (selectedFood && grams > 0) {
      onAddFood(selectedFood, grams);
      setSelectedFood(null); 
      setGrams(100);
    }
  };

  return (
    <div className="glass-card p-5 rounded-2xl space-y-3">
      <h3 className="text-lg font-bold font-spartan text-slate-900">Add Food</h3>
      
      {/* Recents Section - More Compact */}
      {recentFoods.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-slate-500 flex items-center gap-1.5">
            <History size={14}/> Recents
          </h4>
          <div className="flex flex-wrap gap-1.5">
            {recentFoods.map(food => (
              <button 
                key={food.id} 
                onClick={() => handleSelectFood(food)}
                className="bg-slate-200 hover:bg-slate-300 text-slate-700 text-xs py-1 px-2.5 rounded-full transition-colors"
              >
                {food.name}
              </button>
            ))}
          </div>
        </div>
      )}
      
      {/* Search Input */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
        <input
          type="text"
          placeholder="Search for a food..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full bg-white border border-slate-300 rounded-lg p-2.5 pl-10 text-sm text-slate-800 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition duration-200"
        />
        {searchResults.length > 0 && (
          <ul className="absolute z-10 w-full bg-white border border-slate-300 rounded-lg mt-1 max-h-48 overflow-y-auto shadow-lg">
            {searchResults.map(food => (
              <li
                key={food.id}
                onClick={() => handleSelectFood(food)}
                className="p-2.5 hover:bg-slate-100 cursor-pointer text-sm text-slate-800"
                dangerouslySetInnerHTML={{ __html: food.highlightedName }}
              >
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Add Food Section - More Compact */}
      {selectedFood && (
        <div className="bg-slate-100/50 p-3 rounded-lg space-y-2 animate-pop-in">
          <p className="font-bold text-sm text-slate-800">{selectedFood.name}</p>
          <div className="text-xs text-slate-500 flex flex-wrap justify-between gap-x-2">
            <span>{selectedFood.calories} kcal</span>
            <span>P: {selectedFood.protein}g</span>
            <span>C: {selectedFood.carbs}g</span>
            <span>F: {selectedFood.fat}g</span>
          </div>
          <div className="flex items-center gap-3 pt-1">
            <div className="relative flex-grow">
               <Scale className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
               <input
                type="number"
                value={grams}
                onChange={(e) => setGrams(Number(e.target.value))}
                className="w-full bg-white border border-slate-300 rounded-lg p-2 pl-9 pr-14 text-sm text-slate-800 text-center"
                min="1"
              />
               <span className="absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-slate-400">grams</span>
            </div>
            <button 
              onClick={handleAddClick} 
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold p-2 rounded-lg transition duration-300"
            >
                <PlusCircle size={20} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FoodSearchAndAdd;