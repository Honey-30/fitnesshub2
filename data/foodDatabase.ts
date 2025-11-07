import { FoodItem } from '../types';

export const foodDatabase: FoodItem[] = [
  // Proteins
  { id: '1', name: 'Chicken Breast (Cooked)', calories: 165, protein: 31, carbs: 0, fat: 3.6 },
  { id: '2', name: 'Salmon (Cooked)', calories: 206, protein: 22, carbs: 0, fat: 13 },
  { id: '3', name: 'Lean Beef (Ground, Cooked)', calories: 250, protein: 26, carbs: 0, fat: 15 },
  { id: '4', name: 'Eggs (Large)', calories: 155, protein: 13, carbs: 1.1, fat: 11 }, // Note: per 100g, not per egg
  { id: '5', name: 'Tofu (Firm)', calories: 76, protein: 8, carbs: 1.9, fat: 4.8 },
  { id: '6', name: 'Greek Yogurt (Plain, Non-fat)', calories: 59, protein: 10, carbs: 3.6, fat: 0.4 },
  { id: '7', name: 'Lentils (Cooked)', calories: 116, protein: 9, carbs: 20, fat: 0.4 },

  // Carbohydrates
  { id: '8', name: 'White Rice (Cooked)', calories: 130, protein: 2.7, carbs: 28, fat: 0.3 },
  { id: '9', name: 'Brown Rice (Cooked)', calories: 111, protein: 2.6, carbs: 23, fat: 0.9 },
  { id: '10', name: 'Quinoa (Cooked)', calories: 120, protein: 4.1, carbs: 21, fat: 1.9 },
  { id: '11', name: 'Oats (Rolled, Dry)', calories: 389, protein: 16.9, carbs: 66, fat: 6.9 },
  { id: '12', name: 'Sweet Potato (Cooked)', calories: 86, protein: 1.6, carbs: 20, fat: 0.1 },
  { id: '13', name: 'Whole Wheat Bread', calories: 265, protein: 13, carbs: 49, fat: 3.2 },
  { id: '14', name: 'Pasta (Dry)', calories: 371, protein: 13, carbs: 75, fat: 1.5 },

  // Fats
  { id: '15', name: 'Avocado', calories: 160, protein: 2, carbs: 8.5, fat: 15 },
  { id: '16', name: 'Almonds', calories: 579, protein: 21, carbs: 22, fat: 49 },
  { id: '17', name: 'Olive Oil', calories: 884, protein: 0, carbs: 0, fat: 100 },
  { id: '18', name: 'Peanut Butter', calories: 588, protein: 25, carbs: 20, fat: 50 },

  // Fruits & Vegetables
  { id: '19', name: 'Banana', calories: 89, protein: 1.1, carbs: 23, fat: 0.3 },
  { id: '20', name: 'Apple', calories: 52, protein: 0.3, carbs: 14, fat: 0.2 },
  { id: '21', name: 'Broccoli (Raw)', calories: 34, protein: 2.8, carbs: 7, fat: 0.4 },
  { id: '22', name: 'Spinach (Raw)', calories: 23, protein: 2.9, carbs: 3.6, fat: 0.4 },
  { id: '23', name: 'Tomato', calories: 18, protein: 0.9, carbs: 3.9, fat: 0.2 },
  { id: '24', name: 'Carrots (Raw)', calories: 41, protein: 0.9, carbs: 10, fat: 0.2 },
];
