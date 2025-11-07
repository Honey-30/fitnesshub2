
import { ActivityLevel, BmiInfo, CalculationResults, Gender, Goal, MacronutrientInfo, UserData } from '../types';

// Calculates Body Mass Index (BMI)
const calculateBmi = (weight: number, height: number): BmiInfo => {
  const heightInMeters = height / 100;
  const bmi = parseFloat((weight / (heightInMeters * heightInMeters)).toFixed(1));
  let category = '';
  if (bmi < 18.5) category = 'Underweight';
  else if (bmi < 25) category = 'Healthy Weight';
  else if (bmi < 30) category = 'Overweight';
  else category = 'Obesity';
  return { value: bmi, category };
};

// Estimates Lean Body Mass (LBM) using the Boer formula, which is widely used when body fat percentage is unknown.
const calculateLbm = (weight: number, height: number, gender: Gender): number => {
    if (gender === Gender.Male) {
        return (0.407 * weight) + (0.267 * height) - 19.2;
    }
    // Female
    return (0.252 * weight) + (0.473 * height) - 48.3;
};

// Calculate Body Fat Percentage estimate using BMI-based formula
const estimateBodyFat = (bmi: number, age: number, gender: Gender): number => {
    if (gender === Gender.Male) {
        return (1.20 * bmi) + (0.23 * age) - 16.2;
    }
    // Female
    return (1.20 * bmi) + (0.23 * age) - 5.4;
};


// Calculates Basal Metabolic Rate (BMR) using the Katch-McArdle formula based on LBM for improved accuracy.
const calculateBmr = (data: UserData): number => {
  const lbm = calculateLbm(data.weight, data.height, data.gender);
  return 370 + (21.6 * lbm);
};

// Calculates Total Daily Energy Expenditure (TDEE)
const calculateTdee = (bmr: number, activityLevel: ActivityLevel): number => {
  const multipliers = {
    [ActivityLevel.Sedentary]: 1.2,
    [ActivityLevel.Light]: 1.375,
    [ActivityLevel.Moderate]: 1.55,
    [ActivityLevel.Active]: 1.725,
    [ActivityLevel.VeryActive]: 1.9,
  };
  return bmr * multipliers[activityLevel];
};

// Calculates ideal daily calorie intake based on goal
const calculateIdealCalories = (tdee: number, goal: Goal): number => {
  switch (goal) {
    case Goal.Lose:
      return tdee - 500; // 500 calorie deficit
    case Goal.Gain:
      return tdee + 500; // 500 calorie surplus
    case Goal.Maintain:
    default:
      return tdee;
  }
};

// Calculates macronutrient breakdown in grams based on refined, goal-specific ratios.
const calculateMacros = (calories: number, goal: Goal): MacronutrientInfo => {
  const ratios = {
    // Higher protein to preserve muscle during a deficit.
    [Goal.Lose]: { p: 0.40, c: 0.35, f: 0.25 },
    // Standard balanced approach for maintenance.
    [Goal.Maintain]: { p: 0.30, c: 0.50, f: 0.20 },
    // Higher protein for muscle synthesis, balanced with carbs for energy.
    [Goal.Gain]: { p: 0.35, c: 0.45, f: 0.20 },
  };
  const { p, c, f } = ratios[goal];
  const protein = Math.round((calories * p) / 4);
  const carbs = Math.round((calories * c) / 4);
  const fat = Math.round((calories * f) / 9);
  return { protein, carbs, fat };
};

// Calculates recommended daily water intake
const calculateWaterIntake = (weight: number): number => {
  // A common recommendation is 30-35 ml per kg of body weight
  return Math.round(weight * 35);
};

// Main function to orchestrate all calculations
export const calculateFitnessPlan = (data: UserData): CalculationResults => {
  const bmi = calculateBmi(data.weight, data.height);
  const bmr = calculateBmr(data);
  const tdee = calculateTdee(bmr, data.activityLevel);
  const idealCalories = Math.round(calculateIdealCalories(tdee, data.goal));
  const macros = calculateMacros(idealCalories, data.goal);
  const waterIntake = calculateWaterIntake(data.weight);

  return {
    bmi,
    tdee: Math.round(tdee),
    idealCalories,
    macros,
    waterIntake,
  };
};
