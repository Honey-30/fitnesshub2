export enum Gender {
  Male = 'male',
  Female = 'female',
}

export enum Goal {
  Lose = 'lose_weight',
  Maintain = 'maintain',
  Gain = 'gain_muscle',
}

export enum ActivityLevel {
  Sedentary = 'sedentary',
  Light = 'light',
  Moderate = 'moderate',
  Active = 'active',
  VeryActive = 'very_active',
}

export interface UserData {
  gender: Gender;
  age: number;
  height: number; // in cm
  weight: number; // in kg
  activityLevel: ActivityLevel;
  goal: Goal;
}

export interface MacronutrientInfo {
  protein: number;
  carbs: number;
  fat: number;
}

export interface BmiInfo {
  value: number;
  category: string;
}

export interface CalculationResults {
  bmi: BmiInfo;
  tdee: number;
  idealCalories: number;
  macros: MacronutrientInfo;
  waterIntake: number; // in ml
}

// New types for Macro Tracker
export interface FoodItem {
  id: string;
  name: string;
  // per 100g serving
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
}

export interface LoggedFoodItem extends FoodItem {
  logId: string; // unique ID for the logged instance
  grams: number; 
}

// --- Workout Tracker Types ---

export interface Feedback {
    message: string; // For on-screen display
    voiceMessage: string; // For detailed voice-over
    type: 'critical' | 'adjustment' | 'positive' | 'info' | 'encouragement' | 'warning';
    problemLandmarks?: number[];
    confidence?: number; // AI confidence level
    biomechanicalRisk?: 'low' | 'medium' | 'high'; // Injury risk assessment
    priority?: 'immediate' | 'high' | 'medium' | 'low'; // Feedback timing priority
    actionableCue?: string; // Simple, actionable instruction
    detailedExplanation?: string; // Why this matters (educational)
    progressiveHint?: string; // Hint that builds on previous feedback
    encouragementLevel?: number; // 0-100, how much to encourage
}

export interface PhysicsData {
    velocity: number; // Movement velocity
    acceleration: number; // Movement acceleration
    momentum: number; // Movement momentum
    stability: number; // Joint stability score (0-100)
    powerOutput: number; // Estimated power in watts
}

export interface BiomechanicalMetrics {
    jointAngles: Record<string, number>; // All major joint angles
    rangeOfMotion: number; // Percentage of optimal ROM
    asymmetry: number; // Left-right asymmetry score
    balanceScore: number; // Overall balance (0-100)
    depthEstimate: number; // Estimated depth/distance from camera
}

export interface AnalysisResult {
    feedback: Feedback;
    stage: string;
    repCounted: boolean;
    mistake: string | null;
    formScore: number;
    physics?: PhysicsData; // Physics-based metrics
    biomechanics?: BiomechanicalMetrics; // Biomechanical analysis
    ensembleConfidence?: number; // Multi-model consensus confidence
    predictedNextStage?: string; // ML-based prediction
    improvementTrend?: 'improving' | 'stable' | 'declining'; // Performance trend
    repeatMistakeCount?: number; // How many times same mistake occurred
    shouldSpeak?: boolean; // Whether to trigger voice feedback now
    movementQuality?: { // Advanced movement quality metrics
        overall: number; // Overall quality score (0-100)
        smoothness: number; // Movement smoothness (0-100)
        control: number; // Movement control (0-100)
        efficiency: number; // Movement efficiency (0-100)
    };
    fatigueLevel?: number; // Detected fatigue (0-100)
    injuryRisk?: number; // Predicted injury risk (0-100)
}


// New types for Workout Tracker
export interface Exercise {
    id: string;
    name: string;
    description: string;
    metValue: number; // Metabolic Equivalent of Task for calorie calculation
    type: 'reps' | 'isometric';
    difficulty?: 'beginner' | 'intermediate' | 'advanced';
    muscleGroups?: string[]; // Target muscle groups
}

export interface WorkoutSession {
    exercise: Exercise;
    cleanReps: number;
    mistakes: Record<string, number>;
    durationSeconds: number;
    caloriesBurned: number;
    timeHeldSeconds?: number;
    avgFormScore: number;
    peakPower?: number; // Maximum power output
    avgVelocity?: number; // Average movement velocity
    consistencyScore?: number; // Rep-to-rep consistency
    biomechanicalEfficiency?: number; // Overall efficiency score
}

// Enhanced Landmark with temporal data
export interface EnhancedLandmark {
    x: number;
    y: number;
    z: number;
    visibility: number;
    velocity?: { x: number; y: number; z: number };
    acceleration?: { x: number; y: number; z: number };
    smoothed?: { x: number; y: number; z: number }; // Smoothed coordinates
}


// Navigation type
export type Page = 'landing' | 'calculator' | 'results' | 'food-tracker' | 'water-tracker' | 'workout' | 'api-settings';