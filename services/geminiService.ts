import { GoogleGenAI } from "@google/genai";
import { getApiKey } from '../pages/ApiKeySettings';
import { CalculationResults, UserData, WorkoutSession } from '../types';

const getGoalDescription = (goal: string) => {
    switch(goal) {
        case 'lose_weight': return 'lose weight';
        case 'gain_muscle': return 'gain muscle and strength';
        case 'maintain': return 'maintain their current weight';
    }
    return '';
}

export const getFitnessTips = async (
  userData: UserData,
  results: CalculationResults
): Promise<string> => {
  const apiKey = getApiKey();
  
  if (!apiKey) {
    return "⚠️ **API Key Required**\n\nTo get AI-powered insights, please configure your Gemini API key in the Settings page.\n\n1. Click the Settings button in the header\n2. Enter your API key\n3. Save and return here\n\nYour personalized plan is ready and waiting!";
  }

  const ai = new GoogleGenAI({ apiKey });

  const goalDescription = getGoalDescription(userData.goal);
  
  const systemInstruction = `You are 'Atlas', an expert-level AI strength and conditioning coach. You are positive, encouraging, and precise. Your task is to write a brief, encouraging, and highly actionable summary for a user based on their personalized fitness and nutrition plan.

Structure your response in Markdown.
1.  Start with a positive and motivational opening sentence acknowledging their goal.
2.  Create a section titled "**Your Action Plan**".
3.  Under "Your Action Plan", provide 3-4 bullet points with practical tips tailored to their plan and goal.
4.  Include one specific nutrition tip.
5.  Include one specific exercise or activity tip.
6.  Include one lifestyle tip.
7.  Keep the tone professional, inspiring, and easy to understand. Do not give medical advice.`;
  
  const contents = `A user has just calculated their personalized fitness and nutrition plan.

Here is their data:
- Gender: ${userData.gender}
- Age: ${userData.age} years old
- Goal: To ${goalDescription}
- Activity Level: ${userData.activityLevel}
- BMI: ${results.bmi.value} (${results.bmi.category})

Here is their calculated daily plan:
- Calorie Target: ${results.idealCalories} kcal
- Protein: ${results.macros.protein}g
- Carbohydrates: ${results.macros.carbs}g
- Fat: ${results.macros.fat}g
- Water Intake: ${results.waterIntake}ml
`;

  try {
    const response = await ai.models.generateContent({
        model: "gemini-2.0-flash-exp",
        contents: contents,
        config: {
            systemInstruction: systemInstruction,
        }
    });
    return response.text;
  } catch (error) {
    console.error("Error calling Gemini API:", error);
    throw new Error("Could not retrieve AI-powered tips. The generative model may be unavailable.");
  }
};

export const getWorkoutAnalysis = async (session: WorkoutSession): Promise<string> => {
    const apiKey = getApiKey();
    
    if (!apiKey) {
        return "⚠️ **API Key Required**\n\nTo get AI-powered workout analysis, please configure your Gemini API key in the Settings page.\n\n1. Click the Settings button in the header\n2. Enter your API key\n3. Save and return to workout\n\nGreat job on completing your workout!";
    }

    const ai = new GoogleGenAI({ apiKey });

    const systemInstruction = `You are 'Atlas', an expert AI biomechanics coach with deep knowledge of exercise physiology, kinesiology, and injury prevention. You are positive and encouraging. A user just finished a workout with advanced motion capture analysis. Your task is to provide a brief, actionable summary of their performance in Markdown.
    
    1.  Start with an encouraging sentence about their effort and performance metrics.
    2.  Create a "**Performance Insights**" section.
    3.  Analyze their biomechanical efficiency, consistency, and form scores to identify the single biggest area for improvement.
    4.  If they had high peak power or velocity, acknowledge their explosive strength.
    5.  If their consistency score is low, focus on maintaining form throughout all reps.
    6.  Provide one simple, actionable cue with biomechanical reasoning (e.g., "focus on keeping your chest up to maintain spinal neutrality and reduce lower back strain").
    7.  End with a motivational quote about consistency, progressive overload, or movement quality.
    
    Focus on the most important fix with scientific backing.`;

    const mistakeSummary = Object.entries(session.mistakes)
        .sort(([, a], [, b]) => (b as number) - (a as number))
        .map(([mistake, count]) => `- ${mistake}: ${count} times`)
        .join('\n');
    
    const isIsometric = session.exercise.type === 'isometric';
    
    const advancedMetrics = `
${session.peakPower !== undefined ? `- Peak Power Output: ${session.peakPower}W` : ''}
${session.avgVelocity !== undefined ? `- Average Movement Velocity: ${session.avgVelocity.toFixed(3)}m/s` : ''}
${session.consistencyScore !== undefined ? `- Rep-to-Rep Consistency: ${session.consistencyScore}%` : ''}
${session.biomechanicalEfficiency !== undefined ? `- Biomechanical Efficiency: ${session.biomechanicalEfficiency}%` : ''}
    `.trim();

    const contents = `The user completed a workout session with the following results:
- Exercise: ${session.exercise.name}
${isIsometric 
    ? `- Time Held: ${session.timeHeldSeconds} seconds` 
    : `- Clean Reps: ${session.cleanReps}`
}
- Average Form Score: ${session.avgFormScore}/100 (AI Ensemble Model)
- Duration: ${session.durationSeconds} seconds
- Calories Burned: ${session.caloriesBurned} kcal

Advanced Biomechanical Metrics:
${advancedMetrics}

Mistakes Logged:
${mistakeSummary || "- No significant mistakes logged. Excellent movement quality!"}
`;
    
    try {
        const response = await ai.models.generateContent({
            model: "gemini-2.0-flash-exp",
            contents: contents,
            config: {
                systemInstruction: systemInstruction,
            }
        });
        return response.text;
    } catch (error) {
        console.error("Error calling Gemini API for workout analysis:", error);
        throw new Error("Could not retrieve AI workout analysis.");
    }
};

export const chatWithAI = async (session: WorkoutSession, messageHistory: { role: string; parts: string }[], newMessage: string): Promise<string> => {
    const apiKey = getApiKey();
    
    if (!apiKey) {
        return "⚠️ I'd love to chat, but I need an API key first! Please configure your Gemini API key in the Settings page (click the Settings button in the header). Once you've added your key, I'll be ready to answer all your fitness questions! 💪";
    }

    const ai = new GoogleGenAI({ apiKey });
    
    const mistakeSummary = Object.entries(session.mistakes)
        .map(([mistake, count]) => `${mistake} (${count} times)`)
        .join(', ');
        
    const isIsometric = session.exercise.type === 'isometric';
    
    const advancedMetrics = `
${session.peakPower !== undefined ? `Peak Power: ${session.peakPower}W, ` : ''}
${session.avgVelocity !== undefined ? `Avg Velocity: ${session.avgVelocity.toFixed(3)}m/s, ` : ''}
${session.consistencyScore !== undefined ? `Consistency: ${session.consistencyScore}%, ` : ''}
${session.biomechanicalEfficiency !== undefined ? `Bio-Efficiency: ${session.biomechanicalEfficiency}%` : ''}
    `.trim();

    const systemInstruction = `You are 'Atlas', a friendly and knowledgeable AI biomechanics and exercise physiology coach. You are having a conversation with a user who just finished a workout with advanced motion tracking.
    
    Here is the context of their completed workout:
    - Exercise: ${session.exercise.name}
    ${isIsometric 
        ? `- Time Held: ${session.timeHeldSeconds} seconds` 
        : `- Clean Reps: ${session.cleanReps}`
    }
    - Average Form Score: ${session.avgFormScore}/100 (Multi-model AI Ensemble)
    - Mistakes Made: ${mistakeSummary || "None"}
    - Duration: ${session.durationSeconds} seconds
    - Advanced Metrics: ${advancedMetrics}
    
    Answer the user's questions based on this context. Provide insights about:
    - Biomechanics and proper form with scientific reasoning
    - How to improve their metrics (power output, velocity, consistency)
    - Injury prevention and movement quality
    - Progressive overload and training principles
    
    Be concise, helpful, and maintain a positive, encouraging tone. If the question is unrelated to fitness, politely steer the conversation back to their workout performance. Do not give medical advice.`;

    try {
        const chat = ai.chats.create({
          model: 'gemini-2.0-flash-exp',
          config: { systemInstruction: systemInstruction },
        });
        
        const response = await chat.sendMessage({ message: newMessage });
        return response.text;

    } catch (error) {
        console.error("Error in AI Chat:", error);
        throw new Error("The AI chat is currently unavailable.");
    }
};