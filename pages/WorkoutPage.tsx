import React, { useState } from 'react';
import ExerciseSelection from '../components/ExerciseSelection';
import PoseCanvas from '../components/PoseCanvas';
import WorkoutSummary from '../components/WorkoutSummary';
import { Exercise, WorkoutSession } from '../types';

const availableExercises: Exercise[] = [
    { 
        id: 'squats', 
        name: 'Squats', 
        description: 'A fundamental compound exercise that targets your thighs, hips, and glutes with advanced biomechanical analysis.', 
        metValue: 5.0, 
        type: 'reps',
        difficulty: 'beginner',
        muscleGroups: ['Quadriceps', 'Glutes', 'Hamstrings', 'Core']
    },
    { 
        id: 'pushups', 
        name: 'Push-ups', 
        description: 'A classic bodyweight exercise that builds upper body strength with real-time elbow tracking and core stability monitoring.', 
        metValue: 8.0, 
        type: 'reps',
        difficulty: 'beginner',
        muscleGroups: ['Chest', 'Shoulders', 'Triceps', 'Core']
    },
    { 
        id: 'lunges', 
        name: 'Lunges', 
        description: 'An excellent exercise for targeting the quadriceps, glutes, and hamstrings with balance analysis and knee safety tracking.', 
        metValue: 3.8, 
        type: 'reps',
        difficulty: 'intermediate',
        muscleGroups: ['Quadriceps', 'Glutes', 'Hamstrings', 'Calves']
    },
    { 
        id: 'plank', 
        name: 'Plank', 
        description: 'A core-strengthening isometric exercise with advanced stability scoring and spinal alignment monitoring.', 
        metValue: 3.0, 
        type: 'isometric',
        difficulty: 'beginner',
        muscleGroups: ['Core', 'Shoulders', 'Glutes']
    },
];

interface WorkoutPageProps {
    userData: { weight: number }; // User weight in kg for calorie calculation
}

const WorkoutPage: React.FC<WorkoutPageProps> = ({ userData }) => {
    const [workoutState, setWorkoutState] = useState<'selecting' | 'active' | 'summary'>('selecting');
    const [selectedExercise, setSelectedExercise] = useState<Exercise | null>(null);
    const [sessionSummary, setSessionSummary] = useState<WorkoutSession | null>(null);

    const handleSelectExercise = (exercise: Exercise) => {
        // --- CRITICAL FIX for Voice Feedback ---
        // Most browsers block speech synthesis until a user interaction.
        // This silent utterance on the *first* user click "unlocks" the audio context,
        // allowing the PoseCanvas to speak feedback later without being blocked.
        try {
            const silentUtterance = new SpeechSynthesisUtterance("");
            silentUtterance.volume = 0;
            speechSynthesis.speak(silentUtterance);
        } catch (e) {
            console.error("Could not initialize speech synthesis:", e);
        }
        // --- END OF FIX ---

        setSelectedExercise(exercise);
        setWorkoutState('active');
    };

    const handleWorkoutComplete = (session: WorkoutSession) => {
        setSessionSummary(session);
        setWorkoutState('summary');
    };

    const handleRestart = () => {
        setSelectedExercise(null);
        setSessionSummary(null);
        setWorkoutState('selecting');
    };

    const renderContent = () => {
        switch (workoutState) {
            case 'selecting':
                return <ExerciseSelection exercises={availableExercises} onSelect={handleSelectExercise} />;
            case 'active':
                if (selectedExercise) {
                    return <PoseCanvas exercise={selectedExercise} onWorkoutComplete={handleWorkoutComplete} userWeight={userData.weight} />;
                }
                return <p>Error: No exercise selected.</p>;
            case 'summary':
                if (sessionSummary) {
                    return <WorkoutSummary session={sessionSummary} onRestart={handleRestart} />;
                }
                return <p>Error: No session summary available.</p>;
            default:
                return null;
        }
    };

    return (
        <div>
            {renderContent()}
        </div>
    );
};

export default WorkoutPage;