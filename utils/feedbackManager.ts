import { Feedback } from '../types';

// Advanced feedback manager with timing intelligence
export class FeedbackManager {
    private mistakeHistory: Map<string, number> = new Map();
    private lastFeedbackTime: number = 0;
    private lastMistakeFeedbackTime: Map<string, number> = new Map(); // Track per-mistake timing
    private feedbackQueue: Array<{ feedback: Feedback; timestamp: number }> = [];
    private consecutiveGoodReps: number = 0;
    private totalReps: number = 0;
    private learningPhase: boolean = true;

    // Intelligent timing based on feedback priority and context
    shouldDeliverFeedback(feedback: Feedback, currentTime: number): boolean {
        const timeSinceLastFeedback = currentTime - this.lastFeedbackTime;
        
        // Immediate priority - always deliver
        if (feedback.priority === 'immediate' || feedback.type === 'critical') {
            this.lastFeedbackTime = currentTime;
            return true;
        }
        
        // High priority - deliver if >2 seconds since last
        if (feedback.priority === 'high' && timeSinceLastFeedback > 2000) {
            this.lastFeedbackTime = currentTime;
            return true;
        }
        
        // Medium priority - deliver if >4 seconds since last
        if (feedback.priority === 'medium' && timeSinceLastFeedback > 4000) {
            this.lastFeedbackTime = currentTime;
            return true;
        }
        
        // Low priority - deliver if >6 seconds since last
        if (feedback.priority === 'low' && timeSinceLastFeedback > 6000) {
            this.lastFeedbackTime = currentTime;
            return true;
        }
        
        // Positive feedback - always encourage, but throttle
        if (feedback.type === 'positive' && timeSinceLastFeedback > 1500) {
            this.lastFeedbackTime = currentTime;
            return true;
        }
        
        return false;
    }
    
    // Check if we should repeat feedback for persistent mistakes (after 12 seconds)
    shouldRepeatMistakeFeedback(mistake: string, currentTime: number): boolean {
        const lastTime = this.lastMistakeFeedbackTime.get(mistake) || 0;
        const timeSince = currentTime - lastTime;
        
        // Repeat detailed feedback every 12 seconds for same mistake
        if (timeSince > 12000) {
            this.lastMistakeFeedbackTime.set(mistake, currentTime);
            return true;
        }
        
        return false;
    }

    // Track mistake patterns and provide progressive hints
    recordMistake(mistake: string): number {
        const count = (this.mistakeHistory.get(mistake) || 0) + 1;
        this.mistakeHistory.set(mistake, count);
        return count;
    }

    // Get progressive hint based on mistake count
    getProgressiveHint(mistake: string, count: number): string {
        const hints: Record<string, string[]> = {
            'Chest Falling': [
                'Try to keep your chest proud and eyes forward.',
                'Imagine pushing your chest through a doorway in front of you.',
                'Engage your upper back muscles to maintain an upright torso.',
                'Think about leading with your chest as you rise.'
            ],
            'Knee Valgus': [
                'Push your knees outward, tracking over your toes.',
                'Think about spreading the floor apart with your feet.',
                'Engage your glutes to keep knees stable and outward.',
                'Imagine a resistance band around your knees pushing them in - fight it!'
            ],
            'Not Deep Enough': [
                'Try to sit back a bit more, like sitting into a chair.',
                'Lower until your hip crease is below your knee.',
                'Focus on controlled descent - depth comes with practice.',
                'Keep weight on your heels and sink deeper into the movement.'
            ],
            'Hip Sag': [
                'Engage your core - pull your belly button toward your spine.',
                'Squeeze your glutes to lift your hips into alignment.',
                'Think about creating a straight line from head to heels.',
                'Breathe steadily and maintain core tension throughout.'
            ],
            'Unstable Knees': [
                'Slow down your movement to improve control.',
                'Engage your quadriceps and focus on balance.',
                'Keep your core tight to stabilize your entire body.',
                'Try a narrower stance or adjust your foot position.'
            ],
            'Knee Over Toes': [
                'Shift your weight back onto your heels.',
                'Keep your shin more vertical - your knee should stay behind your toes.',
                'Think about sitting back into the lunge rather than forward.',
                'Focus on dropping your back knee down, not pushing your front knee forward.'
            ],
            'Elbows Flaring': [
                'Keep your elbows at a 45-degree angle from your body.',
                'Imagine tucking your elbows into your sides as you descend.',
                'This protects your shoulders - keep elbows closer to your ribs.',
                'Think "elbows back" not "elbows out" throughout the movement.'
            ]
        };

        const mistakeHints = hints[mistake] || ['Focus on your form and control.'];
        const index = Math.min(count - 1, mistakeHints.length - 1);
        return mistakeHints[index];
    }

    // Enhanced feedback with learning progression
    enhanceFeedback(baseFeedback: Feedback, mistake: string | null, repCount: number): Feedback {
        this.totalReps = repCount;
        
        // Exit learning phase after 3 good reps
        if (baseFeedback.type === 'positive') {
            this.consecutiveGoodReps++;
            if (this.consecutiveGoodReps >= 3) {
                this.learningPhase = false;
            }
        } else {
            this.consecutiveGoodReps = 0;
        }

        // If there's a mistake, get progressive hint
        if (mistake) {
            const mistakeCount = this.recordMistake(mistake);
            const progressiveHint = this.getProgressiveHint(mistake, mistakeCount);
            
            return {
                ...baseFeedback,
                progressiveHint,
                priority: mistakeCount > 2 ? 'immediate' : 'high',
                encouragementLevel: Math.max(0, 100 - mistakeCount * 15),
                actionableCue: this.getActionableCue(mistake),
                detailedExplanation: this.getDetailedExplanation(mistake)
            };
        }

        // Enhance positive feedback
        if (baseFeedback.type === 'positive') {
            const encouragementMessage = this.getEncouragementMessage(repCount);
            return {
                ...baseFeedback,
                voiceMessage: encouragementMessage,
                priority: 'medium',
                encouragementLevel: 100
            };
        }

        return {
            ...baseFeedback,
            priority: baseFeedback.type === 'critical' ? 'immediate' : 'medium'
        };
    }

    // Get simple actionable cue
    private getActionableCue(mistake: string): string {
        const cues: Record<string, string> = {
            'Chest Falling': 'Chest up, eyes forward',
            'Knee Valgus': 'Knees out, track over toes',
            'Not Deep Enough': 'Sit back deeper',
            'Hip Sag': 'Engage core, squeeze glutes',
            'Unstable Knees': 'Slow down, control movement',
            'Knee Over Toes': 'Weight on heels, shift back',
            'Elbows Flaring': 'Elbows at 45°, tuck in',
            'Lower Chest More': 'Chest closer to floor',
            'Poor Balance': 'Core tight, find center',
            'Incorrect Depth': 'Aim for 90° angles',
            'Hips Too High': 'Lower hips to alignment',
            'Hips Too Low': 'Lift hips, engage core',
            'Unstable Core': 'Brace abs, breathe steady'
        };
        return cues[mistake] || 'Focus on control';
    }

    // Get detailed explanation for education
    private getDetailedExplanation(mistake: string): string {
        const explanations: Record<string, string> = {
            'Chest Falling': 'Maintaining an upright chest keeps your spine neutral, distributing load safely across your back and preventing lower back strain. This protects your vertebrae and ensures proper muscle engagement.',
            'Knee Valgus': 'Knees caving inward puts dangerous stress on your ACL and meniscus. Keeping knees out protects your joint and activates your glutes properly. This is critical for knee health and preventing serious injury.',
            'Not Deep Enough': 'Full range of motion activates more muscle fibers and builds strength through the complete movement pattern. Going deeper (safely) increases muscle growth and functional strength.',
            'Hip Sag': 'A sagging hip position puts excessive stress on your lower back and reduces core engagement. This can lead to lower back pain and reduces the effectiveness of the exercise.',
            'Unstable Knees': 'Knee wobbling indicates weak stabilizer muscles. Control builds strength and prevents injury. Stable knees mean safer, more effective training.',
            'Knee Over Toes': 'When your knee travels past your toes, it increases shear force on your knee joint and patellar tendon. This can cause knee pain and long-term damage. Keep your shin vertical.',
            'Elbows Flaring': 'Wide elbows stress your shoulder joint and rotator cuff. A 45° angle is biomechanically optimal, protecting your shoulders while maximizing chest and tricep activation.',
            'Lower Chest More': 'Not lowering enough reduces the exercise effectiveness. A full range of motion builds more strength and muscle. Aim to get your chest close to the ground.',
            'Poor Balance': 'Balance issues indicate weak core stability, which is essential for all compound movements. A strong core prevents injury and improves performance in all exercises.',
            'Hips Too High': 'High hips reduce core activation and shift work away from your abdominals. This makes the plank less effective and can strain your shoulders.',
            'Unstable Core': 'Core instability reduces movement efficiency and increases injury risk to your spine. A stable core is the foundation of all movement.',
            'Incorrect Depth': 'Proper depth (90° angles) ensures full muscle activation and joint health. Too shallow reduces benefits; too deep can cause strain if you lack mobility.'
        };
        return explanations[mistake] || 'Proper form prevents injury and maximizes results. Focus on quality movement patterns.';
    }
    
    // Get comprehensive repeated feedback for persistent mistakes
    getRepeatedMistakeFeedback(mistake: string, count: number): string {
        const hint = this.getProgressiveHint(mistake, count);
        const explanation = this.getDetailedExplanation(mistake);
        const cue = this.getActionableCue(mistake);
        
        return `You've had ${count} instances of "${mistake}". Let's fix this: ${cue}. ${hint} WHY THIS MATTERS: ${explanation} Take a moment to reset and focus on this correction.`;
    }

    // Get encouraging message based on progress
    getEncouragementMessage(repCount: number): string {
        const messages = [
            'Excellent form! That\'s how it\'s done!',
            `${repCount} perfect reps - you're crushing it!`,
            'Beautiful technique! Keep this consistency!',
            'That\'s textbook form! Outstanding!',
            `${repCount} clean reps and counting - incredible work!`,
            'Your form is spot-on! This is elite-level execution!',
            'Perfect alignment! You\'re moving like a pro!',
            `Rep ${repCount} - absolutely flawless! Keep it up!`
        ];
        
        if (repCount === 0) return messages[0];
        if (repCount >= 10) return messages[5];
        if (repCount >= 5) return messages[4];
        
        return messages[repCount % messages.length];
    }

    // Get current learning progress
    getProgress(): { learningPhase: boolean; consecutiveGoodReps: number; totalMistakes: number } {
        const totalMistakes = Array.from(this.mistakeHistory.values()).reduce((a, b) => a + b, 0);
        return {
            learningPhase: this.learningPhase,
            consecutiveGoodReps: this.consecutiveGoodReps,
            totalMistakes
        };
    }

    // Reset for new workout
    reset(): void {
        this.mistakeHistory.clear();
        this.lastFeedbackTime = 0;
        this.feedbackQueue = [];
        this.consecutiveGoodReps = 0;
        this.totalReps = 0;
        this.learningPhase = true;
    }
}

// Singleton instance
export const feedbackManager = new FeedbackManager();
