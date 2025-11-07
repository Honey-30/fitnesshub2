import { NormalizedLandmark } from '@mediapipe/pose';

type Landmark = NormalizedLandmark;

/**
 * Advanced Rep Counter with State Machine Logic
 * Uses hysteresis, debouncing, and quality scoring for maximum accuracy
 */

interface RepState {
    phase: 'up' | 'down' | 'transition_up' | 'transition_down' | 'left_down' | 'right_down' | 'holding';
    confidence: number;
    frameCount: number;
    lastTransitionTime: number;
}

interface RepQuality {
    formScore: number;
    depthScore: number;
    stabilityScore: number;
    tempoScore: number;
    overallScore: number;
}

export class RepCounter {
    private state: RepState;
    private repCount: number = 0;
    private validReps: number = 0;
    private invalidReps: number = 0;
    private repHistory: RepQuality[] = [];
    private lastRepTime: number = 0;
    private minRepDuration: number = 300; // Reduced from 400ms - allow faster reps
    private maxRepDuration: number = 15000; // Increased from 10s - allow slower beginner reps
    private transitionFrames: number = 2; // Reduced from 3 - faster response
    private hysteresisMargin: number = 5; // Degrees of hysteresis to prevent bouncing
    
    // Thresholds with hysteresis
    private thresholds: {
        [key: string]: {
            down: { enter: number; exit: number };
            up: { enter: number; exit: number };
        };
    };

    constructor(exerciseType: 'squats' | 'pushups' | 'lunges' | 'plank') {
        this.state = {
            phase: exerciseType === 'plank' ? 'holding' : 'up',
            confidence: 1.0,
            frameCount: 0,
            lastTransitionTime: Date.now(),
        };

        // Define thresholds with hysteresis for each exercise
        // Made more forgiving for beginners - partial range of motion counts!
        this.thresholds = {
            squats: {
                down: { enter: 120, exit: 130 }, // Easier - even quarter squats count
                up: { enter: 150, exit: 140 },   // Lowered from 160
            },
            pushups: {
                down: { enter: 110, exit: 120 }, // More forgiving - partial pushups count
                up: { enter: 150, exit: 140 },
            },
            lunges: {
                down: { enter: 120, exit: 130 }, // Easier for beginners
                up: { enter: 150, exit: 140 },
            },
            plank: {
                down: { enter: 0, exit: 0 },
                up: { enter: 0, exit: 0 },
            },
        };
    }

    /**
     * Update state machine with new measurements
     */
    updateState(
        angle: number,
        formScore: number,
        landmarks: Landmark[],
        exerciseType: 'squats' | 'pushups' | 'lunges' | 'plank'
    ): {
        counted: boolean;
        phase: string;
        quality: RepQuality | null;
        confidence: number;
    } {
        const now = Date.now();
        const timeSinceLastTransition = now - this.state.lastTransitionTime;
        const timeSinceLastRep = now - this.lastRepTime;

        let counted = false;
        let quality: RepQuality | null = null;

        // Plank doesn't count reps
        if (exerciseType === 'plank') {
            return {
                counted: false,
                phase: 'holding',
                quality: null,
                confidence: 1.0,
            };
        }

        const threshold = this.thresholds[exerciseType];

        // State machine with hysteresis
        if (this.state.phase === 'up' || this.state.phase === 'transition_down') {
            // Check if moving down
            if (angle < threshold.down.enter) {
                this.state.frameCount++;
                this.state.phase = 'transition_down';

                // Confirm transition after required frames
                if (this.state.frameCount >= this.transitionFrames) {
                    this.state.phase = 'down';
                    this.state.frameCount = 0;
                    this.state.lastTransitionTime = now;
                    this.state.confidence = this.calculateConfidence(angle, threshold.down.enter);
                }
            } else if (angle > threshold.down.exit && this.state.phase === 'transition_down') {
                // Bounced back up, cancel transition
                this.state.phase = 'up';
                this.state.frameCount = 0;
            }
        } else if (this.state.phase === 'down' || this.state.phase === 'transition_up') {
            // Check if moving up
            if (angle > threshold.up.enter) {
                this.state.frameCount++;
                this.state.phase = 'transition_up';

                // Confirm transition and count rep
                if (this.state.frameCount >= this.transitionFrames) {
                    // Validate rep timing
                    if (
                        timeSinceLastTransition >= this.minRepDuration &&
                        timeSinceLastTransition <= this.maxRepDuration
                    ) {
                        // Calculate rep quality
                        quality = this.calculateRepQuality(
                            formScore,
                            angle,
                            timeSinceLastTransition,
                            landmarks
                        );

                        // Count ALL reps regardless of quality - user is trying!
                        // Even beginners with low scores deserve rep counts
                        this.repCount++;
                        
                        // Track quality separately for statistics
                        if (quality.overallScore > 45) {
                            this.validReps++;
                        } else {
                            this.invalidReps++;
                        }
                        
                        counted = true;
                        this.repHistory.push(quality);
                        this.lastRepTime = now;
                    }

                    this.state.phase = 'up';
                    this.state.frameCount = 0;
                    this.state.lastTransitionTime = now;
                    this.state.confidence = this.calculateConfidence(angle, threshold.up.enter);
                }
            } else if (angle < threshold.up.exit && this.state.phase === 'transition_up') {
                // Dropped back down, cancel transition
                this.state.phase = 'down';
                this.state.frameCount = 0;
            }
        }

        return {
            counted,
            phase: this.state.phase,
            quality,
            confidence: this.state.confidence,
        };
    }

    /**
     * Calculate confidence based on how far past threshold
     */
    private calculateConfidence(angle: number, threshold: number): number {
        const margin = Math.abs(angle - threshold);
        return Math.min(1.0, 0.5 + margin / 100);
    }

    /**
     * Calculate comprehensive rep quality score
     */
    private calculateRepQuality(
        formScore: number,
        peakAngle: number,
        duration: number,
        landmarks: Landmark[]
    ): RepQuality {
        // Depth score: How deep did they go?
        const depthScore = Math.min(100, (180 - peakAngle) * 1.5);

        // Stability score: How steady were the landmarks?
        const stabilityScore = this.calculateStabilityScore(landmarks);

        // Tempo score: Was the rep too fast or too slow?
        const idealDuration = 2000; // 2 seconds ideal
        const tempoDeviation = Math.abs(duration - idealDuration) / idealDuration;
        const tempoScore = Math.max(0, 100 - tempoDeviation * 50);

        // Overall score (weighted average)
        const overallScore =
            formScore * 0.4 +
            depthScore * 0.25 +
            stabilityScore * 0.2 +
            tempoScore * 0.15;

        return {
            formScore,
            depthScore,
            stabilityScore,
            tempoScore,
            overallScore,
        };
    }

    /**
     * Calculate stability from landmark visibility and consistency
     */
    private calculateStabilityScore(landmarks: Landmark[]): number {
        const keyLandmarks = [11, 12, 23, 24, 25, 26]; // Shoulders, hips, knees
        let totalVisibility = 0;
        let count = 0;

        for (const idx of keyLandmarks) {
            if (landmarks[idx]) {
                totalVisibility += landmarks[idx].visibility || 0;
                count++;
            }
        }

        return count > 0 ? (totalVisibility / count) * 100 : 0;
    }

    /**
     * Get current rep count
     */
    getRepCount(): number {
        return this.repCount;
    }

    /**
     * Get statistics
     */
    getStats(): {
        total: number;
        valid: number;
        invalid: number;
        accuracy: number;
        averageQuality: number;
        bestQuality: number;
        consistency: number;
    } {
        const total = this.validReps + this.invalidReps;
        const accuracy = total > 0 ? (this.validReps / total) * 100 : 100;

        const avgQuality =
            this.repHistory.length > 0
                ? this.repHistory.reduce((sum, q) => sum + q.overallScore, 0) / this.repHistory.length
                : 0;

        const bestQuality =
            this.repHistory.length > 0
                ? Math.max(...this.repHistory.map((q) => q.overallScore))
                : 0;

        // Consistency: Standard deviation of quality scores
        const variance =
            this.repHistory.length > 1
                ? this.repHistory.reduce(
                      (sum, q) => sum + Math.pow(q.overallScore - avgQuality, 2),
                      0
                  ) / this.repHistory.length
                : 0;
        const consistency = Math.max(0, 100 - Math.sqrt(variance));

        return {
            total,
            valid: this.validReps,
            invalid: this.invalidReps,
            accuracy,
            averageQuality: avgQuality,
            bestQuality,
            consistency,
        };
    }

    /**
     * Get recent quality trend
     */
    getQualityTrend(lastN: number = 5): 'improving' | 'stable' | 'declining' {
        if (this.repHistory.length < lastN) {
            return 'stable';
        }

        const recent = this.repHistory.slice(-lastN);
        const firstHalf = recent.slice(0, Math.floor(lastN / 2));
        const secondHalf = recent.slice(Math.floor(lastN / 2));

        const firstAvg =
            firstHalf.reduce((sum, q) => sum + q.overallScore, 0) / firstHalf.length;
        const secondAvg =
            secondHalf.reduce((sum, q) => sum + q.overallScore, 0) / secondHalf.length;

        const diff = secondAvg - firstAvg;

        if (diff > 5) return 'improving';
        if (diff < -5) return 'declining';
        return 'stable';
    }

    /**
     * Reset counter
     */
    reset(): void {
        this.repCount = 0;
        this.validReps = 0;
        this.invalidReps = 0;
        this.repHistory = [];
        this.lastRepTime = 0;
        this.state.phase = 'up';
        this.state.frameCount = 0;
        this.state.confidence = 1.0;
    }
}
