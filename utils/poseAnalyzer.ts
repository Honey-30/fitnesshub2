import { NormalizedLandmark } from '@mediapipe/pose';
import { AnalysisResult, BiomechanicalMetrics, Feedback, PhysicsData } from '../types';
import { feedbackManager } from './feedbackManager';

type Landmark = NormalizedLandmark;

// Performance history for trend analysis
let formScoreHistory: number[] = [];
let mistakePattern: Map<string, number[]> = new Map(); // Track when mistakes occur

// Advanced ML-based performance tracking
interface UserPerformanceProfile {
    baselineRangeOfMotion: Map<string, number>; // Exercise -> ROM
    fatigueThreshold: number; // When performance drops significantly
    learningRate: number; // How quickly user improves
    injuryRiskFactors: Set<string>; // Tracked risk patterns
    optimalTempoRange: { min: number; max: number }; // Best speed for this user
    strengthProfile: Map<string, number>; // Muscle group strength estimates
}

const userProfiles: Map<string, UserPerformanceProfile> = new Map();

/**
 * Adaptive Learning System
 * Learns user's baseline performance and adjusts thresholds
 */
class AdaptiveLearningSystem {
    private repData: Array<{ formScore: number; rom: number; tempo: number; timestamp: number }> = [];
    private readonly LEARNING_WINDOW = 30; // Last 30 reps
    
    recordRep(formScore: number, rom: number, tempo: number) {
        this.repData.push({
            formScore,
            rom,
            tempo,
            timestamp: Date.now()
        });
        
        if (this.repData.length > this.LEARNING_WINDOW) {
            this.repData.shift();
        }
    }
    
    getAdaptiveThreshold(metric: 'formScore' | 'rom' | 'tempo'): number {
        if (this.repData.length < 5) {
            // Default thresholds for beginners
            return metric === 'formScore' ? 70 : metric === 'rom' ? 90 : 2.0;
        }
        
        const values = this.repData.map(r => r[metric]);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const stdDev = Math.sqrt(
            values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length
        );
        
        // Adaptive threshold: mean - 0.5 * stdDev (allows some variance)
        return mean - (0.5 * stdDev);
    }
    
    detectFatigue(): { isFatigued: boolean; severity: number } {
        if (this.repData.length < 10) {
            return { isFatigued: false, severity: 0 };
        }
        
        // Compare recent 5 reps vs previous 5 reps
        const recent = this.repData.slice(-5);
        const previous = this.repData.slice(-10, -5);
        
        const recentAvg = recent.reduce((sum, r) => sum + r.formScore, 0) / 5;
        const previousAvg = previous.reduce((sum, r) => sum + r.formScore, 0) / 5;
        
        const performanceDrop = previousAvg - recentAvg;
        
        return {
            isFatigued: performanceDrop > 15, // 15+ point drop indicates fatigue
            severity: Math.min(100, Math.max(0, performanceDrop * 5))
        };
    }
    
    predictInjuryRisk(currentRom: number, currentTempo: number): number {
        if (this.repData.length < 10) return 0;
        
        const avgRom = this.repData.reduce((sum, r) => sum + r.rom, 0) / this.repData.length;
        const avgTempo = this.repData.reduce((sum, r) => sum + r.tempo, 0) / this.repData.length;
        
        let riskScore = 0;
        
        // Sudden ROM increase = injury risk
        if (currentRom > avgRom * 1.2) {
            riskScore += 30;
        }
        
        // Too fast tempo = injury risk
        if (currentTempo < avgTempo * 0.6) {
            riskScore += 25;
        }
        
        // Check fatigue
        const { isFatigued, severity } = this.detectFatigue();
        if (isFatigued) {
            riskScore += severity * 0.3;
        }
        
        return Math.min(100, riskScore);
    }
}

const adaptiveSystem = new AdaptiveLearningSystem();

/**
 * Movement Quality Assessment System
 * Multi-dimensional quality scoring beyond simple form
 */
class MovementQualityAssessor {
    assessMovementQuality(
        landmarks: Landmark[],
        formScore: number,
        physics: PhysicsData,
        biomechanics: BiomechanicalMetrics
    ): {
        overallQuality: number;
        smoothness: number;
        control: number;
        efficiency: number;
        details: string[];
    } {
        // Smoothness: Low jitter + consistent velocity
        const smoothness = this.calculateSmoothness(physics);
        
        // Control: Ability to maintain stable positions
        const control = this.calculateControl(physics, biomechanics);
        
        // Efficiency: Optimal power output for movement
        const efficiency = this.calculateEfficiency(physics, biomechanics);
        
        // Overall quality: Weighted combination
        const overallQuality = (
            formScore * 0.4 +
            smoothness * 0.25 +
            control * 0.2 +
            efficiency * 0.15
        );
        
        const details: string[] = [];
        if (smoothness < 70) details.push("Movement is jerky - focus on smooth, controlled motion");
        if (control < 70) details.push("Improve stability - engage core and maintain balance");
        if (efficiency < 70) details.push("Using excess energy - relax and find optimal path");
        
        return { overallQuality, smoothness, control, efficiency, details };
    }
    
    private calculateSmoothness(physics: PhysicsData): number {
        // High stability + moderate velocity = smooth movement
        const velocityScore = Math.max(0, 100 - Math.abs(physics.velocity - 0.5) * 100);
        return (physics.stability * 0.6 + velocityScore * 0.4);
    }
    
    private calculateControl(physics: PhysicsData, biomechanics: BiomechanicalMetrics): number {
        // Balance + low asymmetry = good control
        return (biomechanics.balanceScore * 0.7 + (100 - biomechanics.asymmetry * 2) * 0.3);
    }
    
    private calculateEfficiency(physics: PhysicsData, biomechanics: BiomechanicalMetrics): number {
        // Power output relative to movement quality
        const powerEfficiency = Math.min(100, physics.powerOutput / 5); // Normalize
        return powerEfficiency * 0.5 + biomechanics.balanceScore * 0.5;
    }
}

const qualityAssessor = new MovementQualityAssessor();

// Temporal smoothing buffer for landmark positions
/**
 * Advanced Landmark Buffer with Kalman-like filtering
 * Provides temporal smoothing, outlier rejection, and predictive tracking
 */
class LandmarkBuffer {
    private buffer: Landmark[][] = [];
    private maxSize: number = 10; // Increased from 5 for better smoothing
    private velocityBuffer: Map<number, { x: number; y: number; z: number }[]> = new Map();
    private confidenceThreshold: number = 0.5;
    
    add(landmarks: Landmark[]) {
        // Outlier rejection: Check if landmarks are drastically different
        if (this.buffer.length > 0) {
            const lastLandmarks = this.buffer[this.buffer.length - 1];
            const maxMovement = 0.15; // Maximum 15% movement between frames
            
            let isOutlier = false;
            for (let i = 0; i < Math.min(landmarks.length, lastLandmarks.length); i++) {
                const dist = Math.sqrt(
                    Math.pow(landmarks[i].x - lastLandmarks[i].x, 2) +
                    Math.pow(landmarks[i].y - lastLandmarks[i].y, 2)
                );
                
                if (dist > maxMovement && landmarks[i].visibility > this.confidenceThreshold) {
                    isOutlier = true;
                    break;
                }
            }
            
            // Skip outlier frames (likely detection errors)
            if (isOutlier) {
                return;
            }
        }
        
        this.buffer.push(landmarks);
        if (this.buffer.length > this.maxSize) {
            this.buffer.shift();
        }
    }
    
    /**
     * Get smoothed landmark with Gaussian-weighted average
     */
    getSmoothed(index: number): Landmark | null {
        if (this.buffer.length === 0) return null;
        
        // Gaussian weights for more recent frames
        const weights = this.getGaussianWeights(this.buffer.length);
        let totalWeight = 0;
        let sumX = 0, sumY = 0, sumZ = 0, sumVis = 0;
        
        for (let i = 0; i < this.buffer.length; i++) {
            const landmark = this.buffer[i][index];
            if (landmark && landmark.visibility > this.confidenceThreshold) {
                const weight = weights[i];
                sumX += landmark.x * weight;
                sumY += landmark.y * weight;
                sumZ += landmark.z * weight;
                sumVis += landmark.visibility * weight;
                totalWeight += weight;
            }
        }
        
        if (totalWeight === 0) return this.buffer[this.buffer.length - 1][index];
        
        return {
            x: sumX / totalWeight,
            y: sumY / totalWeight,
            z: sumZ / totalWeight,
            visibility: sumVis / totalWeight
        };
    }
    
    /**
     * Generate Gaussian weights (more weight to recent frames)
     */
    private getGaussianWeights(length: number): number[] {
        const weights: number[] = [];
        const sigma = length / 3; // Standard deviation
        const mean = length - 1; // Peak at most recent
        
        for (let i = 0; i < length; i++) {
            const exponent = -Math.pow(i - mean, 2) / (2 * Math.pow(sigma, 2));
            weights.push(Math.exp(exponent));
        }
        
        return weights;
    }
    
    /**
     * Get velocity with sub-frame precision
     */
    getVelocity(index: number): { x: number; y: number; z: number } | null {
        if (this.buffer.length < 3) return null;
        
        const current = this.buffer[this.buffer.length - 1][index];
        const previous = this.buffer[this.buffer.length - 3][index];
        
        if (!current || !previous) return null;
        
        // Calculate velocity (assuming 30 FPS)
        const deltaTime = 2 / 30; // 2 frames at 30fps
        const velocity = {
            x: (current.x - previous.x) / deltaTime,
            y: (current.y - previous.y) / deltaTime,
            z: (current.z - previous.z) / deltaTime
        };
        
        // Store in velocity buffer for acceleration calculation
        if (!this.velocityBuffer.has(index)) {
            this.velocityBuffer.set(index, []);
        }
        const velBuffer = this.velocityBuffer.get(index)!;
        velBuffer.push(velocity);
        if (velBuffer.length > 5) {
            velBuffer.shift();
        }
        
        return velocity;
    }
    
    /**
     * Get acceleration (derivative of velocity)
     */
    getAcceleration(index: number): { x: number; y: number; z: number } | null {
        const velBuffer = this.velocityBuffer.get(index);
        if (!velBuffer || velBuffer.length < 2) return null;
        
        const currentVel = velBuffer[velBuffer.length - 1];
        const previousVel = velBuffer[velBuffer.length - 2];
        const deltaTime = 1 / 30; // 1 frame at 30fps
        
        return {
            x: (currentVel.x - previousVel.x) / deltaTime,
            y: (currentVel.y - previousVel.y) / deltaTime,
            z: (currentVel.z - previousVel.z) / deltaTime
        };
    }
    
    /**
     * Predict next position using velocity
     */
    predict(index: number): Landmark | null {
        const current = this.getSmoothed(index);
        const velocity = this.getVelocity(index);
        
        if (!current || !velocity) return current;
        
        const deltaTime = 1 / 30; // Next frame
        
        return {
            x: current.x + velocity.x * deltaTime,
            y: current.y + velocity.y * deltaTime,
            z: current.z + velocity.z * deltaTime,
            visibility: current.visibility
        };
    }
    
    /**
     * Get jitter amount (stability metric)
     */
    getJitter(index: number): number {
        if (this.buffer.length < 2) return 0;
        
        let totalJitter = 0;
        for (let i = 1; i < this.buffer.length; i++) {
            const curr = this.buffer[i][index];
            const prev = this.buffer[i - 1][index];
            
            if (curr && prev) {
                const dist = Math.sqrt(
                    Math.pow(curr.x - prev.x, 2) +
                    Math.pow(curr.y - prev.y, 2)
                );
                totalJitter += dist;
            }
        }
        
        return totalJitter / (this.buffer.length - 1);
    }
}

const landmarkBuffer = new LandmarkBuffer();

// Enhanced angle calculation with 3D depth consideration
export const calculateAngle = (a: Landmark, b: Landmark, c: Landmark): number => {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs(radians * 180.0 / Math.PI);
  if (angle > 180.0) {
    angle = 360 - angle;
  }
  return angle;
};

// Calculate 3D angle using depth information
const calculate3DAngle = (a: Landmark, b: Landmark, c: Landmark): number => {
    const ba = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
    const bc = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
    
    const dotProduct = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
    const magnitudeBA = Math.sqrt(ba.x ** 2 + ba.y ** 2 + ba.z ** 2);
    const magnitudeBC = Math.sqrt(bc.x ** 2 + bc.y ** 2 + bc.z ** 2);
    
    const cosAngle = dotProduct / (magnitudeBA * magnitudeBC);
    return Math.acos(Math.max(-1, Math.min(1, cosAngle))) * (180 / Math.PI);
};

// Calculate depth estimate from camera
const estimateDepth = (landmarks: Landmark[]): number => {
    // Use shoulder width as reference (average ~40cm)
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];
    const shoulderDistance = Math.sqrt(
        (rightShoulder.x - leftShoulder.x) ** 2 + 
        (rightShoulder.y - leftShoulder.y) ** 2
    );
    
    // Normalize depth (larger shoulder distance = closer to camera)
    return Math.max(0.5, Math.min(2.0, 0.15 / shoulderDistance));
};

// Calculate joint stability score using advanced metrics
const calculateStability = (landmark: Landmark, landmarkIndex: number): number => {
    // Use jitter as primary stability metric
    const jitter = landmarkBuffer.getJitter(landmarkIndex);
    const velocity = landmarkBuffer.getVelocity(landmarkIndex);
    
    // Jitter score (lower jitter = higher stability)
    const jitterScore = Math.max(0, 100 - jitter * 5000);
    
    // Velocity score (controlled movement = higher stability)
    let velocityScore = 100;
    if (velocity) {
        const velocityMagnitude = Math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2);
        velocityScore = Math.max(0, 100 - velocityMagnitude * 100);
    }
    
    // Visibility score (consistent detection = higher stability)
    const visibilityScore = landmark.visibility * 100;
    
    // Combined stability score (weighted average)
    return jitterScore * 0.5 + velocityScore * 0.3 + visibilityScore * 0.2;
};

// Calculate physics-based metrics
const calculatePhysics = (
    landmarks: Landmark[], 
    previousLandmarks: Landmark[] | null,
    userWeight: number = 70
): PhysicsData => {
    // Use multiple key points for accurate movement detection
    const hipCenter = {
        x: (landmarks[23].x + landmarks[24].x) / 2,
        y: (landmarks[23].y + landmarks[24].y) / 2,
        z: (landmarks[23].z + landmarks[24].z) / 2
    };
    
    const kneeCenter = {
        x: (landmarks[25].x + landmarks[26].x) / 2,
        y: (landmarks[25].y + landmarks[26].y) / 2,
        z: (landmarks[25].z + landmarks[26].z) / 2
    };
    
    const shoulderCenter = {
        x: (landmarks[11].x + landmarks[12].x) / 2,
        y: (landmarks[11].y + landmarks[12].y) / 2,
        z: (landmarks[11].z + landmarks[12].z) / 2
    };
    
    let velocity = 0;
    let acceleration = 0;
    
    if (previousLandmarks) {
        const prevHipCenter = {
            x: (previousLandmarks[23].x + previousLandmarks[24].x) / 2,
            y: (previousLandmarks[23].y + previousLandmarks[24].y) / 2,
            z: (previousLandmarks[23].z + previousLandmarks[24].z) / 2
        };
        
        const prevKneeCenter = {
            x: (previousLandmarks[25].x + previousLandmarks[26].x) / 2,
            y: (previousLandmarks[25].y + previousLandmarks[26].y) / 2,
            z: (previousLandmarks[25].z + previousLandmarks[26].z) / 2
        };
        
        // Calculate displacement of center of mass (hip + knee + shoulder)
        const hipDisplacement = Math.sqrt(
            Math.pow(hipCenter.x - prevHipCenter.x, 2) +
            Math.pow(hipCenter.y - prevHipCenter.y, 2) +
            Math.pow(hipCenter.z - prevHipCenter.z, 2)
        );
        
        const kneeDisplacement = Math.sqrt(
            Math.pow(kneeCenter.x - prevKneeCenter.x, 2) +
            Math.pow(kneeCenter.y - prevKneeCenter.y, 2) +
            Math.pow(kneeCenter.z - prevKneeCenter.z, 2)
        );
        
        // Average displacement across body parts (more accurate than single point)
        const avgDisplacement = (hipDisplacement + kneeDisplacement) / 2;
        
        // Convert to m/s (assuming 30 FPS, multiply by frame rate and scale factor)
        // Screen coordinates are 0-1, approximate real-world scale: 1.7m height
        const heightScale = 1.7; // Average human height in meters
        velocity = avgDisplacement * heightScale * 30; // displacement * scale * fps
    }
    
    const hipVelocity = landmarkBuffer.getVelocity(23);
    const hipAcceleration = landmarkBuffer.getAcceleration(23);
    
    if (hipAcceleration) {
        const accMagnitude = Math.sqrt(
            hipAcceleration.x ** 2 + 
            hipAcceleration.y ** 2 + 
            hipAcceleration.z ** 2
        );
        acceleration = accMagnitude * 1.7 * 30; // Scale to m/s²
    }
    
    const momentum = userWeight * velocity;
    
    // Recalibrated power calculation using vertical displacement
    // Power = mass × gravity × vertical_velocity (more accurate than generic velocity)
    const verticalVelocity = previousLandmarks ? Math.abs(
        hipCenter.y - ((previousLandmarks[23].y + previousLandmarks[24].y) / 2)
    ) * 1.7 * 30 : 0;
    
    const powerOutput = userWeight * 9.81 * verticalVelocity; // Watts
    
    const stability = calculateStability(landmarks[23], 23); // Hip center stability
    
    return {
        velocity: parseFloat(velocity.toFixed(3)),
        acceleration: parseFloat(acceleration.toFixed(3)),
        momentum: parseFloat(momentum.toFixed(2)),
        stability: parseFloat(stability.toFixed(1)),
        powerOutput: parseFloat(powerOutput.toFixed(2))
    };
};

// Calculate biomechanical metrics
const calculateBiomechanics = (landmarks: Landmark[]): BiomechanicalMetrics => {
    const leftKneeAngle = calculate3DAngle(landmarks[23], landmarks[25], landmarks[27]);
    const rightKneeAngle = calculate3DAngle(landmarks[24], landmarks[26], landmarks[28]);
    const leftElbowAngle = calculate3DAngle(landmarks[11], landmarks[13], landmarks[15]);
    const rightElbowAngle = calculate3DAngle(landmarks[12], landmarks[14], landmarks[16]);
    const leftHipAngle = calculate3DAngle(landmarks[11], landmarks[23], landmarks[25]);
    const rightHipAngle = calculate3DAngle(landmarks[12], landmarks[24], landmarks[26]);
    
    const jointAngles = {
        leftKnee: leftKneeAngle,
        rightKnee: rightKneeAngle,
        leftElbow: leftElbowAngle,
        rightElbow: rightElbowAngle,
        leftHip: leftHipAngle,
        rightHip: rightHipAngle
    };
    
    // Calculate asymmetry
    const kneeAsymmetry = Math.abs(leftKneeAngle - rightKneeAngle);
    const elbowAsymmetry = Math.abs(leftElbowAngle - rightElbowAngle);
    const hipAsymmetry = Math.abs(leftHipAngle - rightHipAngle);
    const asymmetry = (kneeAsymmetry + elbowAsymmetry + hipAsymmetry) / 3;
    
    // Balance score based on center of mass
    const comX = landmarks.reduce((sum, lm) => sum + lm.x, 0) / landmarks.length;
    const leftFoot = landmarks[27];
    const rightFoot = landmarks[28];
    const footCenterX = (leftFoot.x + rightFoot.x) / 2;
    const balanceOffset = Math.abs(comX - footCenterX);
    const balanceScore = Math.max(0, 100 - balanceOffset * 500);
    
    const depthEstimate = estimateDepth(landmarks);
    
    return {
        jointAngles,
        rangeOfMotion: 85, // Placeholder - would need exercise-specific ROM data
        asymmetry: parseFloat(asymmetry.toFixed(2)),
        balanceScore: parseFloat(balanceScore.toFixed(1)),
        depthEstimate: parseFloat(depthEstimate.toFixed(2))
    };
};

// Enhanced ensemble analysis with trend detection
const ensembleAnalysis = (
    basicAnalysis: AnalysisResult,
    physicsData: PhysicsData,
    biomechanics: BiomechanicalMetrics
): number => {
    // Weight different factors
    const formWeight = 0.4;
    const physicsWeight = 0.3;
    const biomechanicsWeight = 0.3;
    
    const physicsScore = Math.min(100, physicsData.stability);
    const biomechanicsScore = Math.max(0, 100 - biomechanics.asymmetry * 2);
    
    return (
        basicAnalysis.formScore * formWeight +
        physicsScore * physicsWeight +
        biomechanicsScore * biomechanicsWeight
    );
};

// Analyze performance trend
const analyzeTrend = (currentScore: number): 'improving' | 'stable' | 'declining' => {
    formScoreHistory.push(currentScore);
    if (formScoreHistory.length > 10) {
        formScoreHistory.shift();
    }
    
    if (formScoreHistory.length < 3) return 'stable';
    
    const recent = formScoreHistory.slice(-3);
    const earlier = formScoreHistory.slice(-6, -3);
    
    if (earlier.length === 0) return 'stable';
    
    const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
    const earlierAvg = earlier.reduce((a, b) => a + b, 0) / earlier.length;
    
    if (recentAvg > earlierAvg + 5) return 'improving';
    if (recentAvg < earlierAvg - 5) return 'declining';
    return 'stable';
};

// Create enhanced, user-friendly feedback
const createEnhancedFeedback = (
    message: string,
    voiceMessage: string,
    type: 'critical' | 'adjustment' | 'positive' | 'info' | 'encouragement' | 'warning',
    problemLandmarks: number[] = [],
    confidence: number = 1.0,
    biomechanicalRisk: 'low' | 'medium' | 'high' = 'low',
    priority: 'immediate' | 'high' | 'medium' | 'low' = 'medium'
): Feedback => {
    return {
        message,
        voiceMessage,
        type,
        problemLandmarks,
        confidence,
        biomechanicalRisk,
        priority,
        encouragementLevel: type === 'positive' ? 100 : type === 'critical' ? 20 : 60
    };
};

let previousLandmarks: Landmark[] | null = null;


const analyzeSquat = (landmarks: Landmark[], currentStage: string, repCount: number = 0): AnalysisResult => {
    let feedback: Feedback = createEnhancedFeedback(
        '', '', 'info', [], 1.0, 'low', 'low'
    );
    let mistake: string | null = null;
    let stage = currentStage, repCounted = false, formScore = 100;
    let shouldSpeak = false;
    
    // Update landmark buffer for temporal analysis
    landmarkBuffer.add(landmarks);
    
    const [leftHip, leftKnee, leftAnkle, leftShoulder] = [landmarks[23], landmarks[25], landmarks[27], landmarks[11]];
    const [rightHip, rightKnee, rightAnkle, rightShoulder] = [landmarks[24], landmarks[26], landmarks[28], landmarks[12]];

    if (leftHip.visibility < 0.8 || rightHip.visibility < 0.8 || leftKnee.visibility < 0.8 || rightKnee.visibility < 0.8) {
        return { 
            feedback: createEnhancedFeedback(
                'Step back please',
                'I can\'t see you clearly. Please take a step back from the camera so I can track your full movement.',
                'warning',
                [],
                0.3,
                'high',
                'immediate'
            ),
            stage, 
            repCounted, 
            mistake: 'Poor Visibility', 
            formScore: 0,
            shouldSpeak: true
        };
    }

    // Use 3D angles for more accurate analysis
    const avgKneeAngle = (calculate3DAngle(leftHip, leftKnee, leftAnkle) + calculate3DAngle(rightHip, rightKnee, rightAnkle)) / 2;
    const avgHipAngle = (calculate3DAngle(leftShoulder, leftHip, leftKnee) + calculate3DAngle(rightShoulder, rightHip, rightKnee)) / 2;

    // Calculate physics and biomechanics
    const physics = calculatePhysics(landmarks, previousLandmarks);
    const biomechanics = calculateBiomechanics(landmarks);
    
    // ===== CRITICAL FIX: Detect if user is actually exercising =====
    // Check if angles indicate actual squatting position
    const isSquatting = avgKneeAngle < 170 || avgHipAngle < 170;
    const isStanding = avgKneeAngle >= 165 && avgHipAngle >= 165;
    
    // If standing still and no movement detected
    if (isStanding && physics.velocity < 0.05 && currentStage === 'up') {
        return {
            feedback: createEnhancedFeedback(
                'Ready to Start',
                'I can see you clearly. Begin your squat by bending your knees and hips. Go down slowly and controlled.',
                'info',
                [],
                1.0,
                'low',
                'low'
            ),
            stage: 'up',
            repCounted: false,
            mistake: null,
            formScore: 0,
            physics,
            biomechanics,
            shouldSpeak: false
        };
    }
    
    // Assess injury risk based on joint angles and stability
    let biomechanicalRisk: 'low' | 'medium' | 'high' = 'low';
    if (biomechanics.asymmetry > 15 || physics.stability < 50) {
        biomechanicalRisk = 'high';
        formScore -= 15;
    } else if (biomechanics.asymmetry > 8 || physics.stability < 70) {
        biomechanicalRisk = 'medium';
        formScore -= 5;
    }

    // Stage detection based on actual joint angles
    if (avgHipAngle > 160 && avgKneeAngle > 160) {
        stage = 'up';
    }
    
    if (stage === 'down' || (avgKneeAngle < 150 && avgHipAngle < 150 && currentStage === 'up')) {
        const torsoAngle = Math.atan2(leftShoulder.y - leftHip.y, leftShoulder.x - leftHip.x);
        const shinAngle = Math.atan2(leftKnee.y - leftAnkle.y, leftKnee.x - leftAnkle.x);
        const torsoShinAngleDiff = Math.abs((torsoAngle - shinAngle) * (180 / Math.PI));
        
        // Enhanced knee valgus detection with depth consideration
        const leftKneeValgus = leftKnee.x > leftAnkle.x + 0.02; // More precise threshold
        const rightKneeValgus = rightKnee.x < rightAnkle.x - 0.02;
        
        // Check knee stability using advanced jitter detection
        const kneeStability = calculateStability(leftKnee, 25); // Left knee index

        if (torsoShinAngleDiff > 50) {
            mistake = "Chest Falling"; formScore -= 40; 
            biomechanicalRisk = 'high';
            shouldSpeak = true;
            feedback = createEnhancedFeedback(
                'Chest Up! 📐',
                'Your chest is dropping forward. Keep it proud and upright to protect your back.',
                'critical',
                [11, 12, 23, 24],
                0.9,
                biomechanicalRisk,
                'immediate'
            );
        } else if (leftKneeValgus || rightKneeValgus) { 
            mistake = "Knee Valgus"; formScore -= 30;
            biomechanicalRisk = 'high';
            shouldSpeak = true;
            feedback = createEnhancedFeedback(
                'Knees Out! ⚠️',
                'Your knees are caving inward. Push them out to align with your toes - this protects your joints.',
                'critical',
                [25, 26],
                0.85,
                biomechanicalRisk,
                'immediate'
            );
        } else if (kneeStability < 60) {
            mistake = "Unstable Knees"; formScore -= 25;
            biomechanicalRisk = 'medium';
            shouldSpeak = true;
            feedback = createEnhancedFeedback(
                'Stabilize 🎯',
                'Your knees are wobbling. Slow down and control the movement to build stability.',
                'adjustment',
                [25, 26],
                0.75,
                biomechanicalRisk,
                'high'
            );
        } else if (leftHip.y > leftKnee.y + 0.03 && rightHip.y > rightKnee.y + 0.03) {
            mistake = "Not Deep Enough"; formScore -= 20;
            shouldSpeak = false; // Don't spam this feedback
            feedback = createEnhancedFeedback(
                'Go Deeper 📏',
                'Great control! Try going a bit deeper - your hips should reach knee level for full range of motion.',
                'adjustment',
                [23, 24],
                0.8,
                'low',
                'low'
            );
        } else {
            shouldSpeak = false;
            feedback = createEnhancedFeedback(
                'Perfect! 💪',
                'Excellent form! Your technique is spot-on!',
                'positive',
                [],
                0.95,
                'low',
                'medium'
            );
        }
        
        // ENHANCED: Check for repeated mistakes and provide detailed explanation after 12 seconds
        if (mistake && feedbackManager.shouldRepeatMistakeFeedback(mistake, Date.now())) {
            const mistakeCount = feedbackManager.recordMistake(mistake);
            if (mistakeCount >= 3) {
                shouldSpeak = true;
                const detailedFeedback = feedbackManager.getRepeatedMistakeFeedback(mistake, mistakeCount);
                feedback = createEnhancedFeedback(
                    `⚠️ ${mistake} (×${mistakeCount})`,
                    detailedFeedback,
                    'critical',
                    feedback.problemLandmarks,
                    0.9,
                    biomechanicalRisk,
                    'immediate'
                );
            }
        }
    }

    // Detect squat bottom position - requires BOTH angle and movement
    if (avgHipAngle < 120 && avgKneeAngle < 130 && currentStage === 'up') {
        // Lowered threshold from 100 to allow partial squats to count
        // Only transition if there was actual downward movement
        if (physics.velocity > 0.05) { // Made more sensitive
            stage = 'down'; 
            shouldSpeak = false;
            feedback = createEnhancedFeedback(
                'Down... ⬇️',
                'Descending',
                'info',
                [],
                1.0,
                'low',
                'low'
            );
        }
    } else if (stage === 'up' && currentStage === 'down') {
        // Rep completed - coming back up
        // BEGINNER-FRIENDLY: Count ALL reps, give feedback on quality separately
        if (physics.velocity > 0.03) { // More sensitive to movement
            repCounted = true; // Count the rep no matter what!
            console.log('🎯 REP COUNTED! Form Score:', formScore, '| Rep #', repCount + 1);
            
            // NOW provide quality-based feedback (not blocking the count)
            const encouragement = feedbackManager.getEncouragementMessage(repCount + 1);
            
            // Dynamic feedback based on actual form, but always encouraging
            let repMessage = '';
            let feedbackType: 'positive' | 'adjustment' | 'info' = 'positive';
            
            if (formScore >= 85) {
                repMessage = `${encouragement} Outstanding rep! Form score: ${formScore}%. You're crushing it! 💪`;
                shouldSpeak = true;
            } else if (formScore >= 70) {
                repMessage = `${encouragement} Good rep! ${formScore}% form. ${mistake ? `Small tip: work on ${mistake.toLowerCase()}.` : 'Solid work!'}`;
                shouldSpeak = repCount % 3 === 0; // Speak every 3rd rep
            } else if (formScore >= 45) {
                repMessage = `Rep ${repCount + 1} counted. Form at ${formScore}%. ${mistake ? `Focus on: ${mistake}. ` : ''}You're building strength - form will improve!`;
                feedbackType = 'adjustment';
                shouldSpeak = repCount % 5 === 0; // Speak every 5th rep
            } else {
                repMessage = `Rep ${repCount + 1}. ${mistake ? `Watch ${mistake.toLowerCase()} - ` : ''}Quality is ${formScore}%. Slow down and focus on control. Progress takes time! 🎯`;
                feedbackType = 'adjustment';
                shouldSpeak = repCount === 0 || repCount % 5 === 0; // First rep and every 5th
            }
            
            feedback = createEnhancedFeedback(
                `Rep ${repCount + 1} ✓`,
                repMessage,
                feedbackType,
                [],
                0.95,
                formScore >= 75 ? 'low' : 'medium',
                'medium'
            );
        }
    }
    
    previousLandmarks = landmarks;
    const trend = analyzeTrend(formScore);
    
    return { 
        feedback: feedbackManager.enhanceFeedback(feedback, mistake, repCount), 
        stage, 
        repCounted, 
        mistake, 
        formScore: Math.max(0, Math.round(formScore)),
        physics,
        biomechanics,
        ensembleConfidence: feedback.confidence,
        predictedNextStage: stage === 'down' ? 'up' : 'down',
        improvementTrend: trend,
        repeatMistakeCount: mistake ? feedbackManager.recordMistake(mistake) : 0,
        shouldSpeak
    };
};

const analyzePushup = (landmarks: Landmark[], currentStage: string, repCount: number = 0): AnalysisResult => {
    let feedback: Feedback = createEnhancedFeedback(
        '', '', 'info', [], 1.0, 'low', 'low'
    );
    let mistake: string | null = null;
    let stage = currentStage, repCounted = false, formScore = 100;
    let shouldSpeak = false;

    landmarkBuffer.add(landmarks);

    const [leftShoulder, leftElbow, leftWrist, leftHip, leftAnkle] = [landmarks[11], landmarks[13], landmarks[15], landmarks[23], landmarks[27]];
    const [rightShoulder, rightElbow, rightWrist, rightHip, rightAnkle] = [landmarks[12], landmarks[14], landmarks[16], landmarks[24], landmarks[28]];

    if (leftShoulder.visibility < 0.8 || leftElbow.visibility < 0.8 || leftHip.visibility < 0.8) {
         return { 
            feedback: createEnhancedFeedback(
                'Turn Sideways',
                'I can\'t see you clearly. Please position your side to the camera for better tracking.',
                'warning',
                [],
                0.3,
                'high',
                'immediate'
            ),
            stage, 
            repCounted, 
            mistake: 'Poor Visibility', 
            formScore: 0,
            shouldSpeak: true
        };
    }
    
    const avgElbowAngle = (calculate3DAngle(leftShoulder, leftElbow, leftWrist) + calculate3DAngle(rightShoulder, rightElbow, rightWrist)) / 2;
    const avgBodyAngle = (calculate3DAngle(leftShoulder, leftHip, leftAnkle) + calculate3DAngle(rightShoulder, rightHip, rightAnkle)) / 2;

    const physics = calculatePhysics(landmarks, previousLandmarks);
    const biomechanics = calculateBiomechanics(landmarks);
    
    // ===== CRITICAL FIX: Detect if user is actually exercising =====
    const isPushupPosition = avgElbowAngle < 170 || avgBodyAngle < 175;
    const isInStartPosition = avgElbowAngle >= 165 && avgBodyAngle >= 170;
    
    // If in plank position but not moving
    if (isInStartPosition && physics.velocity < 0.05 && currentStage === 'up') {
        return {
            feedback: createEnhancedFeedback(
                'Ready to Start',
                'Good plank position. Begin by bending your elbows and lowering your chest toward the ground.',
                'info',
                [],
                1.0,
                'low',
                'low'
            ),
            stage: 'up',
            repCounted: false,
            mistake: null,
            formScore: 0,
            physics,
            biomechanics,
            shouldSpeak: false
        };
    }
    
    let biomechanicalRisk: 'low' | 'medium' | 'high' = 'low';
    
    // Check elbow flare (should stay close to body)
    const elbowFlare = Math.abs(leftElbow.x - leftShoulder.x) + Math.abs(rightElbow.x - rightShoulder.x);
    if (elbowFlare > 0.3) {
        mistake = "Elbows Flaring";
        formScore -= 20;
        biomechanicalRisk = 'medium';
        shouldSpeak = true;
        feedback = createEnhancedFeedback(
            'Tuck Elbows 💪',
            'Keep your elbows closer to your body - about 45 degrees. This protects your shoulders from stress.',
            'adjustment',
            [13, 14],
            0.8,
            biomechanicalRisk,
            'high'
        );
    }

    if (avgElbowAngle > 160) stage = 'up';

    if (avgBodyAngle < 160) {
        mistake = "Hip Sag"; formScore -= 50; 
        biomechanicalRisk = 'high';
        shouldSpeak = true;
        feedback = createEnhancedFeedback(
            'Engage Core! 🔥',
            'Your hips are sagging. Squeeze your glutes and abs tight - imagine holding a perfect plank position!',
            'critical',
            [23, 24],
            0.9,
            biomechanicalRisk,
            'immediate'
        );
    } else if (stage === 'down' || (avgElbowAngle < 160 && currentStage === 'up')) {
        if (avgElbowAngle < 90) {
             shouldSpeak = false;
             feedback = createEnhancedFeedback(
                'Perfect Depth! 🎯',
                'Excellent depth on that pushup!',
                'positive',
                [],
                0.95,
                'low',
                'medium'
            );
        } else {
            if (mistake !== "Elbows Flaring" && mistake !== "Hip Sag") {
                mistake = "Not Deep Enough"; formScore -= 25;
                shouldSpeak = false;
                feedback = createEnhancedFeedback(
                    'Lower More 📏',
                    'Good control! Try to lower your chest closer to the floor for full range of motion.',
                    'adjustment',
                    [11, 12],
                    0.75,
                    'low',
                    'low'
                );
            }
        }
    }
    
    // Detect bottom position - requires BOTH angle and movement
    if (avgElbowAngle < 120 && currentStage === 'up') {
        // Made more forgiving - even partial pushups count
        // Only transition if there was actual downward movement
        if (physics.velocity > 0.05) { // More sensitive
            stage = 'down'; 
            shouldSpeak = false;
            feedback = createEnhancedFeedback(
                'Down... ⬇️',
                'Descending',
                'info',
                [],
                1.0,
                'low',
                'low'
            );
        }
    } else if (stage === 'up' && currentStage === 'down') {
        // Rep completed - ALWAYS count it!
        if (physics.velocity > 0.03) { // More sensitive
            repCounted = true; // Count every rep!
            console.log('💪 PUSHUP COUNTED! Form Score:', formScore, '| Rep #', repCount + 1);
            
            // Quality-based feedback (not blocking)
            const encouragement = feedbackManager.getEncouragementMessage(repCount + 1);
            let repMessage = '';
            let feedbackType: 'positive' | 'adjustment' | 'info' = 'positive';
            
            if (formScore >= 85) {
                repMessage = `${encouragement} Killer rep! ${formScore}% form. Your technique is excellent! 🔥`;
                shouldSpeak = true;
            } else if (formScore >= 70) {
                repMessage = `${encouragement} Solid pushup at ${formScore}%. ${mistake ? `Tip: ${mistake.toLowerCase()}.` : 'Nice work!'}`;
                shouldSpeak = repCount % 3 === 0;
            } else if (formScore >= 45) {
                repMessage = `Rep ${repCount + 1} done. ${formScore}% form. ${mistake ? `Work on: ${mistake}. ` : ''}Keep going - you're building strength!`;
                feedbackType = 'adjustment';
                shouldSpeak = repCount % 5 === 0;
            } else {
                repMessage = `Rep ${repCount + 1}. ${mistake ? `${mistake} detected - ` : ''}${formScore}% quality. Focus on control. Every rep makes you stronger! 💪`;
                feedbackType = 'adjustment';
                shouldSpeak = repCount === 0 || repCount % 5 === 0;
            }
            
            feedback = createEnhancedFeedback(
                `Rep ${repCount + 1} ✓`,
                repMessage,
                feedbackType,
                [],
                0.95,
                formScore >= 75 ? 'low' : 'medium',
                'medium'
            );
        }
    }
    
    previousLandmarks = landmarks;
    const trend = analyzeTrend(formScore);
    
    return { 
        feedback: feedbackManager.enhanceFeedback(feedback, mistake, repCount),
        stage, 
        repCounted, 
        mistake, 
        formScore: Math.max(0, Math.round(formScore)),
        physics,
        biomechanics,
        ensembleConfidence: feedback.confidence,
        predictedNextStage: stage === 'down' ? 'up' : 'down',
        improvementTrend: trend,
        repeatMistakeCount: mistake ? feedbackManager.recordMistake(mistake) : 0,
        shouldSpeak
    };
};

const analyzeLunge = (landmarks: Landmark[], currentStage: string, repCount: number = 0): AnalysisResult => {
    let feedback: Feedback = createEnhancedFeedback(
        '', '', 'info', [], 1.0, 'low', 'low'
    );
    let mistake: string | null = null;
    let stage = currentStage, repCounted = false, formScore = 100;
    let shouldSpeak = false;

    landmarkBuffer.add(landmarks);

    const [leftHip, leftKnee, leftAnkle, leftShoulder] = [landmarks[23], landmarks[25], landmarks[27], landmarks[11]];
    const [rightHip, rightKnee, rightAnkle, rightShoulder] = [landmarks[24], landmarks[26], landmarks[28], landmarks[12]];

    if (leftKnee.visibility < 0.8 || rightKnee.visibility < 0.8) {
         return { 
            feedback: createEnhancedFeedback(
                'Turn Sideways',
                'Position your side to the camera so I can track your movement properly.',
                'warning',
                [],
                0.3,
                'high',
                'immediate'
            ),
            stage, 
            repCounted, 
            mistake: 'Poor Visibility', 
            formScore: 0,
            shouldSpeak: true
        };
    }
    
    const leftKneeAngle = calculate3DAngle(leftHip, leftKnee, leftAnkle);
    const rightKneeAngle = calculate3DAngle(rightHip, rightKnee, rightAnkle);
    
    const physics = calculatePhysics(landmarks, previousLandmarks);
    const biomechanics = calculateBiomechanics(landmarks);
    
    // ===== CRITICAL FIX: Detect if user is actually exercising =====
    const isLunging = leftKneeAngle < 170 || rightKneeAngle < 170;
    const isStanding = leftKneeAngle >= 165 && rightKneeAngle >= 165;
    
    // If standing still and no movement detected
    if (isStanding && physics.velocity < 0.05 && currentStage === 'up') {
        return {
            feedback: createEnhancedFeedback(
                'Ready to Start',
                'Step one leg forward and lower your body by bending both knees to 90 degrees.',
                'info',
                [],
                1.0,
                'low',
                'low'
            ),
            stage: 'up',
            repCounted: false,
            mistake: null,
            formScore: 0,
            physics,
            biomechanics,
            shouldSpeak: false
        };
    }
    
    let biomechanicalRisk: 'low' | 'medium' | 'high' = 'low';
    
    const isLeftLegForward = leftAnkle.x < rightAnkle.x;
    const [frontKnee, frontAnkle] = isLeftLegForward ? [leftKnee, leftAnkle] : [rightKnee, rightAnkle];
    const frontKneeAngle = isLeftLegForward ? leftKneeAngle : rightKneeAngle;
    
    if (frontKneeAngle > 160) {
        stage = 'up';
    } else {
        if (frontKnee.x < frontAnkle.x) {
            mistake = "Knee Over Toes"; formScore -= 40; 
            biomechanicalRisk = 'high';
            shouldSpeak = true;
            feedback = createEnhancedFeedback(
                'Knee Back! ⚠️',
                'Your knee is going past your toes. Shift your weight back to protect your knee joint.',
                'critical',
                [isLeftLegForward ? 25 : 26],
                0.9,
                biomechanicalRisk,
                'immediate'
            );
        } else if (frontKneeAngle < 80 || frontKneeAngle > 100) {
            mistake = "Incorrect Depth"; formScore -= 30; 
            biomechanicalRisk = 'medium';
            shouldSpeak = true;
            feedback = createEnhancedFeedback(
                'Check Depth 📐',
                'Aim for a 90-degree angle in both knees for optimal form and muscle activation.',
                'adjustment',
                [isLeftLegForward ? 25 : 26],
                0.75,
                biomechanicalRisk,
                'high'
            );
        } else if (biomechanics.balanceScore < 60) {
            mistake = "Poor Balance";
            formScore -= 20;
            biomechanicalRisk = 'medium';
            shouldSpeak = true;
            feedback = createEnhancedFeedback(
                'Balance 🎯',
                'Focus on maintaining your balance. Engage your core and keep your torso upright.',
                'adjustment',
                [23, 24],
                0.7,
                biomechanicalRisk,
                'high'
            );
        } else {
            shouldSpeak = false;
            feedback = createEnhancedFeedback(
                'Perfect! 💪',
                'Excellent lunge form! Your technique is on point!',
                'positive',
                [],
                0.95,
                'low',
                'medium'
            );
        }
    }

    if (frontKneeAngle < 120 && currentStage === 'up') {
        // Made more forgiving for beginners
        stage = isLeftLegForward ? 'left_down' : 'right_down';
        shouldSpeak = false;
        feedback = createEnhancedFeedback(
            'Down... ⬇️',
            'Descending',
            'info',
            [],
            1.0,
            'low',
            'low'
        );
    } else if (stage === 'up' && (currentStage === 'left_down' || currentStage === 'right_down')) {
        // Rep completed - ALWAYS count it!
        repCounted = true;
        console.log('🦵 LUNGE COUNTED! Form Score:', formScore, '| Rep #', repCount + 1);
        
        // Quality feedback (not blocking)
        const encouragement = feedbackManager.getEncouragementMessage(repCount + 1);
        let repMessage = '';
        let feedbackType: 'positive' | 'adjustment' | 'info' = 'positive';
        
        if (formScore >= 85) {
            repMessage = `${encouragement} Perfect lunge! ${formScore}% form. Excellent balance and control! 🎯`;
            shouldSpeak = true;
        } else if (formScore >= 70) {
            repMessage = `${encouragement} Good lunge! ${formScore}% form. ${mistake ? `Note: ${mistake.toLowerCase()}.` : 'Great work!'}`;
            shouldSpeak = repCount % 3 === 0;
        } else if (formScore >= 45) {
            repMessage = `Rep ${repCount + 1} complete. ${formScore}% form. ${mistake ? `Focus: ${mistake}. ` : ''}You're progressing!`;
            feedbackType = 'adjustment';
            shouldSpeak = repCount % 5 === 0;
        } else {
            repMessage = `Rep ${repCount + 1}. ${mistake ? `${mistake} - ` : ''}${formScore}% quality. Take it slow and focus on stability. You've got this! 💪`;
            feedbackType = 'adjustment';
            shouldSpeak = repCount === 0 || repCount % 5 === 0;
        }
        
        feedback = createEnhancedFeedback(
            `Rep ${repCount + 1} ✓`,
            repMessage,
            feedbackType,
            [],
            0.95,
            formScore >= 75 ? 'low' : 'medium',
            'medium'
        );
    }
    
    previousLandmarks = landmarks;
    const trend = analyzeTrend(formScore);
    
    return { 
        feedback: feedbackManager.enhanceFeedback(feedback, mistake, repCount),
        stage, 
        repCounted, 
        mistake, 
        formScore: Math.max(0, Math.round(formScore)),
        physics,
        biomechanics,
        ensembleConfidence: feedback.confidence,
        predictedNextStage: stage === 'up' ? (isLeftLegForward ? 'left_down' : 'right_down') : 'up',
        improvementTrend: trend,
        repeatMistakeCount: mistake ? feedbackManager.recordMistake(mistake) : 0,
        shouldSpeak
    };
};

const analyzePlank = (landmarks: Landmark[], repCount: number = 0): AnalysisResult => {
    let feedback: Feedback = createEnhancedFeedback(
        '', '', 'info', [], 1.0, 'low', 'low'
    );
    let mistake: string | null = null;
    let formScore = 100;
    let shouldSpeak = false;
    
    landmarkBuffer.add(landmarks);
    
    const [leftShoulder, leftHip, leftAnkle] = [landmarks[11], landmarks[23], landmarks[27]];

    if (leftShoulder.visibility < 0.8 || leftHip.visibility < 0.8 || leftAnkle.visibility < 0.8) {
        return { 
            feedback: createEnhancedFeedback(
                'Turn Sideways',
                'I need a clear side view to check your plank form properly.',
                'warning',
                [],
                0.3,
                'high',
                'immediate'
            ),
            stage: 'holding', 
            repCounted: false, 
            mistake: 'Poor Visibility', 
            formScore: 0,
            shouldSpeak: true
        };
    }
    
    const bodyAngle = calculate3DAngle(leftShoulder, leftHip, leftAnkle);
    
    const physics = calculatePhysics(landmarks, previousLandmarks);
    const biomechanics = calculateBiomechanics(landmarks);
    
    let biomechanicalRisk: 'low' | 'medium' | 'high' = 'low';
    
    // Check core stability through hip stability
    const coreStability = physics.stability;
    
    if (bodyAngle > 170 && coreStability > 70) {
        shouldSpeak = false;
        feedback = createEnhancedFeedback(
            'Perfect! 🔥',
            'Perfect form! Keep holding that strong position!',
            'positive',
            [],
            0.95,
            'low',
            'medium'
        );
    } else if (bodyAngle < 155) {
        mistake = "Hip Sag"; formScore -= 30;
        biomechanicalRisk = 'high';
        shouldSpeak = true;
        feedback = createEnhancedFeedback(
            'Lift Hips! 🔥',
            'Engage your core and glutes to lift your hips. Don\'t let your back sag - protect your spine!',
            'critical',
            [23],
            0.9,
            biomechanicalRisk,
            'immediate'
        );
    } else if (bodyAngle > 185) {
        mistake = "Hips Too High"; formScore -= 30;
        biomechanicalRisk = 'medium';
        shouldSpeak = true;
        feedback = createEnhancedFeedback(
            'Lower Hips 📏',
            'Lower your hips to form a straight line from shoulders to ankles for proper alignment.',
            'adjustment',
            [23],
            0.85,
            biomechanicalRisk,
            'high'
        );
    } else if (coreStability < 50) {
        mistake = "Unstable Core"; formScore -= 25;
        biomechanicalRisk = 'medium';
        shouldSpeak = true;
        feedback = createEnhancedFeedback(
            'Stabilize Core 🎯',
            'Your core is shaking. Breathe steadily and engage your abdominal muscles.',
            'adjustment',
            [23],
            0.75,
            biomechanicalRisk,
            'high'
        );
    }
    
    previousLandmarks = landmarks;
    const trend = analyzeTrend(formScore);

    return { 
        feedback: feedbackManager.enhanceFeedback(feedback, mistake, repCount),
        stage: 'holding', 
        repCounted: false, 
        mistake, 
        formScore: Math.max(0, Math.round(formScore)),
        physics,
        biomechanics,
        ensembleConfidence: feedback.confidence,
        predictedNextStage: 'holding',
        improvementTrend: trend,
        repeatMistakeCount: mistake ? feedbackManager.recordMistake(mistake) : 0,
        shouldSpeak
    };
};

export const analyzePose = (
    exercise: { id: string, type: string },
    landmarks: NormalizedLandmark[],
    currentStage: string,
    repCount: number = 0
): AnalysisResult => {
    // Get base analysis from exercise-specific function
    let result: AnalysisResult;
    
    switch (exercise.id) {
        case 'squats':
            result = analyzeSquat(landmarks, currentStage, repCount);
            break;
        case 'pushups':
            result = analyzePushup(landmarks, currentStage, repCount);
            break;
        case 'lunges':
            result = analyzeLunge(landmarks, currentStage, repCount);
            break;
        case 'plank':
            result = analyzePlank(landmarks, repCount);
            break;
        default:
            return { 
                feedback: createEnhancedFeedback(
                    'Unknown Exercise',
                    'This exercise is not recognized. Please select a different exercise.',
                    'warning',
                    [],
                    0.0,
                    'low',
                    'immediate'
                ),
                stage: currentStage, 
                repCounted: false, 
                mistake: null, 
                formScore: 0,
                shouldSpeak: true
            };
    }
    
    // ====== ADVANCED ML ENHANCEMENTS ======
    
    // 1. Record rep data for adaptive learning
    if (result.repCounted && result.physics && result.biomechanics) {
        const rom = result.physics.velocity > 0 ? 100 : 80; // Simplified ROM estimation
        const tempo = 1.5; // Simplified tempo
        adaptiveSystem.recordRep(result.formScore, rom, tempo);
    }
    
    // 2. Fatigue Detection
    const fatigueStatus = adaptiveSystem.detectFatigue();
    if (fatigueStatus.isFatigued && fatigueStatus.severity > 50) {
        result.feedback = createEnhancedFeedback(
            '⚠️ Fatigue Detected',
            `I'm noticing your form is dropping due to fatigue. Consider taking a short rest to maintain quality and prevent injury.`,
            'warning',
            result.feedback.problemLandmarks,
            result.feedback.confidence,
            'medium',
            'high'
        );
        result.shouldSpeak = true;
    }
    
    // 3. Injury Risk Prediction
    if (result.physics && result.biomechanics) {
        const rom = result.physics.velocity > 0 ? 100 : 80;
        const tempo = 1.5;
        const injuryRisk = adaptiveSystem.predictInjuryRisk(rom, tempo);
        
        if (injuryRisk > 60) {
            result.feedback = createEnhancedFeedback(
                '🚨 High Injury Risk',
                `Warning! Your movement pattern shows high injury risk. Slow down, reduce range of motion, and focus on controlled movements.`,
                'critical',
                result.feedback.problemLandmarks,
                result.feedback.confidence,
                'high',
                'immediate'
            );
            result.formScore = Math.min(result.formScore, 50);
            result.shouldSpeak = true;
        } else if (injuryRisk > 40) {
            result.feedback.biomechanicalRisk = 'medium';
        }
    }
    
    // 4. Movement Quality Assessment
    if (result.physics && result.biomechanics) {
        const quality = qualityAssessor.assessMovementQuality(
            landmarks,
            result.formScore,
            result.physics,
            result.biomechanics
        );
        
        // Add movement quality insights to feedback
        if (quality.overallQuality > 85 && result.formScore > 80) {
            result.feedback = createEnhancedFeedback(
                '⭐ Excellent Form!',
                `Outstanding! Your movement quality is exceptional - smooth ${quality.smoothness.toFixed(0)}%, controlled ${quality.control.toFixed(0)}%, efficient ${quality.efficiency.toFixed(0)}%. Keep it up!`,
                'positive',
                [],
                1.0,
                'low',
                'medium'
            );
        } else if (quality.smoothness < 60 && result.formScore > 70) {
            // Good form but jerky movement
            result.feedback = createEnhancedFeedback(
                '🎯 Improve Smoothness',
                `Your form is good, but try to move more smoothly. Control the movement throughout the entire range of motion.`,
                'adjustment',
                result.feedback.problemLandmarks,
                0.9,
                'low',
                'medium'
            );
        }
        
        // Store quality metrics in result
        result.movementQuality = {
            overall: quality.overallQuality,
            smoothness: quality.smoothness,
            control: quality.control,
            efficiency: quality.efficiency
        };
    }
    
    // 5. Adaptive Threshold Application
    const adaptiveFormThreshold = adaptiveSystem.getAdaptiveThreshold('formScore');
    if (result.formScore < adaptiveFormThreshold && result.formScore > 50) {
        // Below personal threshold but not critically bad
        result.feedback = createEnhancedFeedback(
            '📊 Below Your Standard',
            `Your form is at ${result.formScore}, but you typically perform better. Focus on maintaining your usual quality.`,
            'adjustment',
            result.feedback.problemLandmarks,
            0.85,
            'low',
            'medium'
        );
    }
    
    // 6. Progressive Overload Suggestion
    if (repCount > 0 && repCount % 10 === 0 && result.formScore > 85) {
        const avgFormScore = formScoreHistory.length > 0 
            ? formScoreHistory.reduce((a, b) => a + b, 0) / formScoreHistory.length 
            : result.formScore;
            
        if (avgFormScore > 85) {
            result.feedback = createEnhancedFeedback(
                '💪 Ready for Challenge',
                `Amazing work! ${repCount} reps with excellent form. You might be ready to increase difficulty or add weight.`,
                'positive',
                [],
                1.0,
                'low',
                'low'
            );
        }
    }
    
    return result;
};
