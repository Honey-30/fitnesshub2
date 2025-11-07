import drawingUtils from '@mediapipe/drawing_utils';
import mediaPipePose from '@mediapipe/pose';
import { Activity, AlertTriangle, CameraOff, CheckCircle, Loader2, Shield, Timer, TrendingUp } from 'lucide-react';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { usePoseDetection } from '../hooks/usePoseDetection';
import { BiomechanicalMetrics, Exercise, Feedback, PhysicsData, WorkoutSession } from '../types';
import { analyzePose, calculateAngle } from '../utils/poseAnalyzer';

interface PoseCanvasProps {
  exercise: Exercise;
  onWorkoutComplete: (session: WorkoutSession) => void;
  userWeight: number; // in kg
}

const FormGauge: React.FC<{ score: number }> = ({ score }) => {
    const color = score > 80 ? '#22c55e' : score > 50 ? '#eab308' : '#ef4444';
    const circumference = 2 * Math.PI * 28;
    const offset = circumference - (score / 100) * circumference;

    return (
        <div className="relative w-20 h-20">
            <svg className="w-full h-full" viewBox="0 0 60 60">
                <circle cx="30" cy="30" r="28" fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth="4" />
                <circle
                    cx="30"
                    cy="30"
                    r="28"
                    fill="none"
                    stroke={color}
                    strokeWidth="4"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    strokeLinecap="round"
                    transform="rotate(-90 30 30)"
                    style={{ transition: 'stroke-dashoffset 0.3s ease-out, stroke 0.3s' }}
                />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-xl font-bold text-white drop-shadow-md">{score}</span>
            </div>
        </div>
    );
};

const MetricDisplay: React.FC<{ icon: React.ReactNode; label: string; value: string; color: string }> = ({ icon, label, value, color }) => (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/60 backdrop-blur-sm ${color}`}>
        {icon}
        <div className="flex flex-col">
            <span className="text-[10px] uppercase font-semibold tracking-wider opacity-80">{label}</span>
            <span className="text-sm font-bold">{value}</span>
        </div>
    </div>
);

const RiskIndicator: React.FC<{ risk: 'low' | 'medium' | 'high' }> = ({ risk }) => {
    const config = {
        low: { color: 'text-green-400', bg: 'bg-green-500/20', label: 'LOW RISK' },
        medium: { color: 'text-yellow-400', bg: 'bg-yellow-500/20', label: 'MEDIUM' },
        high: { color: 'text-red-400', bg: 'bg-red-500/20', label: 'HIGH RISK' }
    };
    const { color, bg, label } = config[risk];
    
    return (
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${bg} backdrop-blur-sm ${color}`}>
            <Shield size={16} />
            <span className="text-xs font-bold">{label}</span>
        </div>
    );
};

const PoseCanvas: React.FC<PoseCanvasProps> = ({ exercise, onWorkoutComplete, userWeight }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [repCount, setRepCount] = useState(0);
  const [stage, setStage] = useState<string>(exercise.type === 'reps' ? 'up' : 'holding');
  const [feedback, setFeedback] = useState<Feedback>({ 
    message: 'Get into position', 
    voiceMessage: 'Get into position.', 
    type: 'info',
    confidence: 1.0,
    biomechanicalRisk: 'low'
  });
  
  const [startTime] = useState(Date.now());
  const [mistakes, setMistakes] = useState<Record<string, number>>({});
  
  const [formScore, setFormScore] = useState(100);
  const [formScores, setFormScores] = useState<number[]>([]);
  
  // Advanced metrics state
  const [physics, setPhysics] = useState<PhysicsData | null>(null);
  const [biomechanics, setBiomechanics] = useState<BiomechanicalMetrics | null>(null);
  const [peakPower, setPeakPower] = useState(0);
  const [velocityHistory, setVelocityHistory] = useState<number[]>([]);
  
  // New ML-enhanced metrics
  const [movementQuality, setMovementQuality] = useState<{ overall: number; smoothness: number; control: number; efficiency: number } | null>(null);
  const [fatigueLevel, setFatigueLevel] = useState(0);
  const [injuryRisk, setInjuryRisk] = useState(0);
  
  const [timer, setTimer] = useState(0);
  const [voicesLoaded, setVoicesLoaded] = useState(false);
  
  const lastSpokenTimeRef = useRef(0);
  const lastSpokenMessageRef = useRef('');


  const speak = useCallback((text: string, isHighPriority: boolean) => {
    if (!text || !voicesLoaded || !speechSynthesis) return;

    if (isHighPriority) {
        speechSynthesis.cancel();
    } else if (speechSynthesis.speaking) {
        return; 
    }
    
    const utterance = new SpeechSynthesisUtterance(text);
    const voices = speechSynthesis.getVoices();
    const femaleVoice = voices.find(v => v.name.includes('Google US English') && v.name.includes('Female'));
    utterance.voice = femaleVoice || voices[0];
    utterance.rate = 1.1;
    speechSynthesis.speak(utterance);
  }, [voicesLoaded]);

  // Robust voice loading check & Keep-alive
  useEffect(() => {
    let keepAliveInterval: number | null = null;
    let voiceCheckInterval: number | null = null;

    const checkVoices = () => {
      if (speechSynthesis.getVoices().length > 0) {
        setVoicesLoaded(true);
        if (voiceCheckInterval) clearInterval(voiceCheckInterval);
        
        // Start keep-alive only after voices are loaded
        if (keepAliveInterval) clearInterval(keepAliveInterval);
        keepAliveInterval = window.setInterval(() => {
          if (!speechSynthesis.speaking) {
            speechSynthesis.resume();
          }
        }, 10000);

      }
    };
    
    checkVoices();
    speechSynthesis.onvoiceschanged = checkVoices;
    if (!voicesLoaded) {
      voiceCheckInterval = window.setInterval(checkVoices, 250);
    }

    return () => {
      speechSynthesis.onvoiceschanged = null;
      if (voiceCheckInterval) clearInterval(voiceCheckInterval);
      if (keepAliveInterval) clearInterval(keepAliveInterval);
      speechSynthesis.cancel();
    };
  }, [voicesLoaded]);

  // State-driven effect for speaking feedback
  useEffect(() => {
    if (!feedback || !feedback.voiceMessage) return;

    const now = Date.now();
    const isCritical = feedback.type === 'critical';
    const isRepCount = feedback.message === 'Great Rep!';
    const isHighPriority = isCritical || isRepCount;
    
    if (lastSpokenMessageRef.current === feedback.voiceMessage && !isRepCount) {
        return;
    }

    if (isHighPriority) {
        speak(feedback.voiceMessage, true);
        lastSpokenTimeRef.current = now;
        lastSpokenMessageRef.current = feedback.voiceMessage;
    } else if (now - lastSpokenTimeRef.current > 3000) {
        speak(feedback.voiceMessage, false);
        lastSpokenTimeRef.current = now;
        lastSpokenMessageRef.current = feedback.voiceMessage;
    }
  }, [feedback, speak]);

  const onResults = useCallback((results: any) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (results.poseLandmarks) {
      const analysis = analyzePose(exercise, results.poseLandmarks, stage, repCount);
      
      setFormScore(analysis.formScore);
      setFeedback(analysis.feedback);
      
      // Update advanced metrics
      if (analysis.physics) {
        setPhysics(analysis.physics);
        setVelocityHistory(prev => [...prev.slice(-29), analysis.physics!.velocity]);
        if (analysis.physics.powerOutput > peakPower) {
          setPeakPower(analysis.physics.powerOutput);
        }
      }
      
      if (analysis.biomechanics) {
        setBiomechanics(analysis.biomechanics);
      }
      
      // Update ML-enhanced metrics
      if (analysis.movementQuality) {
        setMovementQuality(analysis.movementQuality);
      }
      
      if (analysis.fatigueLevel !== undefined) {
        setFatigueLevel(analysis.fatigueLevel);
      }
      
      if (analysis.injuryRisk !== undefined) {
        setInjuryRisk(analysis.injuryRisk);
      }
      
      if (exercise.type === 'reps' && (stage === 'down' || stage.includes('_down'))) {
          setFormScores(prev => [...prev, analysis.formScore]);
      } else if (exercise.type === 'isometric' && stage === 'holding') {
          setFormScores(prev => [...prev, analysis.formScore]);
      }
      
      // Enhanced landmark visualization with advanced AI-style effects
      const landmarkStyle = (index: number, landmark: any) => {
        const isProblem = analysis.feedback.problemLandmarks?.includes(index);
        const riskLevel = analysis.feedback.biomechanicalRisk || 'low';
        
        // Get visibility/confidence (0-1 scale)
        const confidence = landmark.visibility || 0.5;
        
        let color = '#06b6d4'; // cyan-500 (AI-tech color)
        let fillColor = '#0891b2'; // cyan-600
        
        // Advanced color system based on confidence
        if (!isProblem) {
          if (confidence >= 0.9) {
            // High confidence - bright cyan/neon green (AI confirmed)
            color = '#00ffff'; // Neon cyan
            fillColor = '#00ff88'; // Neon green
          } else if (confidence >= 0.75) {
            // Good confidence - electric blue
            color = '#0ea5e9'; // sky-500
            fillColor = '#06b6d4'; // cyan-500
          } else if (confidence >= 0.6) {
            // Medium confidence - cyan
            color = '#06b6d4'; // cyan-500
            fillColor = '#0891b2'; // cyan-600
          } else if (confidence >= 0.4) {
            // Low-medium confidence - yellow/warning
            color = '#fbbf24'; // amber-400
            fillColor = '#f59e0b'; // amber-500
          } else {
            // Low confidence - orange/alert
            color = '#fb923c'; // orange-400
            fillColor = '#f97316'; // orange-500
          }
        } else {
          // Problem landmarks with high-tech alert colors
          if (riskLevel === 'high') {
            color = '#ff0055'; // Neon red
            fillColor = '#ff0000'; // Bright red
          } else if (riskLevel === 'medium') {
            color = '#ffaa00'; // Bright amber
            fillColor = '#ff8800'; // Orange
          }
        }
        
        return {
          color,
          fillColor,
          lineWidth: 2,
          radius: isProblem ? 7 : (confidence >= 0.8 ? 6 : 5),
        };
      };

      // Draw advanced HUD-style skeleton with futuristic effects
      const drawEnhancedSkeleton = () => {
        const time = Date.now() / 1000;
        
        // === STEP 1: Draw Subtle Grid Overlay ===
        ctx.save();
        ctx.globalAlpha = 0.08;
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 0.5;
        const gridSize = 40;
        for (let x = 0; x < canvas.width; x += gridSize) {
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, canvas.height);
          ctx.stroke();
        }
        for (let y = 0; y < canvas.height; y += gridSize) {
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(canvas.width, y);
          ctx.stroke();
        }
        ctx.restore();
        
        const riskColors = {
          low: '#06b6d4', // cyan-500
          medium: '#fbbf24', // amber-400
          high: '#ff0055' // Neon red
        };
        
        const connectionColor = riskColors[analysis.feedback.biomechanicalRisk || 'low'];
        
        // === STEP 2: Draw Energy Lines (Connections) - NO GLOW ===
        mediaPipePose.POSE_CONNECTIONS.forEach((connection: any) => {
          const from = results.poseLandmarks[connection[0]];
          const to = results.poseLandmarks[connection[1]];
          
          if (from && to) {
            const x1 = from.x * canvas.width;
            const y1 = from.y * canvas.height;
            const x2 = to.x * canvas.width;
            const y2 = to.y * canvas.height;
            
            // Create gradient for plasma effect
            const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
            gradient.addColorStop(0, connectionColor);
            gradient.addColorStop(0.5, connectionColor + 'CC');
            gradient.addColorStop(1, connectionColor);
            
            ctx.strokeStyle = gradient;
            ctx.lineWidth = 3;
            ctx.lineCap = 'round';
            
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        });
        
        // === STEP 3: Draw Hexagonal Rings and Keypoints - NO GLOW ===
        results.poseLandmarks.forEach((landmark: any, index: number) => {
          const x = landmark.x * canvas.width;
          const y = landmark.y * canvas.height;
          const style = landmarkStyle(index, landmark);
          const confidence = landmark.visibility || 0.5;
          const isProblem = analysis.feedback.problemLandmarks?.includes(index);
          
          ctx.save();
          
          // Draw hexagonal ring (futuristic node)
          const drawHexagon = (centerX: number, centerY: number, radius: number, color: string, lineWidth: number, filled: boolean = false) => {
            ctx.beginPath();
            for (let i = 0; i < 6; i++) {
              const angle = (Math.PI / 3) * i - Math.PI / 2;
              const hx = centerX + radius * Math.cos(angle);
              const hy = centerY + radius * Math.sin(angle);
              if (i === 0) ctx.moveTo(hx, hy);
              else ctx.lineTo(hx, hy);
            }
            ctx.closePath();
            
            if (filled) {
              ctx.fillStyle = color;
              ctx.fill();
            } else {
              ctx.strokeStyle = color;
              ctx.lineWidth = lineWidth;
              ctx.stroke();
            }
          };
          
          // Outer pulsating hexagon for high confidence
          if (confidence >= 0.8) {
            const pulseRadius = style.radius + Math.sin(time * 3 + index * 0.5) * 2 + 6;
            drawHexagon(x, y, pulseRadius, style.color + '50', 1.5);
          }
          
          // Middle hexagon ring
          drawHexagon(x, y, style.radius + 4, style.color, 2);
          
          // Inner filled circle (core)
          ctx.beginPath();
          ctx.arc(x, y, style.radius, 0, 2 * Math.PI);
          ctx.fillStyle = style.fillColor;
          ctx.fill();
          
          // Highlight dot
          ctx.beginPath();
          ctx.arc(x - style.radius * 0.3, y - style.radius * 0.3, style.radius * 0.3, 0, 2 * Math.PI);
          ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
          ctx.fill();
          
          ctx.restore();
          
          // === STEP 4: Draw Targeting Reticle for Problem Areas ===
          if (isProblem) {
            ctx.save();
            ctx.strokeStyle = '#ff0055';
            ctx.lineWidth = 2;
            
            const reticleSize = 16;
            
            // Animated rotating reticle
            ctx.translate(x, y);
            ctx.rotate(time * 0.5);
            
            // Crosshair
            ctx.beginPath();
            ctx.moveTo(-reticleSize, 0);
            ctx.lineTo(reticleSize, 0);
            ctx.moveTo(0, -reticleSize);
            ctx.lineTo(0, reticleSize);
            ctx.stroke();
            
            // Corner brackets
            const cornerSize = 6;
            ctx.beginPath();
            // Top-left
            ctx.moveTo(-reticleSize, -reticleSize + cornerSize);
            ctx.lineTo(-reticleSize, -reticleSize);
            ctx.lineTo(-reticleSize + cornerSize, -reticleSize);
            // Top-right
            ctx.moveTo(reticleSize - cornerSize, -reticleSize);
            ctx.lineTo(reticleSize, -reticleSize);
            ctx.lineTo(reticleSize, -reticleSize + cornerSize);
            // Bottom-right
            ctx.moveTo(reticleSize, reticleSize - cornerSize);
            ctx.lineTo(reticleSize, reticleSize);
            ctx.lineTo(reticleSize - cornerSize, reticleSize);
            // Bottom-left
            ctx.moveTo(-reticleSize + cornerSize, reticleSize);
            ctx.lineTo(-reticleSize, reticleSize);
            ctx.lineTo(-reticleSize, reticleSize - cornerSize);
            ctx.stroke();
            
            ctx.restore();
          }
        });
        
        // === STEP 5: Calculate and Display Real Angles for Critical Joints ===
        // Joint definitions: [point1, vertex, point2]
        const jointAngles: { [key: string]: [number, number, number] } = {
          'L.Shoulder': [13, 11, 23], // Left elbow - left shoulder - left hip
          'R.Shoulder': [14, 12, 24], // Right elbow - right shoulder - right hip
          'L.Elbow': [11, 13, 15],    // Left shoulder - left elbow - left wrist
          'R.Elbow': [12, 14, 16],    // Right shoulder - right elbow - right wrist
          'L.Hip': [11, 23, 25],      // Left shoulder - left hip - left knee
          'R.Hip': [12, 24, 26],      // Right shoulder - right hip - right knee
          'L.Knee': [23, 25, 27],     // Left hip - left knee - left ankle
          'R.Knee': [24, 26, 28],     // Right hip - right knee - right ankle
        };
        
        Object.entries(jointAngles).forEach(([jointName, [p1, vertex, p2]]) => {
          const point1 = results.poseLandmarks[p1];
          const vertexPoint = results.poseLandmarks[vertex];
          const point2 = results.poseLandmarks[p2];
          
          if (point1 && vertexPoint && point2) {
            const confidence = vertexPoint.visibility || 0;
            
            // Only show if confidence is good
            if (confidence >= 0.6) {
              const angle = calculateAngle(point1, vertexPoint, point2);
              const x = vertexPoint.x * canvas.width;
              const y = vertexPoint.y * canvas.height;
              const style = landmarkStyle(vertex, vertexPoint);
              
              ctx.save();
              
              // Calculate angle data
              const angleText = `${Math.round(angle)}°`;
              
              // Draw angular callout box
              const boxWidth = 50;
              const boxHeight = 22;
              const offsetX = 30;
              const offsetY = -15;
              
              // Draw connecting line
              ctx.strokeStyle = style.color + '80';
              ctx.lineWidth = 1.5;
              ctx.beginPath();
              ctx.moveTo(x, y);
              ctx.lineTo(x + offsetX, y + offsetY);
              ctx.stroke();
              
              // Draw box
              ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
              ctx.strokeStyle = style.color;
              ctx.lineWidth = 1.5;
              
              const boxX = x + offsetX;
              const boxY = y + offsetY - boxHeight / 2;
              
              ctx.fillRect(boxX, boxY, boxWidth, boxHeight);
              ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);
              
              // Draw text
              ctx.fillStyle = style.color;
              ctx.font = 'bold 12px monospace';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText(angleText, boxX + boxWidth / 2, boxY + boxHeight / 2);
              
              ctx.restore();
            }
          }
        });
      };

      drawEnhancedSkeleton();
      
      if (analysis.stage !== stage) {
          setStage(analysis.stage);
          lastSpokenMessageRef.current = '';
      }
      if (analysis.repCounted) {
          setRepCount(prev => prev + 1);
          lastSpokenMessageRef.current = '';
      }
      if (analysis.mistake) setMistakes(prev => ({ ...prev, [analysis.mistake!]: (prev[analysis.mistake!] || 0) + 1 }));
    }
    ctx.restore();
  }, [stage, exercise, formScores, peakPower]);

  const { isInitializing, error } = usePoseDetection({ onResults, videoRef, canvasRef, active: true });
  
  // Timer for isometric exercises
  useEffect(() => {
      let interval: number | undefined;
      if (exercise.type === 'isometric' && stage === 'holding' && !isInitializing && !error) {
          interval = window.setInterval(() => setTimer(prev => prev + 1), 1000);
      }
      return () => clearInterval(interval);
  }, [exercise.type, stage, isInitializing, error]);

  const handleFinish = () => {
    speechSynthesis.cancel();
    const durationSeconds = Math.round((Date.now() - startTime) / 1000);
    const caloriesBurned = (exercise.metValue * userWeight * (durationSeconds / 3600));
    const avgFormScore = formScores.length > 0 ? Math.round(formScores.reduce((a, b) => a + b, 0) / formScores.length) : 100;
    
    // Calculate additional advanced metrics
    const avgVelocity = velocityHistory.length > 0 
      ? velocityHistory.reduce((a, b) => a + b, 0) / velocityHistory.length 
      : 0;
    
    const consistencyScore = formScores.length > 1
      ? Math.max(0, 100 - (Math.max(...formScores) - Math.min(...formScores)))
      : 100;
    
    const biomechanicalEfficiency = biomechanics?.balanceScore 
      ? (biomechanics.balanceScore + (100 - biomechanics.asymmetry * 2)) / 2
      : 85;
    
    const session: WorkoutSession = {
      exercise,
      cleanReps: repCount,
      mistakes,
      durationSeconds,
      caloriesBurned: parseFloat(caloriesBurned.toFixed(2)),
      avgFormScore,
      peakPower: parseFloat(peakPower.toFixed(2)),
      avgVelocity: parseFloat(avgVelocity.toFixed(3)),
      consistencyScore: parseFloat(consistencyScore.toFixed(1)),
      biomechanicalEfficiency: parseFloat(biomechanicalEfficiency.toFixed(1)),
      ...(exercise.type === 'isometric' && { timeHeldSeconds: timer }),
    };
    onWorkoutComplete(session);
  };
  
  const getFeedbackStyles = () => {
    switch(feedback.type) {
        case 'critical': return { bg: 'bg-red-600/90', text: 'text-white', icon: <AlertTriangle /> };
        case 'adjustment': return { bg: 'bg-yellow-500/90', text: 'text-black', icon: <AlertTriangle /> };
        case 'positive': return { bg: 'bg-green-600/90', text: 'text-white', icon: <CheckCircle /> };
        default: return { bg: 'bg-slate-900/80', text: 'text-white', icon: null };
    }
  }
  
  const feedbackStyles = getFeedbackStyles();

  return (
    <div className="w-full max-w-4xl mx-auto flex flex-col items-center gap-4 animate-fade-in-up">
        <div className="relative w-full aspect-video bg-slate-200 rounded-2xl shadow-2xl border border-slate-300 overflow-hidden">
            <video ref={videoRef} className="hidden" autoPlay playsInline></video>
            <canvas ref={canvasRef} width={640} height={480} className="w-full h-full object-cover"></canvas>
            
            {isInitializing && ( <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/80 text-slate-800"><Loader2 className="w-12 h-12 animate-spin text-blue-600 mb-4" /><p>Starting camera...</p></div> )}
            {error && ( <div className="absolute inset-0 flex flex-col items-center justify-center bg-red-100/90 text-red-800 p-4 text-center"><CameraOff className="w-12 h-12 text-red-600 mb-4" /><p className="text-lg font-semibold mb-2">Camera Error</p><p className="text-sm text-red-700">{error}</p></div> )}

            {!isInitializing && !error && (
                <>
                    {/* Main Feedback */}
                    <div 
                      key={feedback.message} 
                      className={`absolute top-4 left-4 p-2 px-4 rounded-full font-semibold text-lg flex items-center gap-2 backdrop-blur-sm shadow-lg animate-pop-in ${feedbackStyles.bg} ${feedbackStyles.text} ${feedback.type === 'critical' ? 'animate-pulse-bg' : ''}`}
                    >
                       {feedbackStyles.icon} <span>{feedback.message}</span>
                    </div>
                    
                    {/* Form Score Gauge */}
                    <div className="absolute top-4 right-4 bg-slate-900/70 backdrop-blur-sm p-2 rounded-2xl">
                       <FormGauge score={formScore} />
                       <p className="text-xs font-semibold text-slate-200 mt-1 text-center">FORM</p>
                    </div>
                    
                    {/* Advanced Metrics Panel - Left Side */}
                    <div className="absolute left-4 top-24 space-y-2">
                        {physics && (
                            <MetricDisplay 
                                icon={<TrendingUp size={16} className="text-purple-400" />}
                                label="Stability"
                                value={`${physics.stability.toFixed(0)}%`}
                                color="text-purple-400"
                            />
                        )}
                        
                        {/* Movement Quality Indicator */}
                        {movementQuality && (
                            <div className="bg-slate-900/60 backdrop-blur-sm px-3 py-2 rounded-lg space-y-1">
                                <div className="flex items-center gap-2 mb-1">
                                    <span className="text-[10px] uppercase font-semibold tracking-wider text-white">Quality</span>
                                    <span className={`text-sm font-bold ${movementQuality.overall >= 85 ? 'text-green-400' : movementQuality.overall >= 70 ? 'text-yellow-400' : 'text-orange-400'}`}>
                                        {movementQuality.overall.toFixed(0)}%
                                    </span>
                                </div>
                                <div className="flex gap-2 text-[9px]">
                                    <span className="text-blue-300">S:{movementQuality.smoothness.toFixed(0)}</span>
                                    <span className="text-purple-300">C:{movementQuality.control.toFixed(0)}</span>
                                    <span className="text-cyan-300">E:{movementQuality.efficiency.toFixed(0)}</span>
                                </div>
                            </div>
                        )}
                        
                        {/* Fatigue Alert */}
                        {fatigueLevel > 40 && (
                            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg backdrop-blur-sm ${
                                fatigueLevel > 70 ? 'bg-red-500/30 text-red-300' : 'bg-orange-500/30 text-orange-300'
                            }`}>
                                <span className="text-xs font-bold">⚠️ Fatigue: {fatigueLevel.toFixed(0)}%</span>
                            </div>
                        )}
                        
                        {/* Injury Risk Warning */}
                        {injuryRisk > 40 && (
                            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg backdrop-blur-sm ${
                                injuryRisk > 60 ? 'bg-red-500/30 text-red-300 animate-pulse' : 'bg-orange-500/30 text-orange-300'
                            }`}>
                                <span className="text-xs font-bold">🚨 Risk: {injuryRisk.toFixed(0)}%</span>
                            </div>
                        )}
                        
                        {/* Risk Indicator */}
                        {feedback.biomechanicalRisk && (
                            <RiskIndicator risk={feedback.biomechanicalRisk} />
                        )}
                        
                        {/* Confidence Score */}
                        {feedback.confidence !== undefined && (
                            <MetricDisplay 
                                icon={<Activity size={16} className="text-blue-400" />}
                                label="AI Confidence"
                                value={`${(feedback.confidence * 100).toFixed(0)}%`}
                                color="text-blue-400"
                            />
                        )}
                    </div>
                    
                    {/* Rep Counter / Timer - Bottom Right */}
                    <div className="absolute bottom-4 right-4 text-center bg-slate-900/70 backdrop-blur-sm p-4 rounded-2xl">
                        {exercise.type === 'reps' ? (
                            <>
                                <p key={repCount} className={`text-white font-spartan font-bold text-7xl drop-shadow-lg ${repCount > 0 ? 'animate-rep-count' : 'animate-pop-in'}`}>{repCount}</p>
                                <p className="text-slate-200 font-semibold text-xl -mt-2">REPS</p>
                            </>
                        ) : (
                             <>
                                <p key={timer} className="text-white font-spartan font-bold text-7xl drop-shadow-lg animate-tick-down">{timer}</p>
                                <p className="text-slate-200 font-semibold text-xl -mt-2 flex items-center justify-center gap-2"><Timer size={20}/> TIME</p>
                            </>
                        )}
                    </div>
                    
                    {/* Biomechanics Info - Bottom Left */}
                    {biomechanics && (
                        <div className="absolute bottom-4 left-4 bg-slate-900/70 backdrop-blur-sm p-3 rounded-xl text-white text-xs space-y-1">
                            <div className="flex justify-between gap-4">
                                <span className="text-slate-400">Balance:</span>
                                <span className="font-bold">{biomechanics.balanceScore.toFixed(0)}%</span>
                            </div>
                            <div className="flex justify-between gap-4">
                                <span className="text-slate-400">Asymmetry:</span>
                                <span className="font-bold">{biomechanics.asymmetry.toFixed(1)}°</span>
                            </div>
                            <div className="flex justify-between gap-4">
                                <span className="text-slate-400">Depth:</span>
                                <span className="font-bold">{biomechanics.depthEstimate.toFixed(2)}m</span>
                            </div>
                        </div>
                    )}
                    
                    {/* Tracking Confidence Legend - Top Center */}
                    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-slate-900/80 backdrop-blur-sm px-4 py-2 rounded-lg text-white text-xs">
                        <div className="flex items-center gap-3">
                            <span className="text-slate-300 font-semibold">Tracking:</span>
                            <div className="flex items-center gap-1.5">
                                <div className="w-2.5 h-2.5 rounded-full bg-emerald-500"></div>
                                <span className="text-[10px]">Excellent</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <div className="w-2.5 h-2.5 rounded-full bg-teal-500"></div>
                                <span className="text-[10px]">Good</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <div className="w-2.5 h-2.5 rounded-full bg-sky-500"></div>
                                <span className="text-[10px]">OK</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <div className="w-2.5 h-2.5 rounded-full bg-yellow-500"></div>
                                <span className="text-[10px]">Low</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <div className="w-2.5 h-2.5 rounded-full bg-orange-500"></div>
                                <span className="text-[10px]">Poor</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <div className="w-2.5 h-2.5 rounded-full bg-red-500"></div>
                                <span className="text-[10px]">Issue</span>
                            </div>
                        </div>
                    </div>
                </>
            )}
        </div>
        <button onClick={handleFinish} disabled={isInitializing || !!error} className="w-full sm:w-auto bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-lg transition duration-300 disabled:bg-slate-400 disabled:cursor-not-allowed text-lg shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40">
            Finish Workout
        </button>
    </div>
  );
};

export default PoseCanvas;