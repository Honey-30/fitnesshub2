import mediaPipeCamera from '@mediapipe/camera_utils';
import mediaPipePose from '@mediapipe/pose';
import { useEffect, useRef, useState } from 'react';

interface UsePoseDetectionProps {
  onResults: (results: any) => void;
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  active: boolean;
}

export const usePoseDetection = ({ onResults, videoRef, canvasRef, active }: UsePoseDetectionProps) => {
  const [error, setError] = useState<string | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);
  
  // These refs are now correctly at the top level of the hook.
  const onResultsRef = useRef(onResults);
  const poseRef = useRef<any | null>(null);
  const cameraRef = useRef<any | null>(null);

  useEffect(() => {
    onResultsRef.current = onResults;
  }, [onResults]);
  
  useEffect(() => {
    // The pose and camera instances are now managed by refs declared outside this effect.
    if (!active) {
      if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
      }
       if (poseRef.current) {
        poseRef.current.close();
        poseRef.current = null;
      }
      return;
    }

    if (!videoRef.current || !canvasRef.current) {
      return;
    }

    const videoElement = videoRef.current;
    
    // Initialize pose only if it doesn't exist
    if (!poseRef.current) {
        const pose = new mediaPipePose.Pose({
          locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
        });

        pose.setOptions({
          modelComplexity: 2, // Maximum complexity for ultra-precise tracking
          smoothLandmarks: true,
          enableSegmentation: true, // Enable for better depth perception
          minDetectionConfidence: 0.8, // Increased to 80% for maximum precision
          minTrackingConfidence: 0.8, // Increased to 80% for ultra-stable tracking
          smoothSegmentation: true,
        });
        
        poseRef.current = pose;
    }
    
    // Always set the latest onResults callback
    poseRef.current.onResults((results: any) => {
      onResultsRef.current(results);
    });
    
    // Initialize camera only if it doesn't exist
    if (!cameraRef.current) {
        const camera = new mediaPipeCamera.Camera(videoElement, {
          onFrame: async () => {
            if (videoElement && !videoElement.paused && videoElement.readyState >= 3 && poseRef.current) {
                 await poseRef.current.send({ image: videoElement });
            }
          },
          width: 640,
          height: 480,
        });
        cameraRef.current = camera;
    }
    
    const startCamera = async () => {
        setIsInitializing(true);
        setError(null);
        try {
            if (cameraRef.current) {
                await cameraRef.current.start();
            }
        } catch (e) {
             console.error("Camera access failed:", e);
             if(e instanceof Error) {
                setError(`Camera access denied. Please enable camera permissions in your browser settings. Error: ${e.message}`);
             } else {
                setError("Camera access denied. Please enable camera permissions in your browser settings.");
             }
        } finally {
            setIsInitializing(false);
        }
    }

    startCamera();

    return () => {
      // The cleanup will stop the camera but won't destroy the instance
      // unless the component unmounts fully, handled by the `active` check.
      if (cameraRef.current) {
        cameraRef.current.stop();
      }
    };
  }, [active, videoRef, canvasRef]);

  return { isInitializing, error };
};
