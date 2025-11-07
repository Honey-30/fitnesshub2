import { Bot, Droplets, GlassWater } from 'lucide-react';
import React, { useEffect, useState } from 'react';

interface WaterTrackerProps {
  targetMl: number;
  consumedMl: number;
  onAddWater: (amount: number) => void;
}

const WaterTracker: React.FC<WaterTrackerProps> = ({ targetMl, consumedMl, onAddWater }) => {
  const percentage = targetMl > 0 ? Math.min((consumedMl / targetMl) * 100, 100) : 0;
  const [animate, setAnimate] = useState(false);

  useEffect(() => {
    if (consumedMl > 0) {
      setAnimate(true);
      const timer = setTimeout(() => setAnimate(false), 1000);
      return () => clearTimeout(timer);
    }
  }, [consumedMl]);

  return (
    <div className="glass-card p-8 rounded-2xl space-y-8">
      <div className="flex items-center justify-between">
        <h3 className="text-2xl font-bold font-spartan text-slate-900 flex items-center gap-2">
          <Droplets className="text-blue-500" size={28} />
          Water Intake
        </h3>
        <div className="text-right">
          <div className="text-3xl font-bold text-slate-900">{Math.round(percentage)}%</div>
          <div className="text-sm text-slate-500">{consumedMl} / {targetMl} ml</div>
        </div>
      </div>
      
      {/* Animated Water Container */}
      <div className="relative w-full h-96 bg-white rounded-3xl overflow-hidden border-4 border-slate-200 shadow-2xl">
        {/* Water with wave animation */}
        <div 
          className="absolute bottom-0 left-0 w-full transition-all duration-1000 ease-out"
          style={{ 
            height: `${percentage}%`,
          }}
        >
          {/* Water gradient */}
          <div className="absolute inset-0 bg-gradient-to-t from-blue-500 via-blue-400 to-cyan-300 opacity-90"></div>
          
          {/* Animated waves */}
          <div className="absolute top-0 left-0 w-full h-8 overflow-hidden">
            <svg className="absolute w-full h-full" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320" preserveAspectRatio="none">
              <path 
                fill="rgba(255,255,255,0.3)" 
                fillOpacity="1" 
                d="M0,96L48,112C96,128,192,160,288,160C384,160,480,128,576,122.7C672,117,768,139,864,138.7C960,139,1056,117,1152,106.7C1248,96,1344,96,1392,96L1440,96L1440,0L1392,0C1344,0,1248,0,1152,0C1056,0,960,0,864,0C768,0,672,0,576,0C480,0,384,0,288,0C192,0,96,0,48,0L0,0Z"
                className="animate-wave"
              ></path>
            </svg>
            <svg className="absolute w-full h-full" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320" preserveAspectRatio="none" style={{ animationDelay: '-2s' }}>
              <path 
                fill="rgba(255,255,255,0.2)" 
                fillOpacity="1" 
                d="M0,64L48,80C96,96,192,128,288,128C384,128,480,96,576,90.7C672,85,768,107,864,106.7C960,107,1056,85,1152,74.7C1248,64,1344,64,1392,64L1440,64L1440,0L1392,0C1344,0,1248,0,1152,0C1056,0,960,0,864,0C768,0,672,0,576,0C480,0,384,0,288,0C192,0,96,0,48,0L0,0Z"
                className="animate-wave-slow"
              ></path>
            </svg>
          </div>
          
          {/* Animated bubbles */}
          {animate && (
            <>
              {[...Array(15)].map((_, i) => (
                <div
                  key={i}
                  className="absolute rounded-full bg-white opacity-40 animate-bubble"
                  style={{
                    width: `${Math.random() * 15 + 5}px`,
                    height: `${Math.random() * 15 + 5}px`,
                    left: `${Math.random() * 100}%`,
                    bottom: '0%',
                    animationDelay: `${Math.random() * 0.5}s`,
                    animationDuration: `${Math.random() * 2 + 2}s`
                  }}
                ></div>
              ))}
            </>
          )}
        </div>
        
        {/* Volume markers */}
        <div className="absolute inset-y-0 left-4 flex flex-col justify-between py-4 pointer-events-none">
          {[100, 75, 50, 25, 0].map((mark) => (
            <div key={mark} className="flex items-center gap-2">
              <div className="w-6 h-0.5 bg-slate-300"></div>
              <span className="text-xs font-semibold text-slate-400">{mark}%</span>
            </div>
          ))}
        </div>
        
        {/* Glass shine effect */}
        <div className="absolute inset-0 bg-gradient-to-br from-white/20 via-transparent to-transparent pointer-events-none"></div>
        <div className="absolute top-4 right-4 w-24 h-48 bg-white/10 blur-2xl rounded-full pointer-events-none"></div>
      </div>

      <div className="grid grid-cols-2 gap-4 pt-2">
        <button 
          onClick={() => onAddWater(250)}
          className="flex items-center justify-center gap-2 bg-white hover:bg-slate-100 text-slate-700 font-semibold py-2 px-4 rounded-md transition-all duration-300 transform active:scale-95 border border-slate-300"
          aria-label="Add 250 milliliters of water"
        >
          <GlassWater size={18} /> Add Glass
        </button>
        <button 
          onClick={() => onAddWater(500)}
          className="flex items-center justify-center gap-2 bg-white hover:bg-slate-100 text-slate-700 font-semibold py-2 px-4 rounded-md transition-all duration-300 transform active:scale-95 border border-slate-300"
          aria-label="Add 500 milliliters of water"
        >
          <Bot size={18} /> Add Bottle
        </button>
      </div>
    </div>
  );
};

export default WaterTracker;