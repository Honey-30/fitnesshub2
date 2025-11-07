import React, { useState, useEffect } from 'react';

interface BmiGaugeProps {
  bmiValue: number;
  bmiCategory: string;
}

const getBmiConfig = (bmi: number) => {
  if (bmi < 18.5) return { color: 'text-sky-500', category: 'Underweight' };
  if (bmi < 25) return { color: 'text-green-500', category: 'Healthy Weight' };
  if (bmi < 30) return { color: 'text-yellow-500', category: 'Overweight' };
  return { color: 'text-red-500', category: 'Obesity' };
};

const valueToAngle = (value: number, min: number, max: number) => {
  const clampedValue = Math.max(min, Math.min(value, max));
  return ((clampedValue - min) / (max - min)) * 180;
};

const polarToCartesian = (centerX: number, centerY: number, radius: number, angleInDegrees: number) => {
  const angleInRadians = (angleInDegrees - 90) * Math.PI / 180.0;
  return {
    x: centerX + (radius * Math.cos(angleInRadians)),
    y: centerY + (radius * Math.sin(angleInRadians))
  };
};

const describeArc = (x: number, y: number, radius: number, startAngle: number, endAngle: number) => {
  const start = polarToCartesian(x, y, radius, endAngle);
  const end = polarToCartesian(x, y, radius, startAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
  return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`;
};

const BmiGauge: React.FC<BmiGaugeProps> = ({ bmiValue, bmiCategory }) => {
  const [angle, setAngle] = useState(0);
  const config = getBmiConfig(bmiValue);
  const minBmi = 15;
  const maxBmi = 40;

  useEffect(() => {
    const targetAngle = valueToAngle(bmiValue, minBmi, maxBmi);
    const timeoutId = setTimeout(() => setAngle(targetAngle), 100);
    return () => clearTimeout(timeoutId);
  }, [bmiValue]);

  // Define the angle ranges for each BMI category
  const underweightEnd = valueToAngle(18.5, minBmi, maxBmi);
  const healthyEnd = valueToAngle(25, minBmi, maxBmi);
  const overweightEnd = valueToAngle(30, minBmi, maxBmi);

  return (
    <div className="flex flex-col items-center justify-center p-4">
       <h2 className="text-xl font-bold font-spartan text-slate-800 mb-4">Your Body Mass Index (BMI)</h2>
      <svg viewBox="0 0 200 120" className="w-full max-w-xs drop-shadow-lg">
        <defs>
          <linearGradient id="needleGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#475569" />
            <stop offset="100%" stopColor="#1e293b" />
          </linearGradient>
          <filter id="needleShadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="0" dy="1" stdDeviation="1.5" floodColor="#000" floodOpacity="0.2" />
          </filter>
        </defs>

        {/* Gauge Background Plate */}
        <path d={describeArc(100, 100, 80, 0, 180)} fill="none" stroke="#e2e8f0" strokeWidth="20" />
        
        {/* BMI Category Zones */}
        <path d={describeArc(100, 100, 80, 0, underweightEnd)} fill="none" stroke="#38bdf8" strokeWidth="20" />
        <path d={describeArc(100, 100, 80, underweightEnd, healthyEnd)} fill="none" stroke="#4ade80" strokeWidth="20" />
        <path d={describeArc(100, 100, 80, healthyEnd, overweightEnd)} fill="none" stroke="#facc15" strokeWidth="20" />
        <path d={describeArc(100, 100, 80, overweightEnd, 180)} fill="none" stroke="#f87171" strokeWidth="20" />

        {/* Labels */}
        <text x="20" y="118" textAnchor="middle" fontSize="12" fill="#64748b" className="font-semibold">{minBmi}</text>
        <text x="180" y="118" textAnchor="middle" fontSize="12" fill="#64748b" className="font-semibold">{maxBmi}+</text>

        {/* Needle */}
        <g 
          transform={`rotate(${angle} 100 100)`} 
          style={{ transition: 'transform 1.5s cubic-bezier(0.16, 1, 0.3, 1)' }} 
        >
          <path d="M 100 25 L 105 100 L 95 100 Z" fill="url(#needleGradient)" filter="url(#needleShadow)" />
          <circle cx="100" cy="100" r="8" fill="#f1f5f9" stroke="#cbd5e1" strokeWidth="2" />
          <circle cx="100" cy="100" r="4" fill="#334155" />
        </g>
      </svg>
      <div className="text-center -mt-16">
        <p className="text-5xl font-bold font-mono text-slate-800 drop-shadow-sm">{bmiValue.toFixed(1)}</p>
        <p className={`text-xl font-semibold ${config.color}`}>{bmiCategory}</p>
      </div>
    </div>
  );
};

export default BmiGauge;