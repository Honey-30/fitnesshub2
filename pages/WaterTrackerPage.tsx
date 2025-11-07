import React, { useState } from 'react';
import { CalculationResults } from '../types';
import WaterTracker from '../components/WaterTracker';

interface WaterTrackerPageProps {
  targets: CalculationResults;
}

const WaterTrackerPage: React.FC<WaterTrackerPageProps> = ({ targets }) => {
  const [waterConsumed, setWaterConsumed] = useState(0);

  const handleAddWater = (amount: number) => {
    setWaterConsumed(prev => prev + amount);
  };

  return (
    <div className="space-y-8 animate-fade-in">
      <div className="max-w-2xl mx-auto">
        <WaterTracker 
          targetMl={targets.waterIntake}
          consumedMl={waterConsumed}
          onAddWater={handleAddWater}
        />
      </div>
    </div>
  );
};

export default WaterTrackerPage;