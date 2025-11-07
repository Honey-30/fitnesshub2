import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell, CartesianGrid } from 'recharts';
import { MacronutrientInfo } from '../types';

interface MacroDistributionChartProps {
  consumedMacros: MacronutrientInfo;
  targetMacros: MacronutrientInfo;
}

const COLORS = {
  protein: '#a855f7', // purple-500
  carbs: '#eab308',   // yellow-500
  fat: '#22c55e',     // green-500
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const originalConsumed = data.originalConsumed;
    const target = data.target;
    const calories = Math.round(originalConsumed * (label === 'Fat' ? 9 : 4));
    
    return (
      <div className="bg-white p-3 border border-slate-200 rounded-md text-sm shadow-lg">
        <p className="font-bold text-slate-800 mb-1">{label}</p>
        <p className="text-slate-600">Consumed: {originalConsumed.toFixed(0)}g / {target.toFixed(0)}g</p>
        <p className="text-slate-400 text-xs">~{calories} kcal</p>
      </div>
    );
  }
  return null;
};

const MacroDistributionChart: React.FC<MacroDistributionChartProps> = ({ consumedMacros, targetMacros }) => {
  const data = [
    { name: 'Protein', originalConsumed: consumedMacros.protein, target: targetMacros.protein },
    { name: 'Carbs', originalConsumed: consumedMacros.carbs, target: targetMacros.carbs },
    { name: 'Fat', originalConsumed: consumedMacros.fat, target: targetMacros.fat },
  ].map(item => ({
    ...item,
    consumed: item.originalConsumed, // The colored part of the bar
    remaining: Math.max(0, item.target - item.originalConsumed), // The grey part
  }));
  
  const totalConsumed = data.reduce((sum, item) => sum + item.originalConsumed, 0);

  return (
    <div className="glass-card p-4 rounded-2xl">
       <h3 className="text-lg font-bold font-spartan text-slate-900 text-center mb-2">Daily Intake vs. Goals</h3>
       {totalConsumed === 0 ? (
         <div className="flex items-center justify-center h-64">
           <p className="text-slate-500">Log food to see your progress.</p>
         </div>
       ) : (
        <div style={{ width: '100%', height: 250 }}>
            <ResponsiveContainer>
                <BarChart data={data} layout="vertical" margin={{ top: 5, right: 20, left: -20, bottom: 5 }} barCategoryGap="35%">
                    <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" hide />
                    <YAxis 
                        type="category" 
                        dataKey="name" 
                        stroke="#64748b" 
                        fontSize={14}
                        axisLine={false}
                        tickLine={false}
                        width={80}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(200, 200, 200, 0.2)' }}/>
                    
                    <Bar dataKey="consumed" stackId="a" radius={[8, 0, 0, 8]}>
                       {data.map((entry) => (
                           <Cell key={`cell-${entry.name}`} fill={COLORS[entry.name.toLowerCase() as keyof typeof COLORS]} />
                       ))}
                    </Bar>
                    <Bar dataKey="remaining" stackId="a" fill="#e2e8f0" radius={[0, 8, 8, 0]} />
                </BarChart>
            </ResponsiveContainer>
        </div>
       )}
    </div>
  );
};

export default MacroDistributionChart;