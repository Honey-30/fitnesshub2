import React, { useState } from 'react';
import { useForm, SubmitHandler, Controller } from 'react-hook-form';
import { UserData, Gender, Goal, ActivityLevel } from '../types';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Loader2, ArrowRight, ArrowLeft, User, Dumbbell, Target, Venus, Mars, Bed, Footprints, Bike, Flame, TrendingDown, Repeat, TrendingUp, Activity } from 'lucide-react';

interface BmiCalculatorFormProps {
  onCalculate: (data: UserData) => void;
  isLoading: boolean;
}

// Fix: With `valueAsNumber: true` in the register function, react-hook-form
// handles the coercion. We should use `z.number()` to validate the result, which also
// resolves the TypeScript type conflicts with the resolver and form handler.
const schema = z.object({
  gender: z.nativeEnum(Gender),
  // Fix: Replaced `invalid_type_error` with `message` as the property is not recognized in the current environment's Zod type definitions.
  // `message` serves as a fallback for type errors (e.g., when input is not a number) while allowing more specific messages for min/max rules.
  age: z.number({ message: 'Age must be a number.' }).min(15, "Must be at least 15").max(99, "Must be 99 or younger"),
  height: z.number({ message: 'Height must be a number.' }).min(100, "Must be at least 100cm").max(250, "Height must be at most 250cm"),
  weight: z.number({ message: 'Weight must be a number.' }).min(30, "Weight must be at least 30kg").max(300, "Weight must be at most 300kg"),
  activityLevel: z.nativeEnum(ActivityLevel),
  goal: z.nativeEnum(Goal),
});


const steps = [
  { id: 1, name: 'About You', fields: ['gender', 'age', 'height', 'weight'] },
  { id: 2, name: 'Your Lifestyle', fields: ['activityLevel'] },
  { id: 3, name: 'Your Goal', fields: ['goal'] },
];

const OptionCard: React.FC<{ icon: React.ReactNode; label: string; value: string; isSelected: boolean; onClick: () => void; ariaLabel: string; }> = ({ icon, label, isSelected, onClick, ariaLabel }) => (
  <button
    type="button"
    onClick={onClick}
    aria-label={ariaLabel}
    className={`w-full p-4 glass-card rounded-xl flex flex-col items-center justify-center space-y-2 transition-all duration-300 transform active:scale-95 glow-border ${
      isSelected ? 'shadow-lg shadow-blue-500/20 border-blue-500/50 scale-105' : 'border-slate-200'
    }`}
  >
    {icon}
    <span className="font-semibold text-slate-800 text-sm text-center">{label}</span>
  </button>
);


const BmiCalculatorForm: React.FC<BmiCalculatorFormProps> = ({ onCalculate, isLoading }) => {
  const [currentStep, setCurrentStep] = useState(0);
  
  const { register, handleSubmit, formState: { errors }, control, trigger, watch } = useForm<UserData>({
    resolver: zodResolver(schema),
    mode: 'onChange',
    defaultValues: {
      gender: Gender.Male,
      age: 25,
      height: 180,
      weight: 75,
      activityLevel: ActivityLevel.Moderate,
      goal: Goal.Maintain,
    }
  });

  const onSubmit: SubmitHandler<UserData> = (data) => onCalculate(data);
  
  const nextStep = async () => {
    const fields = steps[currentStep].fields;
    const output = await trigger(fields as any, { shouldFocus: true });
    if (!output) return;

    if (currentStep < steps.length - 1) {
      setCurrentStep(step => step + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(step => step - 1);
    }
  };
  
  const genderValue = watch('gender');
  const activityLevelValue = watch('activityLevel');
  const goalValue = watch('goal');

  return (
    <div className="max-w-2xl mx-auto glass-card p-6 sm:p-8 rounded-2xl shadow-2xl shadow-black/5">
      <form onSubmit={handleSubmit(onSubmit)}>
        {/* Progress Bar */}
        <div className="mb-8">
            <div className="flex justify-between mb-2">
                {steps.map((step, index) => (
                    <span key={step.id} className={`text-xs font-semibold ${index <= currentStep ? 'text-blue-600' : 'text-slate-500'}`}>
                        {step.name}
                    </span>
                ))}
            </div>
          <div className="w-full bg-slate-200 rounded-full h-2">
            <div
              className="bg-gradient-to-r from-blue-500 to-sky-500 h-2 rounded-full transition-transform duration-500 ease-out origin-left"
              style={{ transform: `scaleX(${currentStep / (steps.length - 1)})` }}
            />
          </div>
        </div>

        {/* Step 1: About You */}
        {currentStep === 0 && (
          <div className="space-y-6 animate-fade-in-up">
             <h2 className="text-2xl font-bold font-spartan text-center text-slate-800 flex items-center justify-center gap-3"><User/> About You</h2>
             <div className="grid grid-cols-2 gap-4">
                 <Controller
                    name="gender"
                    control={control}
                    render={({ field }) => (
                        <>
                         <OptionCard icon={<Mars size={32} className="text-blue-600"/>} label="Male" value={Gender.Male} isSelected={genderValue === Gender.Male} onClick={() => field.onChange(Gender.Male)} ariaLabel="Select Male as gender" />
                         <OptionCard icon={<Venus size={32} className="text-pink-500"/>} label="Female" value={Gender.Female} isSelected={genderValue === Gender.Female} onClick={() => field.onChange(Gender.Female)} ariaLabel="Select Female as gender" />
                        </>
                    )}
                 />
             </div>
             <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                 <div>
                    <label htmlFor="age" className="block text-sm font-medium text-slate-600 mb-2">Age</label>
                    <input id="age" type="number" {...register('age', { valueAsNumber: true })} aria-label="Enter your age" className="w-full bg-white border border-slate-300 rounded-md p-3 text-slate-800 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition duration-200" />
                    {errors.age && <p className="text-red-500 text-xs mt-1">{errors.age?.message}</p>}
                 </div>
                 <div>
                    <label htmlFor="height" className="block text-sm font-medium text-slate-600 mb-2">Height (cm)</label>
                    <input id="height" type="number" {...register('height', { valueAsNumber: true })} aria-label="Enter your height in centimeters" className="w-full bg-white border border-slate-300 rounded-md p-3 text-slate-800 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition duration-200" />
                    {errors.height && <p className="text-red-500 text-xs mt-1">{errors.height?.message}</p>}
                 </div>
                 <div>
                    <label htmlFor="weight" className="block text-sm font-medium text-slate-600 mb-2">Weight (kg)</label>
                    <input id="weight" type="number" {...register('weight', { valueAsNumber: true })} aria-label="Enter your weight in kilograms" className="w-full bg-white border border-slate-300 rounded-md p-3 text-slate-800 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition duration-200" />
                    {errors.weight && <p className="text-red-500 text-xs mt-1">{errors.weight?.message}</p>}
                 </div>
             </div>
          </div>
        )}

        {/* Step 2: Your Lifestyle */}
        {currentStep === 1 && (
            <div className="space-y-6 animate-fade-in-up">
                <h2 className="text-2xl font-bold font-spartan text-center text-slate-800 flex items-center justify-center gap-3"><Dumbbell /> Your Lifestyle</h2>
                <p className="text-center text-slate-600">How active are you on a weekly basis?</p>
                <Controller
                    name="activityLevel"
                    control={control}
                    render={({ field }) => (
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                            <OptionCard label="Sedentary" value={ActivityLevel.Sedentary} isSelected={activityLevelValue === ActivityLevel.Sedentary} onClick={() => field.onChange(ActivityLevel.Sedentary)} icon={<Bed size={24} className="text-blue-600"/>} ariaLabel="Select Sedentary activity level" />
                            <OptionCard label="Light" value={ActivityLevel.Light} isSelected={activityLevelValue === ActivityLevel.Light} onClick={() => field.onChange(ActivityLevel.Light)} icon={<Footprints size={24} className="text-blue-600"/>} ariaLabel="Select Light activity level" />
                            <OptionCard label="Moderate" value={ActivityLevel.Moderate} isSelected={activityLevelValue === ActivityLevel.Moderate} onClick={() => field.onChange(ActivityLevel.Moderate)} icon={<Bike size={24} className="text-blue-600"/>} ariaLabel="Select Moderate activity level" />
                            <OptionCard label="Active" value={ActivityLevel.Active} isSelected={activityLevelValue === ActivityLevel.Active} onClick={() => field.onChange(ActivityLevel.Active)} icon={<Activity size={24} className="text-blue-600"/>} ariaLabel="Select Active activity level" />
                            <OptionCard label="Very Active" value={ActivityLevel.VeryActive} isSelected={activityLevelValue === ActivityLevel.VeryActive} onClick={() => field.onChange(ActivityLevel.VeryActive)} icon={<Flame size={24} className="text-blue-600"/>} ariaLabel="Select Very Active activity level" />
                        </div>
                    )}
                />
            </div>
        )}

        {/* Step 3: Your Goal */}
        {currentStep === 2 && (
            <div className="space-y-6 animate-fade-in-up">
                 <h2 className="text-2xl font-bold font-spartan text-center text-slate-800 flex items-center justify-center gap-3"><Target /> Your Goal</h2>
                 <p className="text-center text-slate-600">What is your primary fitness goal?</p>
                 <Controller
                    name="goal"
                    control={control}
                    render={({ field }) => (
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                            <OptionCard label="Lose Weight" value={Goal.Lose} isSelected={goalValue === Goal.Lose} onClick={() => field.onChange(Goal.Lose)} icon={<TrendingDown size={24} className="text-blue-600"/>} ariaLabel="Select goal: Lose Weight" />
                            <OptionCard label="Maintain" value={Goal.Maintain} isSelected={goalValue === Goal.Maintain} onClick={() => field.onChange(Goal.Maintain)} icon={<Repeat size={24} className="text-blue-600"/>} ariaLabel="Select goal: Maintain Weight" />
                            <OptionCard label="Gain Muscle" value={Goal.Gain} isSelected={goalValue === Goal.Gain} onClick={() => field.onChange(Goal.Gain)} icon={<TrendingUp size={24} className="text-blue-600"/>} ariaLabel="Select goal: Gain Muscle" />
                        </div>
                    )}
                 />
            </div>
        )}


        {/* Navigation */}
        <div className="mt-8 flex justify-between items-center">
          {currentStep > 0 ? (
            <button type="button" onClick={prevStep} className="inline-flex items-center bg-white hover:bg-slate-100 text-slate-700 font-semibold py-2 px-4 rounded-md transition duration-300 border border-slate-300">
              <ArrowLeft className="mr-2 h-4 w-4" /> Back
            </button>
          ) : <div />}

          {currentStep < steps.length - 1 ? (
            <button type="button" onClick={nextStep} className="inline-flex items-center bg-gradient-to-r from-blue-600 to-sky-500 hover:from-blue-500 hover:to-sky-400 text-white font-semibold py-2 px-4 rounded-md transition duration-300">
              Next <ArrowRight className="ml-2 h-4 w-4" />
            </button>
          ) : (
            <button type="submit" disabled={isLoading} className="inline-flex items-center justify-center bg-gradient-to-r from-blue-600 to-sky-500 hover:from-blue-500 hover:to-sky-400 text-white font-bold py-2 px-4 rounded-md transition duration-300 disabled:from-blue-600/50 disabled:to-sky-500/50 disabled:cursor-not-allowed shadow-lg shadow-blue-500/20 hover:shadow-2xl hover:shadow-blue-500/30">
                {isLoading ? <><Loader2 className="mr-2 h-5 w-5 animate-spin" /> Calculating...</> : 'Calculate & Get AI Plan'}
            </button>
          )}
        </div>
      </form>
    </div>
  );
};

export default BmiCalculatorForm;