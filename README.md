<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# AI Fitness Hub 🏋️‍♂️

A comprehensive AI-powered fitness application with real-time pose detection, nutrition tracking, and personalized workout guidance. Built with cutting-edge technologies to deliver a premium, sophisticated fitness experience.

## 🌟 Features

### 🎯 AI Pose Detection & Form Analysis
- **Real-time pose tracking** using MediaPipe Pose with 33 body landmarks
- **Advanced HUD visualization** with hexagonal keypoints, grid overlay, and data callouts
- **Biomechanical angle measurements** for critical joints (shoulders, elbows, hips, knees)
- **AI-powered form feedback** using Google Gemini for personalized corrections
- **Rep counting** with automatic exercise detection
- **Risk assessment** with color-coded confidence indicators

### 🍎 Nutrition & Macros Tracker
- **Comprehensive food database** with 100+ common foods
- **Smart food search** with real-time filtering
- **Macro tracking** (Protein, Carbs, Fats) with visual progress indicators
- **Calorie monitoring** with daily goals
- **Food logging** with portion control and meal organization

### 💧 Water Intake Tracker
- **Daily hydration goals** (customizable target)
- **Visual progress tracking** with animated water waves
- **Quick add buttons** for common serving sizes
- **Streak tracking** to build healthy habits
- **Premium animations** for engaging user experience

### 🏃 Workout Library
- **10+ exercises** including Push-ups, Squats, Planks, Lunges, and more
- **AI coaching** with real-time form corrections
- **Exercise cards** with difficulty levels and muscle group targeting
- **Premium UI** with subtle animations and glass-morphism design

## 🛠️ Tech Stack & Why We Chose Them

### **Frontend Framework**
- **React 19.2.0** - Latest version for optimal performance, concurrent rendering, and improved hooks
- **TypeScript** - Type safety, better developer experience, and fewer runtime errors

### **Build Tool**
- **Vite 6.2.0** - Lightning-fast HMR (Hot Module Replacement), optimized builds, and modern ES modules support
- **Why?** Significantly faster than webpack, better developer experience with instant updates

### **AI & Machine Learning**
- **MediaPipe Pose 0.5.1635988162** - Google's state-of-the-art pose detection model
  - **Why?** Industry-leading accuracy (33 landmarks), real-time performance in browser, no backend required
- **Google Gemini 2.0 Flash** - Advanced AI for workout coaching and form analysis
  - **Why?** Multimodal capabilities, fast response times, natural language understanding for personalized feedback

### **Styling & UI**
- **Tailwind CSS** - Utility-first CSS framework for rapid UI development
  - **Why?** Consistent design system, no CSS file bloat, highly customizable
- **Lucide React** - Modern icon library with 1000+ icons
  - **Why?** Lightweight, consistent design, tree-shakeable for smaller bundle sizes
- **Custom Animations** - Hand-crafted CSS animations with cubic-bezier easing
  - **Why?** Premium feel, smooth transitions, better performance than JS animations

### **State Management**
- **React Hooks (useState, useEffect, useCallback)** - Modern React patterns
  - **Why?** No additional libraries needed, cleaner code, better performance
- **LocalStorage API** - Persistent data storage for user preferences
  - **Why?** No backend required, instant access, works offline

### **Canvas Rendering**
- **HTML5 Canvas 2D Context** - For custom pose visualization
  - **Why?** High-performance rendering, full control over graphics, hardware-accelerated
- **Custom HUD System** - Hexagonal grids, scan effects, and data callouts
  - **Why?** Unique premium aesthetic, professional biomechanical analysis display

### **Video Processing**
- **MediaStream API** - Access user's webcam
  - **Why?** Browser-native, no plugins required, privacy-focused (local processing)
- **RequestAnimationFrame** - Smooth 60fps rendering
  - **Why?** Optimal performance, battery-efficient, synchronized with display refresh

## 🎨 Design Philosophy

### **Premium & Sophisticated**
- Glass-morphism cards with backdrop blur
- Subtle gradients and depth effects
- Clean, minimal glow effects for important elements
- Professional color scheme (Slate + Cyan accents)

### **Performance-First**
- Optimized re-renders with React.memo and useCallback
- Efficient canvas operations with layer caching
- Debounced search and input handlers
- Lazy loading for heavy components

### **User Experience**
- Instant feedback on all interactions
- Smooth page transitions with staggered animations
- Clear visual hierarchy and information architecture
- Accessible color contrasts and readable typography

## 📦 Installation & Setup

**Prerequisites:** Node.js 16+ and npm/yarn

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Honey-30/fitnesshub2.git
   cd fitnesshub2
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Configure API Key:**
   - Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - The app has a built-in API settings page - no .env file needed!
   - Launch the app and navigate to Settings → API Key Settings
   - Or set `GEMINI_API_KEY` in `.env.local` if preferred

4. **Run the development server:**
   ```bash
   npm run dev
   ```

5. **Open in browser:**
   Navigate to `http://localhost:5173`

## 🚀 Building for Production

```bash
npm run build
npm run preview  # Preview production build locally
```

## 📱 Features Overview

### **Workout Page**
- Select from 10+ exercises
- Real-time pose detection with webcam
- AI coaching with form corrections
- Rep counter and workout timer
- Biomechanical angle measurements
- Risk assessment and confidence tracking

### **Macros Tracker Page**
- Search and add foods
- Track daily nutrition goals
- Visual macro distribution (pie charts)
- Progress circles for each macro
- Food log with calorie breakdown
- Daily progress chart

### **Water Tracker Page**
- Set daily hydration goals
- Quick add buttons (250ml, 500ml, 1L)
- Animated water waves
- Progress percentage
- Reset and adjust intake

### **Landing Page**
- BMI Calculator with gauge visualization
- Feature highlights
- Quick navigation to all tools
- Premium animations

## 🔧 API Configuration

The app uses Google Gemini AI for intelligent workout feedback. You have two options:

1. **In-App Settings** (Recommended):
   - Navigate to Settings page
   - Enter your API key
   - Click "Validate & Save"
   - Key is stored securely in browser localStorage

2. **Environment Variables**:
   - Create `.env.local` file
   - Add: `VITE_GEMINI_API_KEY=your_api_key_here`
   - Restart dev server

## 📊 Project Structure

```
ai-fitness-hub/
├── components/          # Reusable React components
│   ├── PoseCanvas.tsx   # Advanced pose visualization
│   ├── ExerciseSelection.tsx
│   ├── FoodSearchAndAdd.tsx
│   └── WaterTracker.tsx
├── pages/              # Main application pages
│   ├── WorkoutPage.tsx
│   ├── MacrosTrackerPage.tsx
│   ├── WaterTrackerPage.tsx
│   └── LandingPage.tsx
├── utils/              # Helper functions and algorithms
│   ├── poseAnalyzer.ts  # Pose analysis & angle calculations
│   ├── repCounter.ts    # Rep counting logic
│   └── calculator.ts    # BMI and nutrition calculations
├── services/           # External service integrations
│   └── geminiService.ts # AI coaching integration
├── hooks/              # Custom React hooks
│   └── usePoseDetection.ts
├── data/               # Static data
│   └── foodDatabase.ts
└── types.ts           # TypeScript type definitions
```

## 🎯 Why This Stack?

### **No Backend Required**
- All processing happens in the browser (MediaPipe runs locally)
- Privacy-first: Your video never leaves your device
- No server costs, instant deployment to static hosting
- Works offline after initial load

### **Modern & Future-Proof**
- React 19 with latest features
- TypeScript for maintainability
- Vite for blazing-fast development
- ES modules for optimal tree-shaking

### **AI-Powered Intelligence**
- Gemini 2.0 for natural language coaching
- MediaPipe for accurate pose detection
- Real-time biomechanical analysis
- Personalized feedback based on user form

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Links

- **Live Demo**: [AI Fitness Hub](https://ai.studio/apps/drive/1Nrv7KUPUtTeDC1HfWWjwFOsRPclGXIyc)
- **Repository**: [GitHub](https://github.com/Honey-30/fitnesshub2)
- **Issues**: [Report a Bug](https://github.com/Honey-30/fitnesshub2/issues)

---

<div align="center">
Made with ❤️ by Honey-30
</div>
