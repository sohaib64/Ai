import React from 'react';
import { Brain, Zap, TrendingUp, Database, Cpu, Code, Layers, Shield, Sparkles, ChevronRight, Activity, BarChart3, Target, CheckCircle2, Rocket, Award, Server, Globe } from 'lucide-react';

const AboutPage = ({ modelInfo }) => {
  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 relative overflow-hidden pt-28 pb-20">
      
      {/* Abstract Background Blobs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-[-10%] w-[500px] h-[500px] bg-blue-600/10 rounded-full blur-[120px] animate-pulse"></div>
        <div className="absolute bottom-20 right-[-10%] w-[500px] h-[500px] bg-purple-600/10 rounded-full blur-[120px] animate-pulse delay-700"></div>
      </div>

      <style>{`
        .glass-card {
          background: rgba(30, 41, 59, 0.4);
          backdrop-filter: blur(12px);
          border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .tech-chip {
          background: rgba(59, 130, 246, 0.1);
          border: 1px solid rgba(59, 130, 246, 0.2);
          transition: all 0.3s ease;
        }
        .tech-chip:hover {
          background: rgba(59, 130, 246, 0.2);
          transform: translateY(-2px);
          box-shadow: 0 5px 15px rgba(59, 130, 246, 0.2);
        }
      `}</style>

      <div className="container mx-auto px-6 relative z-10">
        
        {/* Hero Section */}
        <div className="text-center max-w-4xl mx-auto mb-20 animate-in fade-in slide-in-from-top-4 duration-1000">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 mb-6">
            <Brain className="w-4 h-4" />
            <span className="text-[10px] font-black uppercase tracking-widest">Neural Network Intelligence</span>
          </div>
          <h1 className="text-5xl md:text-7xl font-black text-white mb-6 tracking-tight leading-tight">
            The Science Behind <br/>
            <span className="bg-gradient-to-r from-blue-400 via-indigo-400 to-purple-500 bg-clip-text text-transparent">
               AutoAI Predictions
            </span>
          </h1>
          <p className="text-lg text-slate-400 leading-relaxed">
            Harnessing advanced Random Forest Regressors and Polynomial features to decode the complex Pakistani automotive market volatility.
          </p>
        </div>

        {/* Process Flow (Visual Step-by-Step) */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-20">
          <ProcessStep icon={<Database/>} title="Data Intake" desc="Scraping 5k+ live listings" />
          <ProcessStep icon={<Layers/>} title="Preprocessing" desc="K-Means clustering & cleaning" />
          <ProcessStep icon={<Cpu/>} title="ML Analysis" desc="Feature weight calculation" />
          <ProcessStep icon={<TrendingUp/>} title="Valuation" desc="Final market price output" />
        </div>

        {/* Model Performance Grid */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          
          {/* Algorithm Deep Dive */}
          <div className="glass-card p-8 rounded-[2rem] hover:border-blue-500/30 transition-all duration-500">
            <h3 className="text-2xl font-black text-white mb-8 flex items-center gap-3">
              <BarChart3 className="text-blue-500" /> Model Performance
            </h3>
            
            <div className="grid grid-cols-2 gap-4">
               <MetricCard label="Model Type" value={modelInfo?.model_name || 'Random Forest'} />
               <MetricCard label="R² Accuracy" value={modelInfo?.accuracy_r2 ? `${(modelInfo.accuracy_r2 * 100).toFixed(1)}%` : '94.2%'} highlight />
               <MetricCard label="Training Set" value="5,497 Units" />
               <MetricCard label="Mean Error" value={`±${modelInfo?.mae ? modelInfo.mae.toLocaleString() : '185k'}`} />
            </div>

            <div className="mt-8 p-6 bg-blue-500/5 rounded-2xl border border-blue-500/10">
              <p className="text-sm text-slate-400 leading-relaxed italic">
                "Our regressor evaluates non-linear relationships between mileage and registration years to prevent accuracy drops in older vehicles."
              </p>
            </div>
          </div>

          {/* Core Technologies */}
          <div className="glass-card p-8 rounded-[2rem] hover:border-purple-500/30 transition-all duration-500">
            <h3 className="text-2xl font-black text-white mb-8 flex items-center gap-3">
              <Code className="text-purple-500" /> Technology Stack
            </h3>

            <div className="space-y-8">
              <div>
                <h4 className="text-xs font-black uppercase text-slate-500 tracking-widest mb-4 flex items-center gap-2">
                  <Server className="w-3 h-3" /> Backend / Intelligence
                </h4>
                <div className="flex flex-wrap gap-2">
                  {['Python 3.10', 'Flask', 'Scikit-Learn', 'Pandas', 'NumPy', 'Pickle'].map(tech => (
                    <span key={tech} className="tech-chip px-4 py-2 rounded-lg text-xs font-bold text-slate-300">{tech}</span>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="text-xs font-black uppercase text-slate-500 tracking-widest mb-4 flex items-center gap-2">
                  <Globe className="w-3 h-3" /> Frontend / Experience
                </h4>
                <div className="flex flex-wrap gap-2">
                  {['React 18', 'Tailwind CSS', 'Framer Motion', 'Lucide Icons', 'Responsive UI'].map(tech => (
                    <span key={tech} className="tech-chip px-4 py-2 rounded-lg text-xs font-bold text-slate-300" style={{borderColor: 'rgba(139, 92, 246, 0.2)'}}>{tech}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Pricing Segments */}
        <div className="glass-card p-10 rounded-[2.5rem] mb-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-black text-white mb-2 tracking-tight">Market Segments</h2>
            <p className="text-slate-500 text-sm">How our AI classifies the Pakistani automotive tiers.</p>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <SegmentCard icon="💰" label="Economy" range="Under 20L" color="from-green-500 to-emerald-500" />
            <SegmentCard icon="🚗" label="Mid-Range" range="20L - 45L" color="from-blue-500 to-cyan-500" />
            <SegmentCard icon="✨" label="Premium" range="45L - 80L" color="from-purple-500 to-indigo-500" />
            <SegmentCard icon="👑" label="Luxury" range="Above 80L" color="from-pink-500 to-rose-500" />
          </div>
        </div>

        {/* Final CTA / Summary */}
        <div className="relative group overflow-hidden bg-gradient-to-r from-blue-600 to-purple-600 rounded-[2.5rem] p-12 text-center shadow-2xl shadow-blue-900/20 transition-all hover:scale-[1.01]">
          <div className="absolute top-0 right-0 p-8 opacity-10 rotate-12 scale-150 transition-transform group-hover:rotate-45">
            <Rocket className="w-32 h-32 text-white" />
          </div>
          <div className="relative z-10">
            <h3 className="text-3xl font-black text-white mb-4 italic">Experience Unmatched Accuracy</h3>
            <p className="text-white/80 max-w-xl mx-auto mb-8 font-medium">
              Join thousands of Pakistani users using data-driven insights to buy and sell vehicles at the right price.
            </p>
            <div className="flex justify-center gap-8">
               <div className="text-center">
                  <div className="text-3xl font-black text-white tracking-tighter">94%</div>
                  <div className="text-[10px] uppercase font-bold text-white/60">Reliability</div>
               </div>
               <div className="w-px h-12 bg-white/20"></div>
               <div className="text-center">
                  <div className="text-3xl font-black text-white tracking-tighter">&lt; 2s</div>
                  <div className="text-[10px] uppercase font-bold text-white/60">Speed</div>
               </div>
               <div className="w-px h-12 bg-white/20"></div>
               <div className="text-center">
                  <div className="text-3xl font-black text-white tracking-tighter">5k+</div>
                  <div className="text-[10px] uppercase font-bold text-white/60">Data Points</div>
               </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Sub-components for clean JSX
const ProcessStep = ({ icon, title, desc }) => (
  <div className="glass-card p-6 rounded-2xl group hover:-translate-y-2 transition-all">
    <div className="w-10 h-10 rounded-xl bg-blue-500/10 flex items-center justify-center mb-4 text-blue-400 group-hover:bg-blue-500 group-hover:text-white transition-all">
      {icon}
    </div>
    <h4 className="text-white font-bold mb-1">{title}</h4>
    <p className="text-slate-500 text-[10px] uppercase font-bold tracking-wider">{desc}</p>
  </div>
);

const MetricCard = ({ label, value, highlight }) => (
  <div className="bg-slate-800/50 border border-white/5 p-4 rounded-2xl flex flex-col justify-center">
    <div className="text-[10px] font-black uppercase text-slate-500 mb-1 tracking-widest">{label}</div>
    <div className={`text-xl font-black tracking-tight ${highlight ? 'text-blue-400' : 'text-white'}`}>{value}</div>
  </div>
);

const SegmentCard = ({ icon, label, range, color }) => (
  <div className="glass-card p-6 rounded-2xl text-center group hover:-translate-y-2 transition-all border-b-4 border-transparent hover:border-white/10">
    <div className="text-4xl mb-4 group-hover:scale-125 transition-transform inline-block">{icon}</div>
    <h4 className="text-white font-bold mb-1">{label}</h4>
    <p className="text-slate-500 text-xs font-medium mb-4">{range}</p>
    <div className={`h-1.5 w-full rounded-full bg-slate-800 overflow-hidden`}>
      <div className={`h-full bg-gradient-to-r ${color} rounded-full`} style={{width: '100%'}}></div>
    </div>
  </div>
);

export default AboutPage;