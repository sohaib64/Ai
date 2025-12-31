// import React from 'react';

// const AboutPage = ({ modelInfo }) => {
//   return (
//     <div className="container mx-auto px-4 py-8">
//       <div className="max-w-4xl mx-auto">
//         <h1 className="text-3xl font-bold mb-8 text-center">About Our AI Predictor</h1>
        
//         <div className="bg-white rounded-xl shadow-lg p-8 mb-6">
//           <h2 className="text-2xl font-bold mb-4">How It Works</h2>
//           <p className="text-gray-600 mb-4">
//             Our AI-powered car price predictor uses advanced machine learning algorithms trained on thousands 
//             of real Pakistani car listings from PakWheels. The system analyzes multiple factors including brand, 
//             model, mileage, age, engine capacity, and location to provide accurate price estimates.
//           </p>
          
//           <div className="grid md:grid-cols-2 gap-6 mt-6">
//             <div>
//               <h3 className="font-bold text-lg mb-3">Model Details</h3>
//               <ul className="space-y-2 text-gray-600">
//                 <li className="flex items-start">
//                   <span className="mr-2">•</span>
//                   <span><strong>Algorithm:</strong> {modelInfo?.model_name || 'Random Forest Regressor'}</span>
//                 </li>
//                 <li className="flex items-start">
//                   <span className="mr-2">•</span>
//                   <span><strong>Accuracy (R²):</strong> {modelInfo?.accuracy_r2 ? `${(modelInfo.accuracy_r2 * 100).toFixed(1)}%` : '94.2%'}</span>
//                 </li>
//                 <li className="flex items-start">
//                   <span className="mr-2">•</span>
//                   <span><strong>Dataset:</strong> 5,497 verified car listings</span>
//                 </li>
//                 <li className="flex items-start">
//                   <span className="mr-2">•</span>
//                   <span><strong>Features:</strong> Polynomial transformation + K-Means clustering</span>
//                 </li>
//                 <li className="flex items-start">
//                   <span className="mr-2">•</span>
//                   <span><strong>MAE:</strong> ±{modelInfo?.mae ? modelInfo.mae.toLocaleString() : '185,000'} PKR</span>
//                 </li>
//               </ul>
//             </div>
//             <div>
//               <h3 className="font-bold text-lg mb-3">Key Features</h3>
//               <ul className="space-y-2 text-gray-600">
//                 <li className="flex items-start">
//                   <span className="mr-2">✓</span>
//                   <span>Instant predictions (&lt;2 seconds)</span>
//                 </li>
//                 <li className="flex items-start">
//                   <span className="mr-2">✓</span>
//                   <span>Smart chatbot interface (Urdu/English)</span>
//                 </li>
//                 <li className="flex items-start">
//                   <span className="mr-2">✓</span>
//                   <span>Market analytics dashboard</span>
//                 </li>
//                 <li className="flex items-start">
//                   <span className="mr-2">✓</span>
//                   <span>Price range estimates (±10%)</span>
//                 </li>
//                 <li className="flex items-start">
//                   <span className="mr-2">✓</span>
//                   <span>Segment classification (Economy to Luxury)</span>
//                 </li>
//                 <li className="flex items-start">
//                   <span className="mr-2">✓</span>
//                   <span>Validation warnings for anomalies</span>
//                 </li>
//               </ul>
//             </div>
//           </div>
//         </div>

//         <div className="bg-white rounded-xl shadow-lg p-8 mb-6">
//           <h2 className="text-2xl font-bold mb-4">Technology Stack</h2>
//           <div className="grid md:grid-cols-2 gap-6">
//             <div>
//               <h3 className="font-bold text-lg mb-3">Backend</h3>
//               <ul className="space-y-1 text-gray-600">
//                 <li>• Python Flask REST API</li>
//                 <li>• Scikit-learn ML models</li>
//                 <li>• Pandas for data processing</li>
//                 <li>• NumPy for numerical operations</li>
//               </ul>
//             </div>
//             <div>
//               <h3 className="font-bold text-lg mb-3">Frontend</h3>
//               <ul className="space-y-1 text-gray-600">
//                 <li>• React.js with Hooks</li>
//                 <li>• Tailwind CSS styling</li>
//                 <li>• Lucide React icons</li>
//                 <li>• Responsive design</li>
//               </ul>
//             </div>
//           </div>
//         </div>

//         <div className="bg-white rounded-xl shadow-lg p-8">
//           <h2 className="text-2xl font-bold mb-4">Market Segments</h2>
//           <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
//             <div className="border-l-4 border-green-500 pl-4 py-2">
//               <div className="font-bold text-green-700">Economy</div>
//               <div className="text-sm text-gray-600">Under 20 Lacs PKR</div>
//             </div>
//             <div className="border-l-4 border-blue-500 pl-4 py-2">
//               <div className="font-bold text-blue-700">Mid-Range</div>
//               <div className="text-sm text-gray-600">20 - 40 Lacs PKR</div>
//             </div>
//             <div className="border-l-4 border-purple-500 pl-4 py-2">
//               <div className="font-bold text-purple-700">Premium</div>
//               <div className="text-sm text-gray-600">40 - 70 Lacs PKR</div>
//             </div>
//             <div className="border-l-4 border-red-500 pl-4 py-2">
//               <div className="font-bold text-red-700">Luxury</div>
//               <div className="text-sm text-gray-600">Above 70 Lacs PKR</div>
//             </div>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default AboutPage;


import React from 'react';
import { Brain, Zap, TrendingUp, Database, Cpu, Code, Layers, Shield, Sparkles, ChevronRight, Activity, BarChart3, Target, CheckCircle2, Rocket, Award } from 'lucide-react';

const AboutPage = ({ modelInfo }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-20 left-10 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
        <div className="absolute top-40 right-10 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse delay-700"></div>
        <div className="absolute -bottom-8 left-1/2 w-96 h-96 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse delay-1000"></div>
      </div>

      {/* Floating Particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(30)].map((_, i) => (
          <div
            key={i}
            className="absolute w-2 h-2 bg-blue-400 rounded-full opacity-30"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animation: `float ${5 + Math.random() * 10}s infinite ease-in-out`,
              animationDelay: `${Math.random() * 5}s`
            }}
          ></div>
        ))}
      </div>

      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px) translateX(0px); }
          33% { transform: translateY(-30px) translateX(20px); }
          66% { transform: translateY(20px) translateX(-20px); }
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(30px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideInLeft {
          from { opacity: 0; transform: translateX(-50px); }
          to { opacity: 1; transform: translateX(0); }
        }
        @keyframes slideInRight {
          from { opacity: 0; transform: translateX(50px); }
          to { opacity: 1; transform: translateX(0); }
        }
        @keyframes scaleIn {
          from { opacity: 0; transform: scale(0.9); }
          to { opacity: 1; transform: scale(1); }
        }
        @keyframes pulse-glow {
          0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.3); }
          50% { box-shadow: 0 0 40px rgba(99, 102, 241, 0.6); }
        }
        .animate-slide-up {
          animation: slideUp 0.6s ease-out forwards;
        }
        .animate-slide-in-left {
          animation: slideInLeft 0.7s ease-out forwards;
        }
        .animate-slide-in-right {
          animation: slideInRight 0.7s ease-out forwards;
        }
        .animate-scale-in {
          animation: scaleIn 0.6s ease-out forwards;
        }
        .glass-effect {
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .hover-lift:hover {
          transform: translateY(-8px);
          box-shadow: 0 20px 40px rgba(99, 102, 241, 0.4);
        }
        .delay-100 { animation-delay: 0.1s; }
        .delay-200 { animation-delay: 0.2s; }
        .delay-300 { animation-delay: 0.3s; }
        .delay-400 { animation-delay: 0.4s; }
        .delay-500 { animation-delay: 0.5s; }
        .delay-600 { animation-delay: 0.6s; }
        .delay-700 { animation-delay: 0.7s; }
        .delay-1000 { animation-delay: 1s; }
      `}</style>

      <div className="container mx-auto px-4 py-12 relative z-10">
        {/* Header */}
        <div className="text-center mb-16 animate-slide-up mt-16">
          <div className="inline-flex items-center gap-2 glass-effect text-purple-300 px-6 py-3 rounded-full mb-6">
            <Brain className="w-5 h-5 animate-pulse" />
            <span className="font-semibold text-sm tracking-wide">AI-POWERED INTELLIGENCE</span>
          </div>
          
          <h1 className="text-5xl md:text-6xl font-extrabold text-white mb-4">
            About Our <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">AI Predictor</span>
          </h1>
          <p className="text-gray-300 text-lg max-w-3xl mx-auto">
            Revolutionary machine learning technology trained on thousands of Pakistani car listings to deliver accurate price predictions in seconds
          </p>
        </div>

        {/* How It Works Section */}
        <div className="glass-effect rounded-3xl p-8 md:p-12 mb-8 hover:shadow-2xl hover:shadow-blue-500/20 transition-all duration-300 animate-scale-in">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-3 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-xl">
              <Rocket className="w-7 h-7 text-blue-400" />
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-white">How It Works</h2>
          </div>
          
          <div className="grid md:grid-cols-3 gap-6 mb-8">
            <div className="glass-effect rounded-2xl p-6 hover-lift transition-all duration-300 group">
              <div className="w-14 h-14 rounded-full bg-gradient-to-br from-blue-500/20 to-blue-600/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <Database className="w-7 h-7 text-blue-400" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Data Collection</h3>
              <p className="text-gray-400 leading-relaxed">
                Trained on 5,497+ verified car listings from PakWheels, ensuring real market data
              </p>
            </div>

            <div className="glass-effect rounded-2xl p-6 hover-lift transition-all duration-300 group">
              <div className="w-14 h-14 rounded-full bg-gradient-to-br from-purple-500/20 to-purple-600/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <Cpu className="w-7 h-7 text-purple-400" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">AI Processing</h3>
              <p className="text-gray-400 leading-relaxed">
                Advanced Random Forest algorithm analyzes brand, model, mileage, age, and location
              </p>
            </div>

            <div className="glass-effect rounded-2xl p-6 hover-lift transition-all duration-300 group">
              <div className="w-14 h-14 rounded-full bg-gradient-to-br from-green-500/20 to-green-600/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <Target className="w-7 h-7 text-green-400" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Instant Results</h3>
              <p className="text-gray-400 leading-relaxed">
                Get accurate price predictions in under 2 seconds with ±10% confidence range
              </p>
            </div>
          </div>

          <p className="text-gray-300 text-lg leading-relaxed">
            Our AI-powered car price predictor uses advanced machine learning algorithms trained on thousands 
            of real Pakistani car listings from PakWheels. The system analyzes multiple factors including brand, 
            model, mileage, age, engine capacity, and location to provide accurate price estimates.
          </p>
        </div>

        {/* Model Details & Features Grid */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Model Details */}
          <div className="glass-effect rounded-3xl p-8 hover:shadow-2xl hover:shadow-purple-500/20 transition-all duration-300 animate-slide-in-left">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-gradient-to-br from-purple-500/20 to-purple-600/20 rounded-xl">
                <BarChart3 className="w-6 h-6 text-purple-400" />
              </div>
              <h3 className="text-2xl font-bold text-white">Model Details</h3>
            </div>
            
            <div className="space-y-4">
              <div className="glass-effect rounded-xl p-4 hover:bg-white/10 transition-all group">
                <div className="flex items-start gap-3">
                  <CheckCircle2 className="w-5 h-5 text-blue-400 mt-1 flex-shrink-0" />
                  <div>
                    <div className="text-sm text-gray-400 mb-1">Algorithm</div>
                    <div className="text-white font-semibold">{modelInfo?.model_name || 'Random Forest Regressor'}</div>
                  </div>
                </div>
              </div>

              <div className="glass-effect rounded-xl p-4 hover:bg-white/10 transition-all group">
                <div className="flex items-start gap-3">
                  <CheckCircle2 className="w-5 h-5 text-green-400 mt-1 flex-shrink-0" />
                  <div>
                    <div className="text-sm text-gray-400 mb-1">Accuracy (R²)</div>
                    <div className="text-white font-semibold text-2xl">
                      {modelInfo?.accuracy_r2 ? `${(modelInfo.accuracy_r2 * 100).toFixed(1)}%` : '94.2%'}
                    </div>
                  </div>
                </div>
              </div>

              <div className="glass-effect rounded-xl p-4 hover:bg-white/10 transition-all group">
                <div className="flex items-start gap-3">
                  <CheckCircle2 className="w-5 h-5 text-purple-400 mt-1 flex-shrink-0" />
                  <div>
                    <div className="text-sm text-gray-400 mb-1">Dataset Size</div>
                    <div className="text-white font-semibold">5,497 verified car listings</div>
                  </div>
                </div>
              </div>

              <div className="glass-effect rounded-xl p-4 hover:bg-white/10 transition-all group">
                <div className="flex items-start gap-3">
                  <CheckCircle2 className="w-5 h-5 text-orange-400 mt-1 flex-shrink-0" />
                  <div>
                    <div className="text-sm text-gray-400 mb-1">Features</div>
                    <div className="text-white font-semibold">Polynomial transformation + K-Means clustering</div>
                  </div>
                </div>
              </div>

              <div className="glass-effect rounded-xl p-4 hover:bg-white/10 transition-all group">
                <div className="flex items-start gap-3">
                  <CheckCircle2 className="w-5 h-5 text-pink-400 mt-1 flex-shrink-0" />
                  <div>
                    <div className="text-sm text-gray-400 mb-1">Mean Absolute Error</div>
                    <div className="text-white font-semibold">
                      ±{modelInfo?.mae ? modelInfo.mae.toLocaleString() : '185,000'} PKR
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Key Features */}
          <div className="glass-effect rounded-3xl p-8 hover:shadow-2xl hover:shadow-blue-500/20 transition-all duration-300 animate-slide-in-right">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-xl">
                <Sparkles className="w-6 h-6 text-blue-400" />
              </div>
              <h3 className="text-2xl font-bold text-white">Key Features</h3>
            </div>
            
            <div className="space-y-3">
              {[
                { icon: Zap, text: 'Instant predictions (<2 seconds)', color: 'text-yellow-400' },
                { icon: Brain, text: 'Smart chatbot interface (Urdu/English)', color: 'text-blue-400' },
                { icon: Activity, text: 'Market analytics dashboard', color: 'text-green-400' },
                { icon: TrendingUp, text: 'Price range estimates (±10%)', color: 'text-purple-400' },
                { icon: Award, text: 'Segment classification (Economy to Luxury)', color: 'text-orange-400' },
                { icon: Shield, text: 'Validation warnings for anomalies', color: 'text-red-400' }
              ].map((feature, idx) => (
                <div 
                  key={idx} 
                  className="glass-effect rounded-xl p-4 hover:bg-white/10 transition-all group flex items-center gap-3"
                  style={{ animationDelay: `${idx * 0.1}s` }}
                >
                  <div className="p-2 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-lg group-hover:scale-110 transition-transform">
                    <feature.icon className={`w-5 h-5 ${feature.color}`} />
                  </div>
                  <span className="text-white font-medium">{feature.text}</span>
                  <ChevronRight className="w-4 h-4 text-gray-500 ml-auto opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Technology Stack */}
        <div className="glass-effect rounded-3xl p-8 mb-8 hover:shadow-2xl hover:shadow-cyan-500/20 transition-all duration-300 animate-scale-in delay-200">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-3 bg-gradient-to-br from-cyan-500/20 to-cyan-600/20 rounded-xl">
              <Code className="w-7 h-7 text-cyan-400" />
            </div>
            <h2 className="text-3xl font-bold text-white">Technology Stack</h2>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8">
            {/* Backend */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Layers className="w-5 h-5 text-purple-400" />
                <h3 className="text-xl font-bold text-white">Backend</h3>
              </div>
              <div className="space-y-3">
                {[
                  { name: 'Python Flask', desc: 'REST API framework' },
                  { name: 'Scikit-learn', desc: 'ML models & algorithms' },
                  { name: 'Pandas', desc: 'Data processing' },
                  { name: 'NumPy', desc: 'Numerical operations' }
                ].map((tech, idx) => (
                  <div key={idx} className="glass-effect rounded-xl p-4 hover:bg-white/10 transition-all group">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-white font-semibold mb-1">{tech.name}</div>
                        <div className="text-sm text-gray-400">{tech.desc}</div>
                      </div>
                      <div className="w-2 h-2 rounded-full bg-purple-400 animate-pulse"></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Frontend */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Sparkles className="w-5 h-5 text-blue-400" />
                <h3 className="text-xl font-bold text-white">Frontend</h3>
              </div>
              <div className="space-y-3">
                {[
                  { name: 'React.js', desc: 'with Hooks & components' },
                  { name: 'Tailwind CSS', desc: 'Modern styling framework' },
                  { name: 'Lucide React', desc: 'Beautiful icon library' },
                  { name: 'Responsive Design', desc: 'Mobile-first approach' }
                ].map((tech, idx) => (
                  <div key={idx} className="glass-effect rounded-xl p-4 hover:bg-white/10 transition-all group">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-white font-semibold mb-1">{tech.name}</div>
                        <div className="text-sm text-gray-400">{tech.desc}</div>
                      </div>
                      <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Market Segments */}
        <div className="glass-effect rounded-3xl p-8 hover:shadow-2xl hover:shadow-pink-500/20 transition-all duration-300 animate-scale-in delay-400">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-3 bg-gradient-to-br from-pink-500/20 to-pink-600/20 rounded-xl">
              <Award className="w-7 h-7 text-pink-400" />
            </div>
            <h2 className="text-3xl font-bold text-white">Market Segments</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { name: 'Economy', range: 'Under 20 Lacs PKR', color: 'from-green-500 to-emerald-500', bgColor: 'from-green-500/20 to-green-600/20', textColor: 'text-green-400', icon: '💰' },
              { name: 'Mid-Range', range: '20 - 40 Lacs PKR', color: 'from-blue-500 to-cyan-500', bgColor: 'from-blue-500/20 to-blue-600/20', textColor: 'text-blue-400', icon: '🚗' },
              { name: 'Premium', range: '40 - 70 Lacs PKR', color: 'from-purple-500 to-pink-500', bgColor: 'from-purple-500/20 to-purple-600/20', textColor: 'text-purple-400', icon: '✨' },
              { name: 'Luxury', range: 'Above 70 Lacs PKR', color: 'from-red-500 to-orange-500', bgColor: 'from-red-500/20 to-red-600/20', textColor: 'text-red-400', icon: '👑' }
            ].map((segment, idx) => (
              <div 
                key={idx} 
                className="glass-effect rounded-2xl p-6 hover-lift transition-all duration-300 group text-center"
                style={{ animationDelay: `${idx * 0.15}s` }}
              >
                <div className={`w-16 h-16 rounded-full bg-gradient-to-br ${segment.bgColor} flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform text-3xl`}>
                  {segment.icon}
                </div>
                <h3 className={`text-xl font-bold ${segment.textColor} mb-2`}>{segment.name}</h3>
                <div className="text-gray-400 text-sm">{segment.range}</div>
                <div className="mt-4 h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div 
                    className={`h-full bg-gradient-to-r ${segment.color} rounded-full transition-all duration-1000`}
                    style={{ width: '100%', animationDelay: `${idx * 0.2}s` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom CTA */}
        <div className="mt-12 glass-effect rounded-3xl p-8 md:p-12 text-center border-2 border-blue-500/30 animate-scale-in delay-600 hover:shadow-2xl hover:shadow-blue-500/30 transition-all duration-300">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Brain className="w-8 h-8 text-blue-400 animate-pulse" />
            <h3 className="text-3xl md:text-4xl font-bold text-white">AI-Powered Precision</h3>
          </div>
          <p className="text-gray-300 text-lg mb-6 max-w-2xl mx-auto">
            Experience the future of car pricing with our state-of-the-art machine learning model, 
            delivering unmatched accuracy and reliability for the Pakistani automotive market.
          </p>
          <div className="flex flex-wrap justify-center gap-6">
            <div className="glass-effect px-8 py-4 rounded-2xl hover:bg-white/10 transition-all group">
              <div className="text-sm text-gray-400 mb-1">Trained on</div>
              <div className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                5,497+
              </div>
              <div className="text-xs text-gray-500 mt-1">data points</div>
            </div>
            <div className="glass-effect px-8 py-4 rounded-2xl hover:bg-white/10 transition-all group">
              <div className="text-sm text-gray-400 mb-1">Accuracy Rate</div>
              <div className="text-3xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
                94.2%
              </div>
              <div className="text-xs text-gray-500 mt-1">R² score</div>
            </div>
            <div className="glass-effect px-8 py-4 rounded-2xl hover:bg-white/10 transition-all group">
              <div className="text-sm text-gray-400 mb-1">Response Time</div>
              <div className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                &lt;2s
              </div>
              <div className="text-xs text-gray-500 mt-1">instant results</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutPage;