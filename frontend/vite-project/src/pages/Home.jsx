// import { Link } from 'react-router-dom';
// import { motion } from 'framer-motion';
// import { Car, TrendingUp, Brain, Zap, CheckCircle, ArrowRight } from 'lucide-react';
// import { useEffect, useState } from 'react';
// import { getModelInfo } from '../services/api';

// const Home = () => {
//   const [modelInfo, setModelInfo] = useState(null);

//   useEffect(() => {
//     getModelInfo().then(setModelInfo).catch(console.error);
//   }, []);

//   const features = [
//     {
//       icon: Brain,
//       title: 'AI-Powered',
//       description: 'Random Forest algorithm with 94.65% accuracy',
//       color: 'from-blue-500 to-cyan-500',
//     },
//     {
//       icon: TrendingUp,
//       title: 'Market Analysis',
//       description: 'K-Means clustering for market segmentation',
//       color: 'from-purple-500 to-pink-500',
//     },
//     {
//       icon: Zap,
//       title: 'Instant Results',
//       description: 'Get price predictions in seconds',
//       color: 'from-orange-500 to-red-500',
//     },
//   ];

//   const stats = [
//     { label: 'Accuracy', value: modelInfo ? `${(modelInfo.accuracy * 100).toFixed(1)}%` : '94.7%' },
//     { label: 'Avg Error', value: modelInfo ? `±${(modelInfo.avg_error / 1000).toFixed(0)}k PKR` : '±186k' },
//     { label: 'Model', value: 'Random Forest' },
//   ];

//   return (
//     <div className="min-h-screen">
//       {/* Hero Section */}
//       <section className="relative overflow-hidden py-20 px-4">
//         <div className="max-w-7xl mx-auto">
//           <div className="grid lg:grid-cols-2 gap-12 items-center">
//             <motion.div
//               initial={{ opacity: 0, x: -50 }}
//               animate={{ opacity: 1, x: 0 }}
//               transition={{ duration: 0.6 }}
//             >
//               <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
//                 AI-Powered
//                 <span className="gradient-text block">Car Price Predictor</span>
//               </h1>

//               <p className="text-xl text-gray-600 mb-8">
//                 Get instant, accurate car price predictions using advanced machine learning. 
//                 Trained on 5,000+ Pakistani car listings.
//               </p>

//               <div className="flex flex-wrap gap-4">
//                 <Link
//                   to="/predict"
//                   className="btn-primary inline-flex items-center space-x-2"
//                 >
//                   <span>Start Prediction</span>
//                   <ArrowRight className="w-5 h-5" />
//                 </Link>

//                 <Link
//                   to="/analytics"
//                   className="btn-secondary inline-flex items-center space-x-2"
//                 >
//                   <span>View Analytics</span>
//                   <TrendingUp className="w-5 h-5" />
//                 </Link>
//               </div>

//               {/* Stats */}
//               <div className="grid grid-cols-3 gap-4 mt-12">
//                 {stats.map((stat, idx) => (
//                   <motion.div
//                     key={stat.label}
//                     initial={{ opacity: 0, y: 20 }}
//                     animate={{ opacity: 1, y: 0 }}
//                     transition={{ delay: idx * 0.1 }}
//                     className="text-center"
//                   >
//                     <div className="text-2xl font-bold text-blue-600">{stat.value}</div>
//                     <div className="text-sm text-gray-600">{stat.label}</div>
//                   </motion.div>
//                 ))}
//               </div>
//             </motion.div>

//             <motion.div
//               initial={{ opacity: 0, x: 50 }}
//               animate={{ opacity: 1, x: 0 }}
//               transition={{ duration: 0.6 }}
//               className="relative"
//             >
//               <div className="bg-gradient-to-br from-blue-600 to-purple-600 rounded-3xl p-8 shadow-2xl">
//                 <Car className="w-64 h-64 mx-auto text-white opacity-20" />

//                 <div className="absolute inset-0 flex items-center justify-center">
//                   <div className="bg-white rounded-2xl p-8 shadow-xl max-w-sm">
//                     <h3 className="text-xl font-bold text-gray-900 mb-4">
//                       Quick Preview
//                     </h3>
//                     <div className="space-y-3">
//                       <div className="flex items-center justify-between">
//                         <span className="text-gray-600">Brand</span>
//                         <span className="font-semibold">Honda Civic</span>
//                       </div>
//                       <div className="flex items-center justify-between">
//                         <span className="text-gray-600">Year</span>
//                         <span className="font-semibold">2017</span>
//                       </div>
//                       <div className="flex items-center justify-between">
//                         <span className="text-gray-600">Mileage</span>
//                         <span className="font-semibold">60,000 km</span>
//                       </div>
//                       <div className="border-t pt-3">
//                         <div className="flex items-center justify-between">
//                           <span className="text-gray-600">Predicted Price</span>
//                           <span className="text-2xl font-bold text-blue-600">36.4L</span>
//                         </div>
//                       </div>
//                     </div>
//                   </div>
//                 </div>
//               </div>
//             </motion.div>
//           </div>
//         </div>
//       </section>

//       {/* Features Section */}
//       <section className="py-20 px-4 bg-white">
//         <div className="max-w-7xl mx-auto">
//           <motion.div
//             initial={{ opacity: 0, y: 20 }}
//             whileInView={{ opacity: 1, y: 0 }}
//             viewport={{ once: true }}
//             className="text-center mb-16"
//           >
//             <h2 className="text-4xl font-bold text-gray-900 mb-4">
//               Why Choose Our Predictor?
//             </h2>
//             <p className="text-xl text-gray-600">
//               Powered by cutting-edge machine learning technology
//             </p>
//           </motion.div>

//           <div className="grid md:grid-cols-3 gap-8">
//             {features.map((feature, idx) => {
//               const Icon = feature.icon;
//               return (
//                 <motion.div
//                   key={feature.title}
//                   initial={{ opacity: 0, y: 20 }}
//                   whileInView={{ opacity: 1, y: 0 }}
//                   viewport={{ once: true }}
//                   transition={{ delay: idx * 0.1 }}
//                   className="card-hover bg-white rounded-2xl p-8 shadow-lg border border-gray-100"
//                 >
//                   <div className={`bg-gradient-to-r ${feature.color} w-16 h-16 rounded-2xl flex items-center justify-center mb-6`}>
//                     <Icon className="w-8 h-8 text-white" />
//                   </div>
//                   <h3 className="text-2xl font-bold text-gray-900 mb-3">
//                     {feature.title}
//                   </h3>
//                   <p className="text-gray-600">
//                     {feature.description}
//                   </p>
//                 </motion.div>
//               );
//             })}
//           </div>
//         </div>
//       </section>

//       {/* How It Works */}
//       <section className="py-20 px-4">
//         <div className="max-w-7xl mx-auto">
//           <motion.div
//             initial={{ opacity: 0, y: 20 }}
//             whileInView={{ opacity: 1, y: 0 }}
//             viewport={{ once: true }}
//             className="text-center mb-16"
//           >
//             <h2 className="text-4xl font-bold text-gray-900 mb-4">
//               How It Works
//             </h2>
//             <p className="text-xl text-gray-600">
//               Simple 3-step process to get your prediction
//             </p>
//           </motion.div>

//           <div className="grid md:grid-cols-3 gap-8">
//             {[
//               { step: '1', title: 'Enter Details', desc: 'Provide car information like brand, model, year, and mileage' },
//               { step: '2', title: 'AI Analysis', desc: 'Our Random Forest model analyzes 60+ features instantly' },
//               { step: '3', title: 'Get Results', desc: 'Receive accurate price prediction with confidence score' },
//             ].map((item, idx) => (
//               <motion.div
//                 key={item.step}
//                 initial={{ opacity: 0, scale: 0.9 }}
//                 whileInView={{ opacity: 1, scale: 1 }}
//                 viewport={{ once: true }}
//                 transition={{ delay: idx * 0.1 }}
//                 className="text-center"
//               >
//                 <div className="w-16 h-16 bg-blue-600 text-white rounded-full flex items-center justify-center text-2xl font-bold mx-auto mb-4">
//                   {item.step}
//                 </div>
//                 <h3 className="text-xl font-bold text-gray-900 mb-2">{item.title}</h3>
//                 <p className="text-gray-600">{item.desc}</p>
//               </motion.div>
//             ))}
//           </div>
//         </div>
//       </section>

//       {/* CTA Section */}
//       <section className="py-20 px-4 bg-gradient-to-r from-blue-600 to-purple-600">
//         <motion.div
//           initial={{ opacity: 0, y: 20 }}
//           whileInView={{ opacity: 1, y: 0 }}
//           viewport={{ once: true }}
//           className="max-w-4xl mx-auto text-center"
//         >
//           <h2 className="text-4xl font-bold text-white mb-6">
//             Ready to Predict Your Car's Price?
//           </h2>
//           <p className="text-xl text-blue-100 mb-8">
//             Join thousands of users who trust our AI-powered predictions
//           </p>
//           <Link
//             to="/predict"
//             className="inline-flex items-center space-x-2 bg-white text-blue-600 px-8 py-4 rounded-xl font-bold text-lg hover:shadow-2xl transform hover:-translate-y-1 transition-all"
//           >
//             <span>Get Started Now</span>
//             <ArrowRight className="w-5 h-5" />
//           </Link>
//         </motion.div>
//       </section>
//     </div>
//   );
// };

// export default Home;


import React from 'react';
import { Sparkles, Calculator, Bot, TrendingUp, BarChart3, Zap, Shield, Clock } from 'lucide-react';

const HomePage = ({ modelInfo, setCurrentPage, openChat }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-20 left-10 w-72 h-72 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
        <div className="absolute top-40 right-10 w-72 h-72 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse delay-700"></div>
        <div className="absolute -bottom-8 left-1/2 w-72 h-72 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse delay-1000"></div>
      </div>

      {/* Floating Particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(20)].map((_, i) => (
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
        @keyframes scaleIn {
          from { opacity: 0; transform: scale(0.9); }
          to { opacity: 1; transform: scale(1); }
        }
        .animate-slide-up {
          animation: slideUp 0.8s ease-out forwards;
        }
        .animate-scale-in {
          animation: scaleIn 0.6s ease-out forwards;
        }
        .delay-100 { animation-delay: 0.1s; }
        .delay-200 { animation-delay: 0.2s; }
        .delay-300 { animation-delay: 0.3s; }
        .delay-700 { animation-delay: 0.7s; }
        .delay-1000 { animation-delay: 1s; }
        .glass-effect {
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .hover-glow:hover {
          box-shadow: 0 0 30px rgba(99, 102, 241, 0.5);
        }
      `}</style>

      {/* Hero Section */}
      <div className="container mx-auto px-4 py-20 relative z-10">
        <div className="text-center max-w-5xl mx-auto">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 glass-effect text-blue-300 px-6 py-3 mt-8 rounded-full mb-8 animate-slide-up border border-blue-400/30">
            <Sparkles className="w-5 h-5 animate-pulse" />
            <span className="font-semibold text-sm tracking-wide">NEXT-GEN AI PREDICTIONS</span>
          </div>


          {/* Main Heading */}
          <h1 className="text-6xl md:text-7xl font-extrabold text-white mb-6 animate-slide-up delay-100">
            Discover Your Car's
            <br />
            <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent animate-pulse">
              Real Market Value
            </span>
          </h1>

          {/* Subheading */}
          <p className="text-xl md:text-2xl text-gray-300 mb-12 max-w-3xl mx-auto animate-slide-up delay-200 leading-relaxed">
            Harness cutting-edge machine learning trained on <span className="text-blue-400 font-semibold">5,497+ vehicles</span> to get instant, precise valuations for Pakistani cars
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-wrap gap-6 justify-center mb-16 animate-slide-up delay-300">
            <button
              onClick={() => setCurrentPage('predict')}
              className="group relative flex items-center gap-3 px-10 py-5 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 text-white rounded-2xl font-bold text-lg hover:shadow-2xl hover:shadow-purple-500/50 transition-all duration-300 transform hover:scale-105 hover-glow overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <Calculator className="w-6 h-6 relative z-10" />
              <span className="relative z-10">Get Instant Prediction</span>
              <Zap className="w-5 h-5 relative z-10 animate-pulse" />
            </button>
            <button
              onClick={openChat}
              className="flex items-center gap-3 px-10 py-5 glass-effect text-white border-2 border-purple-400/50 rounded-2xl font-bold text-lg hover:border-purple-400 hover:shadow-xl hover:shadow-purple-500/30 transition-all duration-300 transform hover:scale-105"
            >
              <Bot className="w-6 h-6" />
              <span>Chat with AI Assistant</span>
            </button>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto animate-scale-in delay-300">
            <div className="glass-effect p-8 rounded-2xl hover:shadow-2xl hover:shadow-blue-500/20 transition-all duration-300 transform hover:scale-105 hover:-translate-y-2 group">
              <div className="text-5xl font-extrabold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent mb-3 group-hover:scale-110 transition-transform">
                {modelInfo?.accuracy_r2 ? `${(modelInfo.accuracy_r2 * 100).toFixed(1)}%` : '94.2%'}
              </div>
              <div className="text-gray-300 font-semibold text-lg">Model Accuracy</div>
              <div className="w-full h-2 bg-gray-700 rounded-full mt-4 overflow-hidden">
                <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full" style={{ width: '94.2%' }}></div>
              </div>
            </div>
            <div className="glass-effect p-8 rounded-2xl hover:shadow-2xl hover:shadow-purple-500/20 transition-all duration-300 transform hover:scale-105 hover:-translate-y-2 group">
              <div className="text-5xl font-extrabold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent mb-3 group-hover:scale-110 transition-transform">
                5,497+
              </div>
              <div className="text-gray-300 font-semibold text-lg">Training Dataset</div>
              <div className="flex items-center gap-1 mt-4">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="flex-1 h-2 bg-purple-500 rounded-full"></div>
                ))}
              </div>
            </div>
            <div className="glass-effect p-8 rounded-2xl hover:shadow-2xl hover:shadow-green-500/20 transition-all duration-300 transform hover:scale-105 hover:-translate-y-2 group">
              <div className="text-5xl font-extrabold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent mb-3 group-hover:scale-110 transition-transform">
                &lt;2s
              </div>
              <div className="text-gray-300 font-semibold text-lg">Lightning Fast</div>
              <Clock className="w-8 h-8 text-green-400 mx-auto mt-3 animate-pulse" />
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="relative z-10 py-20 mt-10">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Why We're <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Different</span>
            </h2>
            <p className="text-gray-400 text-lg">Powered by advanced AI technology built for accuracy</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            <div className="glass-effect p-8 rounded-2xl text-center hover:shadow-2xl hover:shadow-blue-500/30 transition-all duration-300 transform hover:scale-105 group">
              <div className="inline-flex p-5 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-2xl mb-6 group-hover:scale-110 transition-transform">
                <TrendingUp className="w-10 h-10 text-blue-400" />
              </div>
              <h3 className="text-2xl font-bold text-white mb-3">Precision Analytics</h3>
              <p className="text-gray-400 leading-relaxed">Advanced Random Forest algorithm with polynomial feature engineering for unmatched accuracy</p>
              <div className="mt-4 flex justify-center gap-2">
                <Shield className="w-5 h-5 text-blue-400" />
                <span className="text-blue-400 font-semibold">Verified Results</span>
              </div>
            </div>

            <div className="glass-effect p-8 rounded-2xl text-center hover:shadow-2xl hover:shadow-purple-500/30 transition-all duration-300 transform hover:scale-105 group">
              <div className="inline-flex p-5 bg-gradient-to-br from-purple-500/20 to-purple-600/20 rounded-2xl mb-6 group-hover:scale-110 transition-transform">
                <Bot className="w-10 h-10 text-purple-400" />
              </div>
              <h3 className="text-2xl font-bold text-white mb-3">AI Chatbot</h3>
              <p className="text-gray-400 leading-relaxed">Conversational interface for effortless predictions - just chat naturally about your vehicle</p>
              <div className="mt-4 flex justify-center gap-2">
                <Sparkles className="w-5 h-5 text-purple-400 animate-pulse" />
                <span className="text-purple-400 font-semibold">Smart & Intuitive</span>
              </div>
            </div>

            <div className="glass-effect p-8 rounded-2xl text-center hover:shadow-2xl hover:shadow-green-500/30 transition-all duration-300 transform hover:scale-105 group">
              <div className="inline-flex p-5 bg-gradient-to-br from-green-500/20 to-green-600/20 rounded-2xl mb-6 group-hover:scale-110 transition-transform">
                <BarChart3 className="w-10 h-10 text-green-400" />
              </div>
              <h3 className="text-2xl font-bold text-white mb-3">Market Intelligence</h3>
              <p className="text-gray-400 leading-relaxed">Real-time insights from comprehensive Pakistani automotive market data and trends</p>
              <div className="mt-4 flex justify-center gap-2">
                <TrendingUp className="w-5 h-5 text-green-400" />
                <span className="text-green-400 font-semibold">Live Data</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom CTA */}
      <div className="relative z-10 pb-20">
        <div className="container mx-auto px-4">
          <div className="glass-effect max-w-4xl mx-auto p-12 rounded-3xl text-center border border-purple-400/30">
            <h3 className="text-3xl font-bold text-white mb-4">Ready to discover your car's value?</h3>
            <p className="text-gray-300 text-lg mb-8">Join thousands who trust our AI-powered predictions</p>
            <button
              onClick={() => setCurrentPage('predict')}
              className="inline-flex items-center gap-3 px-12 py-5 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-2xl font-bold text-lg hover:shadow-2xl hover:shadow-purple-500/50 transition-all duration-300 transform hover:scale-105"
            >
              <Calculator className="w-6 h-6" />
              Start Your Free Prediction
              <Sparkles className="w-5 h-5 animate-pulse" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;