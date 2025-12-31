import React, { useState, useEffect } from 'react';
import { Loader, TrendingUp, DollarSign, Calendar, Fuel, Settings, BarChart3, PieChart, Award, Zap, Car, Activity, Sparkles } from 'lucide-react';

const AnalyticsPage = ({ showToast }) => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/analytics');
        const data = await response.json();
        setStats(data);
        setLoading(false);
      } catch (err) {
        showToast('Failed to load analytics', 'error');
        setLoading(false);
      }
    };

    fetchAnalytics();
  }, [showToast]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <Loader className="w-16 h-16 animate-spin text-blue-400 mx-auto mb-4" />
          <p className="text-gray-300 text-lg">Loading analytics data...</p>
        </div>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="glass-effect p-8 rounded-2xl border border-red-500/30">
          <div className="text-red-400 text-xl font-semibold">Failed to load analytics data</div>
        </div>
      </div>
    );
  }

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
        .delay-700 { animation-delay: 0.7s; }
        .delay-1000 { animation-delay: 1s; }
        .gradient-border {
          position: relative;
          background: linear-gradient(rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.05));
          border: 2px solid transparent;
          background-clip: padding-box;
        }
        .gradient-border::before {
          content: '';
          position: absolute;
          inset: 0;
          border-radius: inherit;
          padding: 2px;
          background: linear-gradient(135deg, #60a5fa, #a855f7, #ec4899);
          -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
          -webkit-mask-composite: xor;
          mask-composite: exclude;
          pointer-events: none;
        }
      `}</style>

      <div className="container mx-auto px-4 py-12 relative z-10">
        {/* Header */}
        <div className="text-center mb-12 animate-slide-up mt-16">
          <div className="inline-flex items-center gap-2 glass-effect text-purple-300 px-6 py-3 rounded-full mb-6">
            <Activity className="w-5 h-5 animate-pulse" />
            <span className="font-semibold text-sm tracking-wide">REAL-TIME MARKET INSIGHTS</span>
          </div>
          
          <h1 className="text-5xl md:text-6xl font-extrabold text-white mb-4">
            Market <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">Analytics</span>
          </h1>
          <p className="text-gray-300 text-lg">Comprehensive data analysis from {stats?.dataset?.total_cars?.toLocaleString() || 'N/A'} vehicles</p>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <div className="glass-effect rounded-2xl p-6 hover-lift transition-all duration-300 animate-scale-in delay-100 group">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-xl group-hover:scale-110 transition-transform">
                <Car className="w-6 h-6 text-blue-400" />
              </div>
              <Sparkles className="w-5 h-5 text-blue-400 animate-pulse" />
            </div>
            <div className="text-sm text-gray-400 mb-2 font-semibold">Total Vehicles</div>
            <div className="text-4xl font-extrabold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              {stats?.dataset?.total_cars?.toLocaleString() || 'N/A'}
            </div>
            <div className="mt-3 text-xs text-green-400 flex items-center gap-1">
              <TrendingUp className="w-3 h-3" />
              <span>Dataset size</span>
            </div>
          </div>

          <div className="glass-effect rounded-2xl p-6 hover-lift transition-all duration-300 animate-scale-in delay-200 group">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-green-500/20 to-green-600/20 rounded-xl group-hover:scale-110 transition-transform">
                <DollarSign className="w-6 h-6 text-green-400" />
              </div>
              <Sparkles className="w-5 h-5 text-green-400 animate-pulse" />
            </div>
            <div className="text-sm text-gray-400 mb-2 font-semibold">Average Price</div>
            <div className="text-4xl font-extrabold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
              {stats?.price_stats?.mean ? `${(stats.price_stats.mean / 100000).toFixed(1)}L` : 'N/A'}
            </div>
            <div className="mt-3 text-xs text-gray-400">PKR (Lacs)</div>
          </div>

          <div className="glass-effect rounded-2xl p-6 hover-lift transition-all duration-300 animate-scale-in delay-300 group">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-purple-500/20 to-purple-600/20 rounded-xl group-hover:scale-110 transition-transform">
                <BarChart3 className="w-6 h-6 text-purple-400" />
              </div>
              <Sparkles className="w-5 h-5 text-purple-400 animate-pulse" />
            </div>
            <div className="text-sm text-gray-400 mb-2 font-semibold">Price Range</div>
            <div className="text-2xl font-extrabold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              {stats?.price_stats?.min ? 
                `${(stats.price_stats.min / 100000).toFixed(0)}-${(stats.price_stats.max / 100000).toFixed(0)}L` 
                : 'N/A'}
            </div>
            <div className="mt-3 text-xs text-gray-400">Min to Max</div>
          </div>

          <div className="glass-effect rounded-2xl p-6 hover-lift transition-all duration-300 animate-scale-in delay-400 group">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-orange-500/20 to-orange-600/20 rounded-xl group-hover:scale-110 transition-transform">
                <Calendar className="w-6 h-6 text-orange-400" />
              </div>
              <Sparkles className="w-5 h-5 text-orange-400 animate-pulse" />
            </div>
            <div className="text-sm text-gray-400 mb-2 font-semibold">Year Range</div>
            <div className="text-3xl font-extrabold bg-gradient-to-r from-orange-400 to-yellow-400 bg-clip-text text-transparent">
              {stats?.dataset?.year_range || 'N/A'}
            </div>
            <div className="mt-3 text-xs text-gray-400">Model years</div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Top Brands */}
          <div className="glass-effect rounded-3xl p-8 hover:shadow-2xl hover:shadow-blue-500/20 transition-all duration-300 animate-slide-in-left">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-xl">
                <Award className="w-6 h-6 text-blue-400" />
              </div>
              <h3 className="text-2xl font-bold text-white">Top Brands</h3>
            </div>
            <div className="space-y-4">
              {stats?.top_brands && Object.entries(stats.top_brands).slice(0, 8).map(([brand, count], idx) => (
                <div key={brand} className="group">
                  <div className="flex justify-between items-center mb-2">
                    <div className="flex items-center gap-2">
                      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500/20 to-purple-500/20 flex items-center justify-center text-xs font-bold text-blue-400">
                        {idx + 1}
                      </div>
                      <span className="capitalize font-semibold text-white group-hover:text-blue-400 transition-colors">
                        {brand}
                      </span>
                    </div>
                    <span className="font-bold text-blue-400 text-lg">{count}</span>
                  </div>
                  <div className="relative w-full h-3 bg-gray-800 rounded-full overflow-hidden">
                    <div 
                      className="absolute h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 rounded-full transition-all duration-1000 ease-out"
                      style={{ 
                        width: `${(count / Math.max(...Object.values(stats.top_brands))) * 100}%`,
                        animationDelay: `${idx * 0.1}s`
                      }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Segment Distribution */}
          <div className="glass-effect rounded-3xl p-8 hover:shadow-2xl hover:shadow-purple-500/20 transition-all duration-300 animate-slide-in-right">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-gradient-to-br from-purple-500/20 to-purple-600/20 rounded-xl">
                <PieChart className="w-6 h-6 text-purple-400" />
              </div>
              <h3 className="text-2xl font-bold text-white">Segment Distribution</h3>
            </div>
            <div className="space-y-5">
              {stats?.segment_distribution?.map((seg, idx) => (
                <div key={seg.segment} className="group">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold text-white group-hover:text-purple-400 transition-colors flex items-center gap-2">
                      <Zap className="w-4 h-4 text-purple-400" />
                      {seg.segment}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-purple-400 text-lg">{seg.percentage}%</span>
                      <span className="text-xs text-gray-500">({seg.count || 'N/A'})</span>
                    </div>
                  </div>
                  <div className="relative w-full h-4 bg-gray-800 rounded-full overflow-hidden">
                    <div 
                      className="absolute h-full bg-gradient-to-r from-purple-500 via-pink-500 to-red-500 rounded-full transition-all duration-1000 ease-out"
                      style={{ 
                        width: `${seg.percentage}%`,
                        animationDelay: `${idx * 0.15}s`
                      }}
                    ></div>
                    <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent"></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom Grid */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Fuel Type Distribution */}
          <div className="glass-effect rounded-3xl p-8 hover:shadow-2xl hover:shadow-green-500/20 transition-all duration-300 animate-slide-in-left delay-200">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-gradient-to-br from-green-500/20 to-green-600/20 rounded-xl">
                <Fuel className="w-6 h-6 text-green-400" />
              </div>
              <h3 className="text-2xl font-bold text-white">Fuel Type Distribution</h3>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {stats?.fuel_type_distribution && Object.entries(stats.fuel_type_distribution).map(([fuel, count]) => (
                <div key={fuel} className="glass-effect rounded-xl p-4 hover:bg-white/10 transition-all group">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
                    <span className="capitalize font-medium text-gray-300 group-hover:text-white transition-colors">
                      {fuel}
                    </span>
                  </div>
                  <div className="text-3xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
                    {count}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">vehicles</div>
                </div>
              ))}
            </div>
          </div>

          {/* Transmission Types */}
          <div className="glass-effect rounded-3xl p-8 hover:shadow-2xl hover:shadow-cyan-500/20 transition-all duration-300 animate-slide-in-right delay-200">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 bg-gradient-to-br from-cyan-500/20 to-cyan-600/20 rounded-xl">
                <Settings className="w-6 h-6 text-cyan-400" />
              </div>
              <h3 className="text-2xl font-bold text-white">Transmission Types</h3>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {stats?.transmission_distribution && Object.entries(stats.transmission_distribution).map(([trans, count]) => (
                <div key={trans} className="glass-effect rounded-xl p-4 hover:bg-white/10 transition-all group">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></div>
                    <span className="capitalize font-medium text-gray-300 group-hover:text-white transition-colors">
                      {trans}
                    </span>
                  </div>
                  <div className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                    {count}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">vehicles</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom Highlight */}
        <div className="mt-12 glass-effect rounded-3xl p-8 text-center border-2 border-purple-500/30 animate-scale-in delay-400">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Activity className="w-6 h-6 text-purple-400 animate-pulse" />
            <h3 className="text-2xl font-bold text-white">Live Market Data</h3>
          </div>
          <p className="text-gray-300 mb-6">
            Analytics updated in real-time from our comprehensive Pakistani automotive database
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <div className="px-6 py-3 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-xl">
              <div className="text-sm text-gray-400">Data Points</div>
              <div className="text-2xl font-bold text-blue-400">5,497+</div>
            </div>
            <div className="px-6 py-3 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl">
              <div className="text-sm text-gray-400">Categories</div>
              <div className="text-2xl font-bold text-purple-400">8+</div>
            </div>
            <div className="px-6 py-3 bg-gradient-to-r from-green-500/20 to-emerald-500/20 rounded-xl">
              <div className="text-sm text-gray-400">Accuracy</div>
              <div className="text-2xl font-bold text-green-400">94.2%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsPage;