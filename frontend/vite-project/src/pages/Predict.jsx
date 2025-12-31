import React, { useState, useEffect } from 'react';
import { Loader, Calculator, Car, MapPin, Fuel, Gauge, Settings, Calendar, TrendingUp, AlertCircle, CheckCircle, Sparkles } from 'lucide-react';

const PredictPage = ({ showToast }) => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [featureOptions, setFeatureOptions] = useState(null);
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [formData, setFormData] = useState({
    car_brand: '',
    car_model: '',
    city: '',
    fuel_type: '',
    engine: '',
    transmission: '',
    registered_in: '',
    mileage: ''
  });

  useEffect(() => {
    let mounted = true;

    const fetchOptions = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/feature-options');
        const data = await response.json();

        if (mounted) {
          setFeatureOptions(data.options);
          setLoadingOptions(false);
        }
      } catch (err) {
        if (mounted) {
          setLoadingOptions(false);
          showToast('Failed to load options', 'error');
        }
      }
    };

    fetchOptions();

    return () => {
      mounted = false;
    };
  }, []);

  const handleSubmit = async () => {
    // Validate all fields are filled
    if (!formData.car_brand || !formData.car_model || !formData.city ||
      !formData.fuel_type || !formData.engine || !formData.transmission ||
      !formData.registered_in || !formData.mileage) {
      showToast('Please fill all fields', 'error');
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      // Convert numeric fields to integers
      const submissionData = {
        ...formData,
        engine: parseInt(formData.engine),
        registered_in: parseInt(formData.registered_in),
        mileage: parseInt(formData.mileage)
      };

      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(submissionData)
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
        showToast('Price predicted successfully!', 'success');
      } else {
        showToast(data.error || 'Prediction failed', 'error');
      }
    } catch (error) {
      showToast('Network error. Please try again.', 'error');
    } finally {
      setLoading(false);
    }
  };

  if (loadingOptions) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <Loader className="w-16 h-16 animate-spin text-blue-400 mx-auto mb-4" />
          <p className="text-gray-300 text-lg">Loading prediction system...</p>
        </div>
      </div>
    );
  }

  if (!featureOptions) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-center glass-effect p-8 rounded-2xl">
          <AlertCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
          <div className="text-red-400 text-xl mb-4">Failed to load form options</div>
          <button
            onClick={() => window.location.reload()}
            className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-semibold hover:shadow-lg transition-all"
          >
            Retry
          </button>
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
        @keyframes slideInLeft {
          from { opacity: 0; transform: translateX(-50px); }
          to { opacity: 1; transform: translateX(0); }
        }
        @keyframes slideInRight {
          from { opacity: 0; transform: translateX(50px); }
          to { opacity: 1; transform: translateX(0); }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes pulse-glow {
          0%, 100% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.5); }
          50% { box-shadow: 0 0 40px rgba(99, 102, 241, 0.8); }
        }
        .animate-slide-in-left {
          animation: slideInLeft 0.6s ease-out forwards;
        }
        .animate-slide-in-right {
          animation: slideInRight 0.6s ease-out forwards;
        }
        .animate-fade-in {
          animation: fadeIn 0.8s ease-out forwards;
        }
        .glass-effect {
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .input-glow:focus {
          box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
          border-color: rgba(99, 102, 241, 0.8);
        }
        .delay-700 { animation-delay: 0.7s; }
        .delay-1000 { animation-delay: 1s; }
      `}</style>

      <div className="container mx-auto px-4 py-12 relative z-10">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12 animate-fade-in mt-16">
            <div className="inline-flex items-center gap-2 glass-effect text-blue-300 px-6 py-4 mt-4 rounded-full mb-4">
              <Sparkles className="w-5 h-5 animate-pulse" />
              <span className="font-semibold text-sm">AI-POWERED PREDICTION ENGINE</span>
            </div>

            <h1 className="text-5xl md:text-6xl font-extrabold text-white mb-4">
              Get Your <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">Instant Price</span>
            </h1>
            <p className="text-gray-300 text-lg">Enter your car details and let our AI calculate the perfect market value</p>
          </div>


          <div className="grid lg:grid-cols-2 gap-8">
            {/* Form Section */}
            <div className="animate-slide-in-left">
              <div className="glass-effect rounded-3xl p-8 hover:shadow-2xl hover:shadow-blue-500/20 transition-all duration-300">
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-3 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-xl">
                    <Car className="w-6 h-6 text-blue-400" />
                  </div>
                  <h2 className="text-2xl font-bold text-white">Vehicle Details</h2>
                </div>

                <div className="space-y-5">
                  {/* Car Brand */}
                  <div className="group">
                    <label className="flex items-center gap-2 text-sm font-semibold text-gray-300 mb-2">
                      <Car className="w-4 h-4 text-blue-400" />
                      Car Brand
                    </label>
                    <select
                      value={formData.car_brand}
                      onChange={(e) => setFormData({ ...formData, car_brand: e.target.value })}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none input-glow transition-all"
                    >
                      <option value="" className="bg-slate-800">Select Brand</option>
                      {featureOptions.car_brand?.map(brand => (
                        <option key={brand} value={brand} className="bg-slate-800">{brand.toUpperCase()}</option>
                      ))}
                    </select>
                  </div>

                  {/* Car Model */}
                  <div className="group">
                    <label className="flex items-center gap-2 text-sm font-semibold text-gray-300 mb-2">
                      <Settings className="w-4 h-4 text-purple-400" />
                      Car Model
                    </label>
                    <select
                      value={formData.car_model}
                      onChange={(e) => setFormData({ ...formData, car_model: e.target.value })}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none input-glow transition-all"
                    >
                      <option value="" className="bg-slate-800">Select Model</option>
                      {featureOptions.car_model?.map(model => (
                        <option key={model} value={model} className="bg-slate-800">{model.toUpperCase()}</option>
                      ))}
                    </select>
                  </div>

                  {/* City */}
                  <div className="group">
                    <label className="flex items-center gap-2 text-sm font-semibold text-gray-300 mb-2">
                      <MapPin className="w-4 h-4 text-green-400" />
                      City
                    </label>
                    <input
                      type="text"
                      value={formData.city}
                      onChange={(e) => setFormData({ ...formData, city: e.target.value })}
                      placeholder="e.g., Karachi, Lahore, Islamabad"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none input-glow transition-all"
                    />
                  </div>

                  {/* Fuel Type */}
                  <div className="group">
                    <label className="flex items-center gap-2 text-sm font-semibold text-gray-300 mb-2">
                      <Fuel className="w-4 h-4 text-orange-400" />
                      Fuel Type
                    </label>
                    <select
                      value={formData.fuel_type}
                      onChange={(e) => setFormData({ ...formData, fuel_type: e.target.value })}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none input-glow transition-all"
                    >
                      <option value="" className="bg-slate-800">Select Fuel Type</option>
                      {featureOptions.fuel_type?.map(fuel => (
                        <option key={fuel} value={fuel} className="bg-slate-800">{fuel.toUpperCase()}</option>
                      ))}
                    </select>
                  </div>

                  {/* Engine */}
                  <div className="group">
                    <label className="flex items-center gap-2 text-sm font-semibold text-gray-300 mb-2">
                      <Gauge className="w-4 h-4 text-red-400" />
                      Engine (cc)
                    </label>
                    <input
                      type="number"
                      value={formData.engine}
                      onChange={(e) => setFormData({ ...formData, engine: e.target.value })}
                      placeholder="e.g., 1300"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none input-glow transition-all"
                    />
                  </div>

                  {/* Transmission */}
                  <div className="group">
                    <label className="flex items-center gap-2 text-sm font-semibold text-gray-300 mb-2">
                      <Settings className="w-4 h-4 text-cyan-400" />
                      Transmission
                    </label>
                    <select
                      value={formData.transmission}
                      onChange={(e) => setFormData({ ...formData, transmission: e.target.value })}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none input-glow transition-all"
                    >
                      <option value="" className="bg-slate-800">Select Transmission</option>
                      {featureOptions.transmission?.map(trans => (
                        <option key={trans} value={trans} className="bg-slate-800">{trans.toUpperCase()}</option>
                      ))}
                    </select>
                  </div>

                  {/* Registration Year */}
                  <div className="group">
                    <label className="flex items-center gap-2 text-sm font-semibold text-gray-300 mb-2">
                      <Calendar className="w-4 h-4 text-yellow-400" />
                      Registration Year
                    </label>
                    <input
                      type="number"
                      value={formData.registered_in}
                      onChange={(e) => setFormData({ ...formData, registered_in: e.target.value })}
                      placeholder="e.g., 2018"
                      min="1990"
                      max="2025"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none input-glow transition-all"
                    />
                  </div>

                  {/* Mileage */}
                  <div className="group">
                    <label className="flex items-center gap-2 text-sm font-semibold text-gray-300 mb-2">
                      <TrendingUp className="w-4 h-4 text-pink-400" />
                      Mileage (km)
                    </label>
                    <input
                      type="number"
                      value={formData.mileage}
                      onChange={(e) => setFormData({ ...formData, mileage: e.target.value })}
                      placeholder="e.g., 50000"
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none input-glow transition-all"
                    />
                  </div>

                  {/* Submit Button */}
                  <button
                    onClick={handleSubmit}
                    disabled={loading}
                    className="w-full py-4 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 text-white rounded-xl font-bold text-lg hover:shadow-2xl hover:shadow-purple-500/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 flex items-center justify-center gap-3 mt-6"
                  >
                    {loading ? (
                      <>
                        <Loader className="w-6 h-6 animate-spin" />
                        <span>Calculating Price...</span>
                      </>
                    ) : (
                      <>
                        <Calculator className="w-6 h-6" />
                        <span>Predict My Car's Value</span>
                        <Sparkles className="w-5 h-5 animate-pulse" />
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Result Section */}
            <div className="animate-slide-in-right lg:sticky lg:top-8 h-fit">
              {result ? (
                <div className="glass-effect rounded-3xl p-8 border-2 border-purple-500/30 hover:shadow-2xl hover:shadow-purple-500/30 transition-all duration-300">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="p-3 bg-gradient-to-br from-green-500/20 to-emerald-500/20 rounded-xl">
                      <CheckCircle className="w-6 h-6 text-green-400" />
                    </div>
                    <h2 className="text-2xl font-bold text-white">Prediction Complete!</h2>
                  </div>

                  {/* Price Display */}
                  <div className="bg-gradient-to-br from-blue-500/20 to-purple-500/20 backdrop-blur rounded-2xl p-6 mb-6 border border-blue-400/30">
                    <div className="text-sm text-blue-300 mb-2 flex items-center gap-2">
                      <Sparkles className="w-4 h-4" />
                      Estimated Market Value
                    </div>
                    <div className="text-5xl font-extrabold text-white mb-2">
                      {result.price_display.formatted}
                    </div>
                    <div className="text-xl text-gray-300 font-semibold">
                      {result.price_display.lacs} Lacs PKR
                    </div>
                  </div>

                  {/* Details Grid */}
                  <div className="space-y-4 mb-6">
                    <div className="glass-effect rounded-xl p-4 hover:bg-white/10 transition-all">
                      <div className="flex justify-between items-center">
                        <span className="text-gray-300 flex items-center gap-2">
                          <TrendingUp className="w-4 h-4 text-blue-400" />
                          Price Range
                        </span>
                        <span className="font-bold text-white">
                          {result.price_range.min_display.lacs} - {result.price_range.max_display.lacs} Lacs
                        </span>
                      </div>
                    </div>

                    <div className="glass-effect rounded-xl p-4 hover:bg-white/10 transition-all">
                      <div className="flex justify-between items-center">
                        <span className="text-gray-300 flex items-center gap-2">
                          <Car className="w-4 h-4 text-purple-400" />
                          Vehicle Segment
                        </span>
                        <span className="font-bold text-white uppercase bg-purple-500/20 px-3 py-1 rounded-lg">
                          {result.segment}
                        </span>
                      </div>
                    </div>

                    <div className="glass-effect rounded-xl p-4 hover:bg-white/10 transition-all">
                      <div className="flex justify-between items-center">
                        <span className="text-gray-300 flex items-center gap-2">
                          <Calendar className="w-4 h-4 text-yellow-400" />
                          Car Age
                        </span>
                        <span className="font-bold text-white">
                          {result.car_info.age} years ({result.car_info.age_category})
                        </span>
                      </div>
                    </div>

                    <div className="glass-effect rounded-xl p-4 hover:bg-white/10 transition-all">
                      <div className="flex justify-between items-center">
                        <span className="text-gray-300 flex items-center gap-2">
                          <CheckCircle className="w-4 h-4 text-green-400" />
                          Confidence Level
                        </span>
                        <span className="font-bold text-white uppercase bg-green-500/20 px-3 py-1 rounded-lg">
                          {result.model_performance.confidence}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Warnings */}
                  {result.warnings && result.warnings.length > 0 && (
                    <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4">
                      <div className="flex items-center gap-2 font-bold text-yellow-400 mb-3">
                        <AlertCircle className="w-5 h-5" />
                        Important Notices
                      </div>
                      <div className="space-y-2">
                        {result.warnings.map((w, i) => (
                          <div key={i} className="text-sm text-yellow-200 flex items-start gap-2">
                            <span className="text-yellow-400 mt-0.5">•</span>
                            <span>{w}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="glass-effect rounded-3xl p-12 text-center min-h-[500px] flex flex-col items-center justify-center">
                  <div className="p-6 bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-full mb-6 animate-pulse">
                    <Calculator className="w-20 h-20 text-gray-400" />
                  </div>
                  <h3 className="text-2xl font-bold text-gray-300 mb-3">Awaiting Your Input</h3>
                  <p className="text-gray-400 max-w-sm">Complete the vehicle details form and click the prediction button to receive your instant AI-powered price estimate</p>
                  <div className="mt-8 flex gap-2">
                    <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse"></div>
                    <div className="w-3 h-3 bg-purple-400 rounded-full animate-pulse delay-200"></div>
                    <div className="w-3 h-3 bg-pink-400 rounded-full animate-pulse delay-500"></div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictPage;