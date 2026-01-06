import React, { useState, useEffect, useRef } from 'react';
import { Loader, Calculator, Car, MapPin, Fuel, Gauge, Settings, Calendar, TrendingUp, AlertCircle, CheckCircle, Sparkles, Download, ShieldCheck } from 'lucide-react';
import jsPDF from 'jspdf';
import * as htmlToImage from 'html-to-image';

const PredictPage = ({ showToast }) => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [featureOptions, setFeatureOptions] = useState(null);
  const [loadingOptions, setLoadingOptions] = useState(true);
  const reportRef = useRef(null); 

  const [formData, setFormData] = useState({
    car_brand: '', car_model: '', city: '', fuel_type: '',
    engine: '', transmission: '', registered_in: '', mileage: ''
  });

  // --- Fixed Progress Bar Logic ---
  const calculateProgress = () => {
    const fields = Object.values(formData);
    const filledFields = fields.filter(f => f !== '').length;
    return (filledFields / fields.length) * 100;
  };

  // --- Perfected PDF Generation ---
  const generatePDF = async () => {
    if (!reportRef.current) return;

    showToast('Finalizing certificate layout...', 'info');
    setLoading(true);

    try {
      // Capture full element including hidden parts
      const dataUrl = await htmlToImage.toPng(reportRef.current, {
        quality: 1,
        pixelRatio: 2,
        backgroundColor: '#0f172a',
        style: {
            transform: 'scale(1)', // Reset any transforms
        }
      });

      const pdf = new jsPDF('p', 'mm', 'a4');
      const imgProps = pdf.getImageProperties(dataUrl);
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;

      // Adjust to fit page and center
      pdf.addImage(dataUrl, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save(`AutoAI_Official_Report.pdf`);
      
      showToast('Certificate saved!', 'success');
    } catch (error) {
      console.error('PDF Error:', error);
      showToast('Rendering failed. Try desktop browser.', 'error');
    } finally {
      setLoading(false);
    }
  };

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
          showToast('Failed to sync options', 'error');
        }
      }
    };
    fetchOptions();
    return () => { mounted = false; };
  }, []);

  const handleSubmit = async () => {
    if (Object.values(formData).some(val => val === '')) {
      showToast('All parameters required for analysis', 'error');
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...formData,
          engine: parseInt(formData.engine),
          registered_in: parseInt(formData.registered_in),
          mileage: parseInt(formData.mileage)
        })
      });
      const data = await response.json();
      if (data.success) {
        setResult(data);
        showToast('AI valuation generated!', 'success');
      }
    } catch (error) {
      showToast('Connection error', 'error');
    } finally {
      setLoading(false);
    }
  };

  if (loadingOptions) {
    return (
      <div className="min-h-screen bg-[#0f172a] flex flex-col items-center justify-center">
        <Loader className="w-10 h-10 animate-spin text-blue-500 mb-4" />
        <p className="text-slate-500 text-xs font-black tracking-[0.3em]">SYNCHRONIZING MODELS</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 pt-28 pb-20 selection:bg-blue-500/30">
      <style>{`
        .glass-input { background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); }
        .glass-input:focus { border-color: #3b82f6; background: rgba(30, 41, 59, 0.6); outline: none; }
        .prediction-card { background: linear-gradient(160deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.95)); border: 1px solid rgba(255, 255, 255, 0.05); }
        /* Watermark style back */
        .watermark-text { 
           position: absolute; top: 50%; left: 50%; 
           transform: translate(-50%, -50%) rotate(-25deg);
           font-size: 8rem; font-weight: 900; color: white; opacity: 0.02;
           pointer-events: none; white-space: nowrap; z-index: 0;
        }
      `}</style>

      <div className="container mx-auto px-6 max-w-7xl">
        <div className="mb-12 animate-in fade-in slide-in-from-top-4 duration-700">
           <h1 className="text-4xl md:text-6xl font-black text-white mb-2 tracking-tighter">AI <span className="text-blue-500">Predictor</span></h1>
           <p className="text-slate-500 font-medium">Neural valuation engine for Pakistani automotive market.</p>
        </div>

        <div className="grid lg:grid-cols-12 gap-10">
          {/* Left: Input Form */}
          <div className="lg:col-span-7 space-y-6">
            {/* Progress Bar Container */}
            <div className="bg-slate-900/50 p-6 rounded-3xl border border-white/5 shadow-inner">
               <div className="flex justify-between items-end mb-3">
                  <span className="text-[10px] font-black uppercase tracking-widest text-slate-500">Analysis Readiness</span>
                  <span className="text-blue-500 font-bold text-sm">{Math.round(calculateProgress())}%</span>
               </div>
               <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-blue-600 to-indigo-500 transition-all duration-700 shadow-[0_0_15px_rgba(37,99,235,0.5)]" 
                    style={{ width: `${calculateProgress()}%` }}
                  />
               </div>
            </div>

            <div className="prediction-card p-8 rounded-[2.5rem] shadow-2xl relative overflow-hidden">
              <div className="grid md:grid-cols-2 gap-6 relative z-10">
                <FormSelect icon={<Car/>} label="Brand" value={formData.car_brand} onChange={(e) => setFormData({...formData, car_brand: e.target.value, car_model: ''})} options={featureOptions.car_brand} />
                <FormSelect icon={<Settings/>} label="Model" value={formData.car_model} onChange={(e) => setFormData({...formData, car_model: e.target.value})} options={featureOptions.car_model} disabled={!formData.car_brand} />
                <FormInput icon={<MapPin/>} label="City" value={formData.city} placeholder="City Name" onChange={(e) => setFormData({...formData, city: e.target.value})} />
                <FormSelect icon={<Fuel/>} label="Fuel" value={formData.fuel_type} onChange={(e) => setFormData({...formData, fuel_type: e.target.value})} options={featureOptions.fuel_type} />
                <FormInput icon={<Gauge/>} label="Engine" type="number" value={formData.engine} placeholder="CC e.g. 1300" onChange={(e) => setFormData({...formData, engine: e.target.value})} />
                <FormSelect icon={<TrendingUp/>} label="Gear" value={formData.transmission} onChange={(e) => setFormData({...formData, transmission: e.target.value})} options={featureOptions.transmission} />
                <FormInput icon={<Calendar/>} label="Year" type="number" value={formData.registered_in} placeholder="Registration Year" onChange={(e) => setFormData({...formData, registered_in: e.target.value})} />
                <FormInput icon={<TrendingUp/>} label="ODO" type="number" value={formData.mileage} placeholder="Total KM" onChange={(e) => setFormData({...formData, mileage: e.target.value})} />
              </div>
              <button 
                onClick={handleSubmit} 
                disabled={loading}
                className="w-full mt-10 py-5 bg-blue-600 hover:bg-blue-500 text-white rounded-2xl font-black transition-all shadow-xl shadow-blue-900/40 active:scale-95 disabled:opacity-50 flex items-center justify-center gap-3"
              >
                {loading ? <Loader className="animate-spin"/> : "Generate AI Valuation"}
              </button>
            </div>
          </div>

          {/* Right: Result Certificate */}
          <div className="lg:col-span-5">
            {result ? (
              <div className="sticky top-28 space-y-6 animate-in zoom-in-95 duration-500">
                <div ref={reportRef} className="prediction-card p-10 rounded-[2.5rem] relative overflow-hidden bg-[#0f172a]">
                  <div className="watermark-text tracking-widest">CERTIFIED</div>
                  
                  <div className="relative z-10">
                    <div className="flex justify-between items-start mb-10 border-b border-white/10 pb-6">
                        <div>
                            <div className="text-blue-500 font-black text-xl italic tracking-tighter">AUTOAI <span className="text-white not-italic">PREDICTOR</span></div>
                            <div className="text-[8px] text-slate-500 uppercase tracking-widest font-black">Official Verification v2.0</div>
                        </div>
                        <div className="text-right text-[10px] font-bold text-slate-400">
                           {new Date().toLocaleDateString()}
                        </div>
                    </div>

                    <div className="mb-10">
                      <div className="text-6xl font-black text-white tracking-tighter mb-1">{result.price_display.formatted}</div>
                      <div className="text-xl font-bold text-emerald-500 tracking-tight">{result.price_display.lacs} Lacs PKR</div>
                    </div>

                    <div className="grid grid-cols-2 gap-3 mb-8">
                      <ReportDetail label="Brand" value={formData.car_brand}/>
                      <ReportDetail label="Variant" value={formData.car_model}/>
                      <ReportDetail label="Mileage" value={`${Number(formData.mileage).toLocaleString()} KM`}/>
                      <ReportDetail label="Year" value={formData.registered_in}/>
                    </div>

                    <div className="pt-6 border-t border-white/10 flex justify-between items-center text-[10px] font-black tracking-[0.2em] text-slate-500 uppercase">
                       <span>Market Accuracy</span>
                       <span className="text-emerald-500">Verified by AutoAI</span>
                    </div>
                  </div>
                </div>

                <button 
                  onClick={generatePDF}
                  disabled={loading}
                  className="w-full py-5 bg-slate-800 hover:bg-slate-700 border border-white/10 text-white rounded-3xl font-black transition-all flex items-center justify-center gap-3 shadow-2xl"
                >
                  <Download className="w-5 h-5 text-blue-500" />
                  DOWNLOAD FULL CERTIFICATE
                </button>
              </div>
            ) : (
              <div className="sticky top-28 h-[550px] border-2 border-dashed border-white/5 rounded-[3rem] flex flex-col items-center justify-center p-12 text-center">
                 <Calculator className="w-12 h-12 text-slate-800 mb-4" />
                 <h3 className="text-slate-400 font-bold">Waiting for Specs</h3>
                 <p className="text-slate-600 text-xs mt-2">Fill the form to unlock AI valuation.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const ReportDetail = ({ label, value }) => (
  <div className="bg-slate-900/50 p-4 rounded-2xl border border-white/5 text-center">
    <div className="text-[8px] uppercase text-slate-500 font-black mb-1 tracking-widest">{label}</div>
    <div className="text-[11px] font-black text-white uppercase truncate">{value}</div>
  </div>
);

const FormInput = ({ icon, label, ...props }) => (
  <div className="space-y-2">
    <label className="flex items-center gap-2 text-[10px] font-black uppercase text-slate-500 tracking-[0.15em]">{icon} {label}</label>
    <input className="w-full px-6 py-4 glass-input rounded-2xl text-white placeholder:text-slate-700 font-bold text-sm transition-all" {...props} />
  </div>
);

const FormSelect = ({ icon, label, options, ...props }) => (
  <div className="space-y-2">
    <label className="flex items-center gap-2 text-[10px] font-black uppercase text-slate-500 tracking-[0.15em]">{icon} {label}</label>
    <select className="w-full px-6 py-4 glass-input rounded-2xl text-white appearance-none cursor-pointer font-bold text-sm transition-all" {...props}>
      <option value="">Choose {label}</option>
      {options?.map(opt => <option key={opt} value={opt} className="bg-slate-900">{opt.toUpperCase()}</option>)}
    </select>
  </div>
);

export default PredictPage;