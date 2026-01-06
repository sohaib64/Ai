import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import HomePage from './pages/Home';
import PredictPage from './pages/Predict';
import AnalyticsPage from './pages/Analytics';
import AboutPage from './pages/About';
import FAQPage from './pages/FAQ';
import Chatbot from './components/Chatbot';
import ToastContainer from './components/ToastContainer';
import { Sparkles } from 'lucide-react';
import { fetchModelInfo } from './services/api';

function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [chatOpen, setChatOpen] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const [toasts, setToasts] = useState([]);

  useEffect(() => {
    fetchModelInfo()
      .then(data => setModelInfo(data))
      .catch(err => console.error('Failed to load model info:', err));

    // Support for internal FAQ chat trigger
    const handleOpenChat = () => setChatOpen(true);
    window.addEventListener('openChat', handleOpenChat);
    return () => window.removeEventListener('openChat', handleOpenChat);
  }, []);

  const showToast = (message, type = 'info') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, message, type }]);
  };

  const removeToast = (id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  };

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 selection:bg-blue-500/30 font-sans overflow-x-hidden">
      <style>{`
        .page-fade-enter { animation: fadeIn 0.4s ease-out forwards; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        ${chatOpen ? 'body { overflow: hidden; }' : ''}
      `}</style>

      {/* Navbar Z-Index set to 40 */}
      <Navbar 
        currentPage={currentPage} 
        setCurrentPage={setCurrentPage}
        openChat={() => setChatOpen(true)}
      />

      <main className="relative z-10 page-fade-enter">
        {currentPage === 'home' && <HomePage modelInfo={modelInfo} setCurrentPage={setCurrentPage} openChat={() => setChatOpen(true)} />}
        {currentPage === 'predict' && <PredictPage showToast={showToast} />}
        {currentPage === 'analytics' && <AnalyticsPage showToast={showToast} />}
        {currentPage === 'about' && <AboutPage modelInfo={modelInfo} />}
        {currentPage === 'faq' && <FAQPage />}
      </main>

      {/* Chatbot Z-Index is set to 100 in its own component */}
      <Chatbot 
        isOpen={chatOpen}
        onClose={() => setChatOpen(false)}
        showToast={showToast}
      />

      <ToastContainer toasts={toasts} removeToast={removeToast} />

      {!chatOpen && (
        <button
          onClick={() => setChatOpen(true)}
          className="fixed bottom-8 right-8 w-16 h-16 bg-gradient-to-br from-blue-600 to-indigo-700 text-white rounded-2xl shadow-[0_0_30px_rgba(37,99,235,0.4)] hover:scale-110 active:scale-95 transition-all flex items-center justify-center z-[30] group animate-bounce"
        >
          <Sparkles className="w-8 h-8 relative z-10" />
        </button>
      )}

      <footer className="py-12 text-center border-t border-white/5 relative z-10 mt-20 bg-slate-950/20">
        <div className="container mx-auto px-6">
           <div className="flex flex-col items-center gap-4">
              <div className="text-xl font-black tracking-tighter text-white">
                AUTO<span className="text-blue-500">AI</span>
              </div>
              <p className="text-slate-500 text-xs max-w-xs leading-relaxed font-medium uppercase tracking-widest">
                Pakistan's Premium AI Car Valuation Engine
              </p>
              <div className="flex gap-6 mt-4">
                {['home', 'predict', 'analytics', 'faq', 'about'].map(id => (
                  <button key={id} onClick={() => setCurrentPage(id)} className="text-[10px] font-bold text-slate-600 hover:text-blue-400 uppercase tracking-widest transition-colors">
                    {id}
                  </button>
                ))}
              </div>
              <p className="text-slate-700 text-[10px] mt-8 uppercase tracking-[0.4em] font-black">
                © 2026 AUTOAI PREDICTOR
              </p>
           </div>
        </div>
      </footer>
    </div>
  );
}

export default App;