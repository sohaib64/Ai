// import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
// import { Toaster } from 'react-hot-toast';
// // import Navbar from './components/Navbar';
// import Home from './pages/Home';
// // import Predict from './pages/Predict';
// // import Analytics from './pages/Analytics';
// // import About from './pages/About';

// function App() {
//   return (
//     <Router>
//       <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
//         {/* <Navbar /> */}
        
//         <Routes>
//           <Route path="/" element={<Home />} />
//           {/* <Route path="/predict" element={<Predict />} />
//           <Route path="/analytics" element={<Analytics />} />
//           <Route path="/about" element={<About />} /> */}
//         </Routes>
        
//         <Toaster
//           position="top-right"
//           toastOptions={{
//             duration: 4000,
//             style: {
//               background: '#fff',
//               color: '#363636',
//               boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
//             },
//             success: {
//               iconTheme: {
//                 primary: '#10B981',
//                 secondary: '#fff',
//               },
//             },
//             error: {
//               iconTheme: {
//                 primary: '#EF4444',
//                 secondary: '#fff',
//               },
//             },
//           }}
//         />
//       </div>
//     </Router>
//   );
// }

// export default App;


import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import HomePage from './pages/Home';
import PredictPage from './pages/Predict';
import AnalyticsPage from './pages/Analytics';
import AboutPage from './pages/About';
import Chatbot from './components/Chatbot';
import ToastContainer from './components/ToastContainer';
import { MessageCircle } from 'lucide-react';
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
  }, []);

  const showToast = (message, type = 'info') => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, message, type }]);
  };

  const removeToast = (id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <style>{`
        @keyframes slide-in {
          from {
            transform: translateX(100%);
            opacity: 0;
          }
          to {
            transform: translateX(0);
            opacity: 1;
          }
        }
        .animate-slide-in {
          animation: slide-in 0.3s ease-out;
        }
      `}</style>

      <Navbar 
        currentPage={currentPage} 
        setCurrentPage={setCurrentPage}
        openChat={() => setChatOpen(true)}
      />

      {currentPage === 'home' && (
        <HomePage 
          modelInfo={modelInfo}
          setCurrentPage={setCurrentPage}
          openChat={() => setChatOpen(true)}
        />
      )}
      
      {currentPage === 'predict' && (
        <PredictPage showToast={showToast} />
      )}
      
      {currentPage === 'analytics' && (
        <AnalyticsPage showToast={showToast} />
      )}
      
      {currentPage === 'about' && (
        <AboutPage modelInfo={modelInfo} />
      )}

      <Chatbot 
        isOpen={chatOpen}
        onClose={() => setChatOpen(false)}
        showToast={showToast}
      />

      <ToastContainer toasts={toasts} removeToast={removeToast} />

      {/* Floating Chat Button */}
      {!chatOpen && (
        <button
          onClick={() => setChatOpen(true)}
          className="fixed bottom-6 right-6 w-14 h-14 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-full shadow-lg hover:shadow-xl transition-all flex items-center justify-center z-40"
        >
          <MessageCircle className="w-6 h-6" />
        </button>
      )}
    </div>
  );
}

export default App;