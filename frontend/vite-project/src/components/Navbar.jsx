// import { Link, useLocation } from 'react-router-dom';
// import { Car, BarChart3, Info, Sparkles } from 'lucide-react';
// import { motion } from 'framer-motion';

// const Navbar = () => {
//   const location = useLocation();
  
//   const isActive = (path) => location.pathname === path;
  
//   const navItems = [
//     { path: '/', label: 'Home', icon: Sparkles },
//     { path: '/predict', label: 'Predict', icon: Car },
//     { path: '/analytics', label: 'Analytics', icon: BarChart3 },
//     { path: '/about', label: 'About', icon: Info },
//   ];
  
//   return (
//     <motion.nav
//       initial={{ y: -100 }}
//       animate={{ y: 0 }}
//       className="bg-white shadow-lg sticky top-0 z-50"
//     >
//       <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//         <div className="flex justify-between items-center h-16">
//           {/* Logo */}
//           <Link to="/" className="flex items-center space-x-3">
//             <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-2 rounded-lg">
//               <Car className="w-6 h-6 text-white" />
//             </div>
//             <span className="text-2xl font-bold gradient-text">
//               AI Car Predictor
//             </span>
//           </Link>
          
//           {/* Nav Links */}
//           <div className="hidden md:flex space-x-1">
//             {navItems.map((item) => {
//               const Icon = item.icon;
//               return (
//                 <Link
//                   key={item.path}
//                   to={item.path}
//                   className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
//                     isActive(item.path)
//                       ? 'bg-blue-600 text-white shadow-lg'
//                       : 'text-gray-600 hover:bg-blue-50'
//                   }`}
//                 >
//                   <Icon className="w-4 h-4" />
//                   <span className="font-medium">{item.label}</span>
//                 </Link>
//               );
//             })}
//           </div>
          
//           {/* Mobile Menu */}
//           <div className="md:hidden flex space-x-2">
//             {navItems.map((item) => {
//               const Icon = item.icon;
//               return (
//                 <Link
//                   key={item.path}
//                   to={item.path}
//                   className={`p-2 rounded-lg ${
//                     isActive(item.path)
//                       ? 'bg-blue-600 text-white'
//                       : 'text-gray-600 hover:bg-blue-50'
//                   }`}
//                 >
//                   <Icon className="w-5 h-5" />
//                 </Link>
//               );
//             })}
//           </div>
//         </div>
//       </div>
//     </motion.nav>
//   );
// };

// export default Navbar;


import React, { useState, useEffect } from 'react';
import { Menu, Car, Home, Calculator, BarChart3, Info, MessageCircle, X } from 'lucide-react';

const Navbar = ({ currentPage, setCurrentPage, openChat }) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'predict', label: 'Predict Price', icon: Calculator },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'about', label: 'About', icon: Info }
  ];

  return (
    <>
      <style>{`
        @keyframes slideDown {
          from { transform: translateY(-100%); opacity: 0; }
          to { transform: translateY(0); opacity: 1; }
        }
        @keyframes glow {
          0%, 100% { box-shadow: 0 0 20px rgba(139, 92, 246, 0.4); }
          50% { box-shadow: 0 0 30px rgba(139, 92, 246, 0.6), 0 0 40px rgba(59, 130, 246, 0.4); }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-5px); }
        }
        .nav-item-active {
          position: relative;
        }
        .nav-item-active::before {
          content: '';
          position: absolute;
          bottom: -2px;
          left: 0;
          right: 0;
          height: 3px;
          background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
          border-radius: 2px;
          animation: glow 2s ease-in-out infinite;
        }
        .mobile-menu-enter {
          animation: slideDown 0.3s ease-out;
        }
      `}</style>

      <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled 
          ? 'bg-slate-900/95 backdrop-blur-xl shadow-2xl shadow-purple-500/20' 
          : 'bg-gradient-to-r from-slate-900/90 via-purple-900/90 to-slate-900/90 backdrop-blur-md'
      }`}>
        {/* Animated top border */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 opacity-70"></div>
        
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-20">
            {/* Logo with animation */}
            <div 
              className="flex items-center gap-3 cursor-pointer group"
              onClick={() => setCurrentPage('home')}
            >
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl blur-lg opacity-50 group-hover:opacity-75 transition-opacity"></div>
                <div className="relative bg-gradient-to-br from-blue-600 to-purple-600 p-2.5 rounded-xl group-hover:scale-110 transition-transform">
                  <Car className="w-7 h-7 text-white" />
                </div>
              </div>
              <div>
                <span className="text-2xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                  AI Car Predictor
                </span>
                <div className="h-0.5 bg-gradient-to-r from-blue-400 to-purple-400 rounded-full transform origin-left scale-x-0 group-hover:scale-x-100 transition-transform"></div>
              </div>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-2">
              {navItems.map((item, index) => (
                <button
                  key={item.id}
                  onClick={() => setCurrentPage(item.id)}
                  className={`relative flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium transition-all duration-300 group ${
                    currentPage === item.id
                      ? 'nav-item-active text-white bg-white/10'
                      : 'text-gray-300 hover:text-white hover:bg-white/5'
                  }`}
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  {currentPage === item.id && (
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-xl blur-sm"></div>
                  )}
                  <item.icon className={`w-5 h-5 relative z-10 transition-transform ${
                    currentPage === item.id ? 'animate-pulse' : 'group-hover:scale-110'
                  }`} />
                  <span className="relative z-10">{item.label}</span>
                </button>
              ))}
              
              {/* Chat AI Button with special styling */}
              <button
                onClick={openChat}
                className="relative flex items-center gap-2 px-6 py-2.5 ml-2 rounded-xl font-semibold text-white overflow-hidden group transition-all duration-300 hover:scale-105"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-green-500 to-emerald-600"></div>
                <div className="absolute inset-0 bg-gradient-to-r from-emerald-400 to-green-500 opacity-0 group-hover:opacity-100 transition-opacity"></div>
                <div className="absolute inset-0 animate-pulse opacity-50">
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent transform -skew-x-12 translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-1000"></div>
                </div>
                <MessageCircle className="w-5 h-5 relative z-10 group-hover:rotate-12 transition-transform" />
                <span className="relative z-10">Chat AI</span>
              </button>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden relative p-3 hover:bg-white/10 rounded-xl transition-all group"
            >
              {mobileMenuOpen ? (
                <X className="w-6 h-6 text-white group-hover:rotate-90 transition-transform" />
              ) : (
                <Menu className="w-6 h-6 text-white group-hover:scale-110 transition-transform" />
              )}
            </button>
          </div>

          {/* Mobile Menu */}
          {mobileMenuOpen && (
            <div className="md:hidden pb-6 space-y-2 mobile-menu-enter">
              <div className="bg-white/5 backdrop-blur-lg rounded-2xl p-4 border border-white/10">
                {navItems.map((item, index) => (
                  <button
                    key={item.id}
                    onClick={() => {
                      setCurrentPage(item.id);
                      setMobileMenuOpen(false);
                    }}
                    className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all duration-300 mb-2 ${
                      currentPage === item.id
                        ? 'bg-gradient-to-r from-blue-500/20 to-purple-500/20 text-white font-semibold border border-purple-400/30'
                        : 'text-gray-300 hover:bg-white/10 hover:text-white'
                    }`}
                    style={{ animationDelay: `${index * 0.05}s` }}
                  >
                    <div className={`p-2 rounded-lg ${
                      currentPage === item.id 
                        ? 'bg-gradient-to-br from-blue-500/30 to-purple-500/30' 
                        : 'bg-white/5'
                    }`}>
                      <item.icon className="w-5 h-5" />
                    </div>
                    <span className="flex-1 text-left">{item.label}</span>
                    {currentPage === item.id && (
                      <div className="w-2 h-2 rounded-full bg-gradient-to-r from-blue-400 to-purple-400 animate-pulse"></div>
                    )}
                  </button>
                ))}
                
                <button
                  onClick={() => {
                    openChat();
                    setMobileMenuOpen(false);
                  }}
                  className="w-full flex items-center gap-3 px-4 py-3.5 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-emerald-500 hover:to-green-600 rounded-xl transition-all duration-300 text-white font-semibold mt-4 group"
                >
                  <div className="p-2 rounded-lg bg-white/20">
                    <MessageCircle className="w-5 h-5 group-hover:rotate-12 transition-transform" />
                  </div>
                  <span className="flex-1 text-left">Chat AI</span>
                  <div className="w-2 h-2 rounded-full bg-white animate-pulse"></div>
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Bottom glow effect */}
        <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-purple-500/50 to-transparent"></div>
      </nav>
    </>
  );
};

export default Navbar;