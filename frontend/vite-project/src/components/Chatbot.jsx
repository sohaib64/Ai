import React, { useState, useEffect, useRef } from 'react';
import { Bot, User, X, Send, Sparkles } from 'lucide-react';
import { sendChatMessage } from '../services/api';

const Chatbot = ({ isOpen, onClose, showToast }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const sessionId = useRef(`session_${Date.now()}`);

  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setMessages([{
        role: 'assistant',
        content: "👋 Assalam-o-Alaikum! Main aapki car price prediction mein madad karunga. Shuru karne ke liye 'predict' likhein!"
      }]);
    }
  }, [isOpen, messages.length]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);
    try {
      const data = await sendChatMessage(sessionId.current, userMsg);
      setMessages(prev => [...prev, { role: 'assistant', content: data.reply, prediction: data.prediction }]);
    } catch (error) {
      showToast('Error sending message', 'error');
    } finally { setLoading(false); }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-slate-950/60 backdrop-blur-sm flex items-end justify-end p-0 md:p-6 z-[100] animate-in fade-in duration-300">
      <div className="bg-[#0f172a] border border-white/10 shadow-2xl w-full md:w-[450px] h-full md:h-[750px] md:rounded-[2.5rem] flex flex-col overflow-hidden animate-in slide-in-from-bottom-10 duration-500">
        
        {/* --- IMPROVED HEADER --- */}
        <div className="relative bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 p-5 flex items-center justify-between shadow-xl">
          {/* Subtle Background Glow */}
          <div className="absolute -top-10 -right-10 w-32 h-32 bg-white/10 rounded-full blur-2xl"></div>
          
          <div className="flex items-center gap-4 relative z-10">
            <div className="w-11 h-11 bg-white/20 backdrop-blur-md rounded-2xl flex items-center justify-center border border-white/30 shadow-inner">
              <Bot className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="font-black text-white tracking-tight">AutoAI Concierge</h3>
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]"></span>
                <span className="text-[10px] uppercase tracking-widest text-blue-100 font-bold">Online</span>
              </div>
            </div>
          </div>
          
          {/* FIX: Better Padding & Positioning for Close Button */}
          <button 
            type="button"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              onClose();
            }} 
            className="relative z-20 p-2.5 bg-white/10 hover:bg-white/20 rounded-xl transition-all active:scale-90 border border-white/10 group"
          >
            <X className="w-5 h-5 text-white group-hover:rotate-90 transition-transform duration-300" />
          </button>
        </div>

        {/* --- MESSAGES AREA --- */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-[#0f172a] custom-scrollbar">
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in slide-in-from-bottom-2`}>
              <div className={`max-w-[85%] px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-lg ${
                msg.role === 'user' 
                  ? 'bg-blue-600 text-white rounded-tr-none' 
                  : 'bg-slate-800 text-slate-200 rounded-tl-none border border-white/5'
              }`}>
                {msg.content}
                
                {msg.prediction && msg.prediction.success && (
                  <div className="mt-3 p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-xl">
                     <div className="text-xl font-black text-emerald-400">{msg.prediction.price_display.formatted}</div>
                     <div className="text-[10px] text-emerald-400/60 font-bold uppercase mt-1">AI Predicted Value</div>
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-slate-800 px-4 py-3 rounded-2xl rounded-tl-none flex gap-1.5">
                <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce"></div>
                <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:0.2s]"></div>
                <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:0.4s]"></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* --- INPUT AREA --- */}
        <div className="p-5 bg-slate-900 border-t border-white/5">
          <div className="flex gap-2">
            <input 
              value={input} 
              onChange={(e) => setInput(e.target.value)} 
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              className="flex-1 bg-slate-800 text-white px-5 py-3.5 rounded-2xl outline-none border border-white/10 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all text-sm"
              placeholder="Ask about car prices..."
              disabled={loading}
            />
            <button 
              onClick={sendMessage} 
              disabled={loading || !input.trim()}
              className="p-4 bg-blue-600 hover:bg-blue-500 text-white rounded-2xl shadow-lg shadow-blue-900/40 transition-all active:scale-95 disabled:opacity-50"
            >
              <Send className="w-5 h-5"/>
            </button>
          </div>
        </div>
      </div>

      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
      `}</style>
    </div>
  );
};

export default Chatbot;