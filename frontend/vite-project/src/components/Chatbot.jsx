import React, { useState, useEffect, useRef } from 'react';
import { Bot, User, X, Send } from 'lucide-react';
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
        content: "👋 Assalam-o-Alaikum! Main aapki car price prediction mein madad karunga.\n\nAap mujhse kuch bhi pooch sakte hain jese:\n• Meri car ki price kya hogi?\n• Kyun meri car sasti hai?\n• Kis brand ki car zyada costly hai?\n\nShuru karne ke liye 'predict' likhein! 🚗"
      }]);
    }
  }, [isOpen, messages.length]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const data = await sendChatMessage(sessionId.current, userMessage);
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.reply,
        prediction: data.prediction
      }]);

      if (data.action === 'predict' && data.prediction?.success) {
        showToast('Price predicted successfully!', 'success');
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '❌ Sorry, network error. Please try again.'
      }]);
      showToast('Failed to send message', 'error');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-end justify-end p-4 z-50">
      <div className="bg-white rounded-t-2xl md:rounded-2xl shadow-2xl w-full md:w-[450px] h-[600px] flex flex-col">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4 rounded-t-2xl flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
              <Bot className="w-6 h-6" />
            </div>
            <div>
              <div className="font-bold">AI Assistant</div>
              <div className="text-xs opacity-90">Online • Urdu/English</div>
            </div>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-white/20 rounded-lg">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
                  <Bot className="w-5 h-5 text-white" />
                </div>
              )}
              
              <div className={`max-w-[75%] ${msg.role === 'user' ? 'order-1' : ''}`}>
                <div className={`p-3 rounded-2xl ${
                  msg.role === 'user' 
                    ? 'bg-blue-600 text-white rounded-tr-none' 
                    : 'bg-white text-gray-900 rounded-tl-none shadow-sm'
                }`}>
                  <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
                  
                  {msg.prediction && msg.prediction.success && (
                    <div className="mt-3 p-3 bg-gradient-to-br from-green-50 to-blue-50 rounded-lg border border-green-200">
                      <div className="text-2xl font-bold text-green-700 mb-1">
                        {msg.prediction.price_display.formatted}
                      </div>
                      <div className="text-xs text-gray-600">
                        Range: {msg.prediction.price_range.min_display.lacs} - {msg.prediction.price_range.max_display.lacs} Lacs
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {msg.role === 'user' && (
                <div className="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center flex-shrink-0">
                  <User className="w-5 h-5 text-gray-700" />
                </div>
              )}
            </div>
          ))}
          
          {loading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="bg-white p-3 rounded-2xl rounded-tl-none shadow-sm">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-4 border-t bg-white rounded-b-2xl">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
              placeholder="Type your message... (Urdu/English)"
              className="flex-1 px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
              disabled={loading}
            />
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          <div className="text-xs text-gray-500 mt-2 text-center">
            Ask in Urdu: "Meri car ki price batao" or English: "predict price"
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;