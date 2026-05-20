import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, 
  Bot, 
  User, 
  Database, 
  Settings, 
  Search, 
  Activity, 
  BookOpen,
  ChevronRight,
  MessageSquare,
  Stethoscope,
  Info
} from 'lucide-react';

/**
 * Thành phần Item điều hướng Sidebar
 */
const NavItem = ({ icon, label, active, onClick }) => (
  <button 
    onClick={onClick}
    className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${
      active 
        ? 'bg-blue-50 text-blue-600' 
        : 'text-slate-500 hover:bg-slate-50 hover:text-slate-800'
    }`}
  >
    <div className={active ? 'text-blue-600' : 'text-slate-400'}>
      {icon}
    </div>
    <span className="text-sm font-semibold hidden lg:block">{label}</span>
    {active && <ChevronRight size={16} className="ml-auto hidden lg:block" />}
  </button>
);

function App() {
  const [messages, setMessages] = useState([
    { 
      id: 1, 
      role: 'assistant', 
      content: 'Xin chào! Tôi là trợ lý AI y tế được huấn luyện bằng QLoRA. Tôi có thể giúp gì cho bạn hôm nay?',
      sources: []
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const [retrievedDocs, setRetrievedDocs] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState(null);

  // Function to export metrics to CSV
  const exportToCSV = () => {
    if (!metrics || metrics.length === 0) return;

    const headers = ['Config', 'RAG', 'BLEU-4', 'ROUGE-L', 'BERTScore F1', 'Recall@5', 'Samples'];
    const csvContent = [
      headers.join(','),
      ...metrics.map(item => [
        `"${item.config}"`,
        item.use_rag ? 'Có' : 'Không',
        item.bleu_4,
        item.rouge_l,
        item.bertscore_f1,
        item.recall_at_5,
        item.n_samples
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', 'metrics_summary.csv');
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  const scrollRef = useRef(null);
  const mockSources = [];

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { id: Date.now(), role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: input }),
      });

      if (!response.ok) {
        throw new Error('API request failed');
      }

      const data = await response.json();
      
      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.answer,
        sources: data.sources
      };
      setMessages(prev => [...prev, assistantMessage]);
      setRetrievedDocs(data.sources || []);
    } catch (error) {
      console.error('Error calling API:', error);
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Xin lỗi, có lỗi xảy ra khi kết nối với hệ thống. Vui lòng thử lại sau.',
        sources: []
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  useEffect(() => {
    if (activeTab === 'stats' && metrics === null && !metricsLoading) {
      fetchMetrics();
    }
  }, [activeTab]);

  const fetchMetrics = async () => {
    setMetricsLoading(true);
    setMetricsError(null);
    try {
      const response = await fetch('http://localhost:8000/metrics_summary');
      if (!response.ok) {
        throw new Error('Không tìm thấy metrics summary');
      }
      const data = await response.json();
      setMetrics(data.metrics || []);
    } catch (error) {
      console.error('Error fetching metrics:', error);
      setMetricsError(error.message || 'Lỗi khi tải metrics');
    } finally {
      setMetricsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-50 font-sans text-slate-900 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-20 lg:w-64 bg-white border-r border-slate-200 flex flex-col transition-all duration-300 z-20">
        <div className="p-6 flex items-center gap-3">
          <div className="bg-blue-600 p-2 rounded-xl text-white">
            <Stethoscope size={24} />
          </div>
          <span className="font-bold text-xl hidden lg:block tracking-tight text-blue-900">MediAI Lab</span>
        </div>

        <nav className="flex-1 px-4 space-y-2 mt-4">
          <NavItem 
            icon={<MessageSquare size={20} />} 
            label="Hội thoại" 
            active={activeTab === 'chat'} 
            onClick={() => setActiveTab('chat')} 
          />
          <NavItem 
            icon={<Database size={20} />} 
            label="Cơ sở tri thức (RAG)" 
            active={activeTab === 'knowledge'} 
            onClick={() => setActiveTab('knowledge')} 
          />
          <NavItem 
            icon={<Activity size={20} />} 
            label="Hiệu năng Model" 
            active={activeTab === 'stats'} 
            onClick={() => setActiveTab('stats')} 
          />
        </nav>

        <div className="p-4 border-t border-slate-100">
          <div className="bg-blue-50 rounded-lg p-3 hidden lg:block">
            <div className="flex items-center gap-2 mb-1">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
              <span className="text-xs font-semibold text-blue-700 uppercase tracking-wider">Model: Online</span>
            </div>
            <p className="text-[10px] text-blue-600 opacity-80 italic font-medium">Fine-tuned w/ QLoRA</p>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative bg-slate-50">
        <header className="h-16 bg-white/80 backdrop-blur-md border-b border-slate-200 px-8 flex items-center justify-between sticky top-0 z-10">
          <div>
            <h1 className="font-semibold text-slate-800">Trợ lý AI Y tế</h1>
            <p className="text-xs text-slate-500">Qwen-2.5 + FAISS RAG System</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white font-bold text-xs shadow-sm">
              AD
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        {activeTab === 'chat' && (
          <>
            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto p-4 lg:p-8 space-y-6" ref={scrollRef}>
              {messages.map((msg) => (
                <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] lg:max-w-[70%] flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 shadow-sm ${
                      msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white border border-slate-200 text-blue-600'
                    }`}>
                      {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                    </div>
                    <div className="space-y-2">
                      <div className={`p-4 rounded-2xl shadow-sm leading-relaxed ${
                        msg.role === 'user' 
                          ? 'bg-blue-600 text-white rounded-tr-none' 
                          : 'bg-white border border-slate-200 text-slate-800 rounded-tl-none'
                      }`}>
                        {msg.content}
                      </div>
                      
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="flex flex-wrap gap-2 mt-2">
                          {msg.sources.map((source, idx) => (
                            <div key={idx} className="flex items-center gap-2 bg-white border border-slate-200 rounded-lg px-3 py-1.5 text-[11px] text-slate-600 hover:border-blue-300 hover:bg-blue-50 transition-all shadow-sm">
                              <BookOpen size={12} className="text-blue-500" />
                              <span className="font-semibold">{source.disease}</span>
                              <span className="opacity-40 font-mono">|</span>
                              <span className="opacity-70">{source.section}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="flex justify-start pl-11">
                   <div className="flex gap-1">
                      <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce"></span>
                      <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:0.2s]"></span>
                      <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:0.4s]"></span>
                   </div>
                </div>
              )}
            </div>

            {/* Input area */}
            <div className="p-4 lg:p-8 bg-gradient-to-t from-slate-50 to-transparent">
              <div className="max-w-4xl mx-auto relative">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
                  placeholder="Hỏi về triệu chứng bệnh..."
                  className="w-full bg-white border border-slate-200 rounded-2xl py-4 pl-6 pr-16 shadow-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all resize-none min-h-[60px]"
                  rows="1"
                />
                <button 
                  onClick={handleSend}
                  disabled={!input.trim()}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-40 shadow-md"
                >
                  <Send size={20} />
                </button>
              </div>
            </div>
          </>
        )}

        {activeTab === 'knowledge' && (
          <div className="flex-1 overflow-y-auto p-4 lg:p-8">
            <div className="max-w-4xl mx-auto">
              <div className="mb-6">
                <h2 className="text-2xl font-bold text-slate-800 mb-2">Tài liệu Truy vấn (RAG)</h2>
                <p className="text-slate-600">Danh sách các tài liệu được lấy từ Knowledge Base cho câu hỏi gần nhất</p>
              </div>

              {retrievedDocs.length === 0 ? (
                <div className="bg-slate-50 border border-slate-200 rounded-lg p-8 text-center">
                  <Database size={32} className="mx-auto text-slate-400 mb-2" />
                  <p className="text-slate-500">Hãy hỏi một câu hỏi để xem các tài liệu được truy vấn</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {retrievedDocs.map((doc, idx) => (
                    <div key={idx} className="bg-white border border-slate-200 rounded-lg p-4 lg:p-6 hover:border-blue-300 hover:shadow-md transition-all">
                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 mb-3 text-xs font-semibold">
                        <div>
                          <span className="text-slate-500">Bệnh</span>
                          <p className="text-blue-600">{doc.disease || 'N/A'}</p>
                        </div>
                        <div>
                          <span className="text-slate-500">Mục</span>
                          <p className="text-blue-600">{doc.section || 'N/A'}</p>
                        </div>
                        <div className="col-span-2 lg:col-span-2">
                          <span className="text-slate-500">Nguồn</span>
                          <p className="text-blue-600 truncate">{doc.source || 'N/A'}</p>
                        </div>
                      </div>
                      
                      <div className="bg-slate-50 rounded p-3 mb-3 max-h-[120px] overflow-y-auto text-sm text-slate-700 leading-relaxed">
                        {doc.content || doc.page_content || 'Không có nội dung'}
                      </div>

                      {doc.url && (
                        <a 
                          href={doc.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 font-semibold"
                        >
                          <BookOpen size={12} />
                          Xem nguồn →
                        </a>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'stats' && (
          <div className="flex-1 overflow-y-auto p-4 lg:p-8">
            <div className="max-w-5xl mx-auto">
              <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-slate-800 mb-2">Đánh giá mô hình</h2>
                  <p className="text-slate-600">Hiển thị kết quả experiment 4 cấu hình, bao gồm BLEU, ROUGE-L, BERTScore và Recall@5.</p>
                </div>
                <div className="flex gap-3">
                  <button
                    type="button"
                    onClick={fetchMetrics}
                    className="inline-flex items-center justify-center rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 transition"
                  >
                    Làm mới dữ liệu
                  </button>
                  <button
                    type="button"
                    onClick={exportToCSV}
                    disabled={!metrics || metrics.length === 0}
                    className="inline-flex items-center justify-center rounded-xl bg-green-600 px-4 py-2 text-sm font-semibold text-white hover:bg-green-700 disabled:bg-slate-300 disabled:cursor-not-allowed transition"
                  >
                    Export CSV
                  </button>
                </div>
              </div>

              {metricsLoading ? (
                <div className="rounded-xl border border-slate-200 bg-white p-8 text-center text-slate-500">Đang tải metrics...</div>
              ) : metricsError ? (
                <div className="rounded-xl border border-rose-200 bg-rose-50 p-6 text-rose-700">{metricsError}</div>
              ) : !metrics || metrics.length === 0 ? (
                <div className="rounded-xl border border-slate-200 bg-white p-8 text-center text-slate-500">
                  Chưa có dữ liệu đánh giá. Hãy chạy experiment và lưu file `evaluation/metrics_summary.json`.
                </div>
              ) : (
                <div className="space-y-6">
                  <div className="overflow-x-auto rounded-3xl border border-slate-200 bg-white p-4 shadow-sm">
                    <table className="min-w-full text-left text-sm text-slate-700">
                      <thead>
                        <tr className="border-b border-slate-200 text-slate-500">
                          <th className="p-3">Config</th>
                          <th className="p-3">RAG</th>
                          <th className="p-3">BLEU-4</th>
                          <th className="p-3">ROUGE-L</th>
                          <th className="p-3">BERTScore F1</th>
                          <th className="p-3">Recall@5</th>
                          <th className="p-3">Samples</th>
                        </tr>
                      </thead>
                      <tbody>
                        {metrics.map((item, idx) => (
                          <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                            <td className="p-3 font-medium text-slate-800">{item.config}</td>
                            <td className="p-3">{item.use_rag ? 'Có' : 'Không'}</td>
                            <td className="p-3">{item.bleu_4}</td>
                            <td className="p-3">{item.rouge_l}</td>
                            <td className="p-3">{item.bertscore_f1}</td>
                            <td className="p-3">{item.recall_at_5}</td>
                            <td className="p-3">{item.n_samples}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="grid gap-4 lg:grid-cols-2">
                    {['bleu_4', 'rouge_l', 'bertscore_f1'].map((metric) => {
                      const best = metrics.reduce((bestItem, current) => {
                        if (!bestItem) return current;
                        return (current[metric] || 0) > (bestItem[metric] || 0) ? current : bestItem;
                      }, null);
                      return (
                        <div key={metric} className="rounded-3xl border border-slate-200 bg-slate-50 p-5 shadow-sm">
                          <p className="text-sm text-slate-500 uppercase tracking-[0.2em]">Tốt nhất theo {metric.replace('_', '.')}</p>
                          <p className="mt-2 text-lg font-semibold text-slate-900">{best?.config || 'N/A'}</p>
                          <p className="mt-1 text-slate-600">Giá trị: {best ? best[metric] : 'N/A'}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;