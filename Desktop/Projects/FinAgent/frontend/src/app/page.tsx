import React, { useState } from "react";
import { useAgentSocket } from "../hooks/useAgentSocket";
import { AgentVisualizer } from "../components/visualizer/AgentVisualizer";
import { ReportViewer } from "../components/report/ReportViewer";
import { Send } from "lucide-react";

export default function Home() {
  const [query, setQuery] = useState("");
  const { updates, isProcessing, finalReport, sendQuery } = useAgentSocket("ws://localhost:8000/api/v1/analyze");
  const [verification, setVerification] = useState({ confidence: 0, conflicts: [] });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    // In a real app, you would extract verification from the socket updates
    setVerification({ confidence: 0.85, conflicts: [] }); 
    sendQuery(query);
  };

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900 p-4 md:p-12">
      <div className="max-w-6xl mx-auto space-y-12">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-5xl font-extrabold tracking-tight text-slate-900">
            FinAgent <span className="text-blue-600">Reasoning</span>
          </h1>
          <p className="text-slate-500 text-lg max-w-2xl mx-auto">
            Evidence-driven financial analysis powered by a multi-agent orchestration system.
          </p>
        </div>

        {/* Input Area */}
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto relative">
          <input 
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Analyze NVDA valuation relative to TSMC supply chain risks..."
            className="w-full p-5 pr-16 rounded-2xl border border-slate-200 bg-white shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none text-lg transition-all"
          />
          <button 
            type="submit"
            disabled={isProcessing}
            className="absolute right-3 top-1/2 -translate-y-1/2 p-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors disabled:bg-slate-300"
          >
            <Send size={20} />
          </button>
        </form>

        {/* Live Visualizer */}
        {isProcessing && (
          <div className="space-y-4 animate-in fade-in duration-500">
            <h3 className="text-center text-sm font-semibold text-slate-400 uppercase tracking-widest">Agent Reasoning Process</h3>
            <AgentVisualizer updates={updates} isProcessing={isProcessing} />
          </div>
        )}

        {/* Final Report */}
        {finalReport && (
          <ReportViewer report={finalReport} verification={verification} />
        )}
      </div>
    </main>
  );
}
