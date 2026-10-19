import React from "react";
import { motion } from "framer-motion";
import { Activity, CheckCircle2, Circle } from "lucide-react";

interface VisualizerProps {
  updates: any[];
  isProcessing: boolean;
}

const NODES = ["planner", "financial_analysis", "news_analysis", "technical_analysis", "risk_analysis", "verifier", "synthesizer"];

export const AgentVisualizer: React.FC<VisualizerProps> = ({ updates, isProcessing }) => {
  const activeNode = updates.length > 0 ? updates[updates.length - 1].node : null;
  const completedNodes = new Set(updates.map(u => u.node));

  return (
    <div className="p-6 bg-slate-900 rounded-2xl border border-slate-800 shadow-xl">
      <div className="flex flex-col items-center gap-8">
        <div className="flex items-center justify-between w-full max-w-4xl relative">
          {/* Connection Line */}
          <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-slate-800 -z-10" />
          
          {NODES.map((node, idx) => {
            const isActive = activeNode === node;
            const isCompleted = completedNodes.has(node);
            
            return (
              <div key={node} className="flex flex-col items-center gap-3 relative">
                <motion.div 
                  animate={{ 
                    scale: isActive ? 1.2 : 1,
                    backgroundColor: isActive ? "#3b82f6" : (isCompleted ? "#10b981" : "#1e293b")
                  }}
                  className="w-10 h-10 rounded-full border-2 border-slate-700 flex items-center justify-center text-white shadow-lg"
                >
                  {isActive ? <Activity size={20} className="animate-pulse" /> : 
                   isCompleted ? <CheckCircle2 size={20} /> : <Circle size={20} />}
                </motion.div>
                <span className={`text-xs font-medium transition-colors ${isActive ? "text-blue-400" : (isCompleted ? "text-emerald-400" : "text-slate-500")}`}>
                  {node.replace("_", " ").toUpperCase()}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
