import { useEffect, useState, useCallback } from "react";

export type AgentUpdate = {
  type: "node_update" | "final_report" | "error";
  node?: string;
  update?: any;
  report?: string;
  verification?: any;
  message?: string;
  status: string;
};

export const useAgentSocket = (url: string) => {
  const [updates, setUpdates] = useState<AgentUpdate[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [finalReport, setFinalReport] = useState<string | null>(null);
  const [socket, setSocket] = useState<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    ws.onmessage = (event) => {
      const data: AgentUpdate = JSON.parse(event.data);
      if (data.type === "node_update") {
        setUpdates((prev) => [...prev, data]);
      } else if (data.type === "final_report") {
        setFinalReport(data.report || null);
        setIsProcessing(false);
      } else if (data.type === "error") {
        console.error("Agent Error:", data.message);
        setIsProcessing(false);
      }
    };
    setSocket(ws);
    return () => ws.close();
  }, [url]);

  const sendQuery = useCallback((query: string) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      setUpdates([]);
      setFinalReport(null);
      setIsProcessing(true);
      socket.send(query);
    }
  }, [socket]);

  return { updates, isProcessing, finalReport, sendQuery };
};
