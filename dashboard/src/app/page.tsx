"use client";

import { useState, useEffect } from "react";
import CameraGrid from "@/components/CameraGrid";
import { AlertTriangle, ShieldCheck } from "lucide-react";

interface AlertRecord {
  id: string;
  message: string;
  timestamp: string;
  image_path: string;
}

export default function Home() {
  const [recentAlerts, setRecentAlerts] = useState<AlertRecord[]>([]);

  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
        const res = await fetch(`${apiBaseUrl}/history`);
        if (res.ok) {
          const data: AlertRecord[] = await res.json();
          // Show most recent 12 alerts
          setRecentAlerts(data.slice(0, 12));
        }
      } catch {
        // Backend not running — that's fine, feed stays empty
      }
    };

    fetchAlerts();
    const interval = setInterval(fetchAlerts, 5000); // Poll slightly faster for real-time responsiveness
    return () => clearInterval(interval);
  }, []);

  const formatTimestamp = (ts: string) => {
    if (!ts || ts.length < 15) return ts;
    try {
      const year = ts.slice(0, 4);
      const month = ts.slice(4, 6);
      const day = ts.slice(6, 8);
      const hour = ts.slice(9, 11);
      const minute = ts.slice(11, 13);
      const second = ts.slice(13, 15);
      
      const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      const monthName = months[parseInt(month, 10) - 1] || month;
      
      return `${monthName} ${parseInt(day, 10)} · ${hour}:${minute}:${second}`;
    } catch {
      return ts;
    }
  };

  const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  return (
    <div className="flex flex-col lg:grid lg:grid-cols-4 gap-6 flex-1 min-h-0">
      {/* Surveillance Matrix - main area */}
      <div className="lg:col-span-3 flex flex-col glass-panel overflow-hidden min-h-[480px] lg:h-[calc(100vh-140px)]">
        <div className="glass-header px-5 py-3 flex items-center justify-between border-b border-glass-border">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-brand animate-pulse" />
            <span className="text-sm font-semibold tracking-wide uppercase text-brand">Surveillance Matrix</span>
          </div>
          <span className="text-[10px] text-foreground/40 font-mono tracking-wider">LIVE · AI-ANNOTATED</span>
        </div>
        <div className="flex-1 w-full min-h-0 bg-black/20">
          <CameraGrid />
        </div>
      </div>

      {/* Incidents Sidebar */}
      <div className="lg:col-span-1 flex flex-col glass-panel p-5 lg:h-[calc(100vh-140px)] min-h-[400px]">
        <h3 className="text-xs font-semibold text-foreground/60 uppercase tracking-widest mb-4 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-danger animate-pulse" />
          Recent Incidents
        </h3>

        <div className="flex-1 overflow-y-auto pr-1 space-y-3 min-h-0 custom-scrollbar">
          {recentAlerts.length === 0 ? (
            <div className="flex flex-col items-center justify-center text-center gap-3 text-foreground/35 text-xs py-12 h-full">
              <ShieldCheck className="w-8 h-8 text-brand/40" />
              <span>No incidents recorded yet.<br />System is monitoring.</span>
            </div>
          ) : (
            recentAlerts.map((alert) => {
              const hasAlertImage = alert.image_path;
              return (
                <div
                  key={alert.id}
                  className="flex gap-3 bg-danger/5 hover:bg-danger/10 border border-danger/15 rounded-xl p-3 transition-all duration-300 hover:scale-[1.01] hover:border-danger/30 group"
                >
                  {hasAlertImage ? (
                    <div className="relative w-16 h-12 rounded-lg overflow-hidden flex-shrink-0 border border-danger/20 bg-black/50">
                      <img
                        src={`${apiBaseUrl}/${alert.image_path}`}
                        alt="Incident"
                        className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                        onError={(e) => {
                          e.currentTarget.style.display = "none";
                        }}
                      />
                    </div>
                  ) : (
                    <div className="w-16 h-12 rounded-lg flex items-center justify-center bg-danger/10 text-danger border border-danger/15 flex-shrink-0">
                      <AlertTriangle className="w-4 h-4" />
                    </div>
                  )}
                  <div className="flex-1 min-w-0 flex flex-col justify-between">
                    <div>
                      <p className="text-xs font-bold text-danger leading-tight uppercase tracking-wider truncate">
                        {alert.message.includes(":") ? alert.message.split(":")[0] : "THEFT ALERT"}
                      </p>
                      <p className="text-[11px] text-foreground/80 mt-0.5 line-clamp-2">
                        {alert.message}
                      </p>
                    </div>
                    <span className="text-[9px] text-foreground/40 font-mono mt-1 block">
                      {formatTimestamp(alert.timestamp)}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
