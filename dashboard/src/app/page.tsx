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
          // Show most recent 8 alerts
          setRecentAlerts(data.slice(0, 8));
        }
      } catch {
        // Backend not running — that's fine, feed stays empty
      }
    };

    fetchAlerts();
    const interval = setInterval(fetchAlerts, 8000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Camera — full width, fixed height */}
      <div className="glass-panel overflow-hidden" style={{ height: "480px" }}>
        <div className="glass-header px-5 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-brand animate-pulse" />
            <span className="text-sm font-semibold tracking-wide">Surveillance Matrix</span>
          </div>
          <span className="text-xs text-foreground/40 font-mono">LIVE · AI-Annotated</span>
        </div>
        <div className="w-full" style={{ height: "calc(480px - 48px)" }}>
          <CameraGrid />
        </div>
      </div>

      {/* Real alerts strip — sourced from /history API, not fake */}
      <div className="glass-panel p-5">
        <h3 className="text-sm font-semibold text-foreground/60 uppercase tracking-widest mb-4 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-amber-500" />
          Recent Incidents
        </h3>
        {recentAlerts.length === 0 ? (
          <div className="flex items-center gap-3 text-foreground/40 text-sm py-2">
            <ShieldCheck className="w-5 h-5 text-brand" />
            No incidents recorded yet. System is monitoring.
          </div>
        ) : (
          <div className="flex gap-3 overflow-x-auto pb-1">
            {recentAlerts.map((alert) => (
              <div
                key={alert.id}
                className="flex-shrink-0 bg-danger/10 border border-danger/20 rounded-xl px-4 py-3 min-w-[200px] max-w-[240px]"
              >
                <p className="text-xs font-semibold text-danger truncate">{alert.message}</p>
                <p className="text-[10px] text-foreground/40 mt-1.5 font-mono">{alert.timestamp}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
