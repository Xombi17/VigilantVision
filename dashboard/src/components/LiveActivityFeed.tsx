"use client";

import { useEffect, useState } from "react";
import { ShieldAlert, UserCheck, AlertTriangle } from "lucide-react";

interface ActivityEvent {
  id: string;
  type: "theft" | "face_match" | "warning";
  title: string;
  timestamp: string;
  camera: string;
}

export default function LiveActivityFeed() {
  const [activities, setActivities] = useState<ActivityEvent[]>([]);

  useEffect(() => {
    // Mocking a live stream of data
    const generateEvent = (): ActivityEvent => {
      const types = ["theft", "face_match", "warning"] as const;
      const type = types[Math.floor(Math.random() * types.length)];
      
      const titles = {
        theft: "Suspicious Action Detected",
        face_match: "Known Entity Recognized",
        warning: "Motion in Restricted Zone"
      };

      const cameras = ["CAM-01 (Front Door)", "CAM-02 (Storage)", "CAM-03 (Aisle 4)", "CAM-04 (Back Exit)"];

      return {
        id: Math.random().toString(36).substr(2, 9),
        type,
        title: titles[type],
        timestamp: new Date().toLocaleTimeString(),
        camera: cameras[Math.floor(Math.random() * cameras.length)]
      };
    };

    // Initial load
    setActivities(Array.from({ length: 5 }, generateEvent));

    const interval = setInterval(() => {
      setActivities(prev => {
        const newActivities = [generateEvent(), ...prev];
        if (newActivities.length > 20) newActivities.pop();
        return newActivities;
      });
    }, 4500);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="glass-panel p-6 flex flex-col h-full max-h-[500px]">
      <h3 className="text-xl font-semibold mb-6 flex items-center gap-2 tracking-tight">
        <span className="relative flex h-3 w-3 mr-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand opacity-75"></span>
          <span className="relative inline-flex rounded-full h-3 w-3 bg-brand"></span>
        </span>
        Live Activity Feed
      </h3>
      
      <div className="flex-1 overflow-y-auto pr-2 space-y-4">
        {activities.map((activity) => (
          <div key={activity.id} className="flex gap-4 p-4 rounded-xl bg-black/20 border border-white/5 hover:bg-black/30 transition-colors">
            <div className={`mt-1 rounded-full p-2 h-max ${
              activity.type === 'theft' ? 'bg-danger/10 text-danger' :
              activity.type === 'face_match' ? 'bg-brand/10 text-brand' :
              'bg-amber-500/10 text-amber-500'
            }`}>
              {activity.type === 'theft' && <ShieldAlert className="w-5 h-5" />}
              {activity.type === 'face_match' && <UserCheck className="w-5 h-5" />}
              {activity.type === 'warning' && <AlertTriangle className="w-5 h-5" />}
            </div>
            
            <div className="flex-1">
              <div className="flex justify-between items-start mb-1">
                <h4 className="font-medium text-sm text-foreground/90">{activity.title}</h4>
                <span className="text-xs text-foreground/50">{activity.timestamp}</span>
              </div>
              <div className="text-xs text-foreground/60">Source: {activity.camera}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
