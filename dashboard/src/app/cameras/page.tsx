"use client";

import { useState, useEffect, useRef } from "react";
import { Camera, Plus, Trash2, Edit, Loader2, AlertCircle, CheckCircle, X, RefreshCw, HelpCircle } from "lucide-react";

interface CameraData {
  id: string;
  name: string;
  source: string;
  status: "active" | "error";
  roi_points: number[][];
}

interface CameraFeed {
  camera_id: string;
  name: string;
  data: string;
}

export default function CamerasPage() {
  const [cameras, setCameras] = useState<CameraData[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<{ text: string; type: "success" | "error" } | null>(null);

  // Add Camera Form
  const [name, setName] = useState("");
  const [source, setSource] = useState("");

  // ROI Canvas Drawer Modal States
  const [selectedCam, setSelectedCam] = useState<CameraData | null>(null);
  const [roiPoints, setRoiPoints] = useState<number[][]>([]);
  const [activeFrameBase64, setActiveFrameBase64] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const fetchCameras = async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/cameras`);
      if (res.ok) {
        const data = await res.json();
        setCameras(data);
      }
    } catch (err) {
      console.error("Error fetching cameras:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCameras();
  }, []);

  // Listen to live camera frames when ROI Modal is open
  useEffect(() => {
    if (!selectedCam) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setActiveFrameBase64(null);
      return;
    }

    const wsUrl = apiBaseUrl.replace("http", "ws") + "/ws";
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "multi_frame" && payload.cameras) {
          const matched = payload.cameras.find((c: CameraFeed) => c.camera_id === selectedCam.id);
          if (matched) {
            setActiveFrameBase64(matched.data);
          }
        }
      } catch (err) {
        console.error("Error parsing WS in ROI modal", err);
      }
    };

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [selectedCam]);

  // Redraw Canvas when points or frames change
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const drawCanvas = () => {
      // 1. Draw live camera frame OR placeholder background
      if (activeFrameBase64) {
        const img = new Image();
        img.src = `data:image/jpeg;base64,${activeFrameBase64}`;
        img.onload = () => {
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          drawPolygon();
        };
      } else {
        // Dark placeholder
        ctx.fillStyle = "#151824";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
        ctx.font = "20px var(--font-geist-sans), sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Surveillance Feed Loading...", canvas.width / 2, canvas.height / 2 - 10);
        ctx.font = "14px var(--font-geist-sans), sans-serif";
        ctx.fillText("(Ensure backend is active & transmitting live stream)", canvas.width / 2, canvas.height / 2 + 20);
        
        drawPolygon();
      }
    };

    const drawPolygon = () => {
      if (roiPoints.length === 0) return;

      // Draw lines
      ctx.beginPath();
      ctx.moveTo(roiPoints[0][0], roiPoints[0][1]);
      for (let i = 1; i < roiPoints.length; i++) {
        ctx.lineTo(roiPoints[i][0], roiPoints[i][1]);
      }
      
      // Close path visually
      ctx.closePath();

      // Style polygon fill & border
      ctx.fillStyle = "rgba(59, 130, 246, 0.25)";
      ctx.fill();
      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = 3;
      ctx.stroke();

      // Draw anchor points
      roiPoints.forEach((pt, index) => {
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 6, 0, 2 * Math.PI);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        ctx.strokeStyle = "#3b82f6";
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Point labels
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 12px sans-serif";
        ctx.fillText((index + 1).toString(), pt[0] + 10, pt[1] - 5);
      });
    };

    drawCanvas();
  }, [roiPoints, activeFrameBase64, selectedCam]);

  const handleAddCamera = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !source) {
      setMessage({ text: "Please enter a name and source value.", type: "error" });
      return;
    }

    setSubmitting(true);
    setMessage(null);

    try {
      const res = await fetch(`${apiBaseUrl}/cameras`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, source }),
      });
      const data = await res.json();

      if (res.ok) {
        setMessage({ text: "Camera added successfully!", type: "success" });
        setName("");
        setSource("");
        await fetchCameras();
      } else {
        setMessage({ text: data.detail || "Failed to add camera.", type: "error" });
      }
    } catch (err) {
      console.error(err);
      setMessage({ text: "Connection error. Make sure backend is running.", type: "error" });
    } finally {
      setSubmitting(false);
    }
  };

  const handleDeleteCamera = async (id: string) => {
    if (!confirm("Are you sure you want to delete this camera stream?")) return;
    setMessage(null);

    try {
      const res = await fetch(`${apiBaseUrl}/cameras/${id}`, {
        method: "DELETE",
      });
      const data = await res.json();

      if (res.ok) {
        setMessage({ text: "Camera deleted successfully.", type: "success" });
        await fetchCameras();
      } else {
        setMessage({ text: data.detail || "Failed to delete camera.", type: "error" });
      }
    } catch (err) {
      console.error(err);
      setMessage({ text: "Connection error.", type: "error" });
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    // Scale standard mouse clicks exactly to 1280x720 video frames
    const x = Math.round((e.clientX - rect.left) * (canvas.width / rect.width));
    const y = Math.round((e.clientY - rect.top) * (canvas.height / rect.height));

    setRoiPoints(prev => [...prev, [x, y]]);
  };

  const clearRoiPoints = () => {
    setRoiPoints([]);
  };

  const openRoiModal = (cam: CameraData) => {
    setSelectedCam(cam);
    setRoiPoints(cam.roi_points || []);
  };

  const handleSaveRoi = async () => {
    if (!selectedCam) return;
    try {
      const res = await fetch(`${apiBaseUrl}/cameras/${selectedCam.id}/roi`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: roiPoints }),
      });
      const data = await res.json();

      if (res.ok && data.status === "success") {
        setMessage({ text: "Region of Interest (ROI) coordinates saved successfully!", type: "success" });
        setSelectedCam(null);
        await fetchCameras();
      } else {
        alert(data.detail || "Failed to save ROI.");
      }
    } catch (err) {
      console.error(err);
      alert("Network error while saving ROI.");
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <Loader2 className="w-8 h-8 animate-spin text-brand" />
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto pb-10">
      <header className="mb-8">
        <h2 className="text-3xl font-bold tracking-tight mb-2">Camera Setup & Configuration</h2>
        <p className="text-foreground/60">
          Manage dynamic USB index/RTSP streams and define camera-specific Regions of Interest (ROI) interactively.
        </p>
      </header>

      {message && (
        <div 
          className={`mb-6 p-4 rounded-lg flex items-center gap-3 border ${
            message.type === "success" 
              ? "bg-green-500/10 border-green-500/30 text-green-400" 
              : "bg-danger/10 border-danger/30 text-danger"
          }`}
        >
          {message.type === "success" ? <CheckCircle className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
          <span className="text-sm font-medium">{message.text}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Add Camera Form */}
        <div className="lg:col-span-1">
          <div className="glass-panel p-6 sticky top-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 rounded bg-brand/20 text-brand">
                <Camera className="w-5 h-5" />
              </div>
              <h3 className="text-xl font-semibold">Connect New Camera</h3>
            </div>

            <form onSubmit={handleAddCamera} className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground/80">Camera Name</label>
                <input 
                  type="text" 
                  value={name}
                  onChange={e => setName(e.target.value)}
                  placeholder="e.g. Checkout Desk A"
                  className="w-full bg-black/40 border border-glass-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand text-foreground placeholder:text-foreground/30"
                  required
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium text-foreground/80">Input Source</label>
                  <span className="text-[10px] text-foreground/45 flex items-center gap-1">
                    <HelpCircle className="w-3 h-3" />
                    Webcam index or RTSP url
                  </span>
                </div>
                <input 
                  type="text" 
                  value={source}
                  onChange={e => setSource(e.target.value)}
                  placeholder="e.g. 0 or rtsp://username:pwd@ip:port/h264"
                  className="w-full bg-black/40 border border-glass-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand text-foreground placeholder:text-foreground/30"
                  required
                />
              </div>

              <button 
                type="submit" 
                disabled={submitting}
                className="w-full mt-4 flex items-center justify-center gap-2 bg-brand hover:bg-brand/90 disabled:opacity-50 text-white py-2 rounded-lg font-medium transition-colors cursor-pointer"
              >
                {submitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
                {submitting ? "Connecting stream..." : "Add Camera Stream"}
              </button>
            </form>
          </div>
        </div>

        {/* Cameras List */}
        <div className="lg:col-span-2">
          <div className="glass-panel p-6">
            <div className="flex justify-between items-center mb-6">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded bg-purple-500/20 text-purple-400">
                  <Plus className="w-5 h-5" />
                </div>
                <h3 className="text-xl font-semibold">Active Camera Feeds</h3>
              </div>
              <button 
                onClick={fetchCameras}
                className="p-2 hover:bg-glass border border-glass-border rounded-lg text-foreground/60 hover:text-foreground transition-colors cursor-pointer"
                title="Refresh"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>

            {cameras.length === 0 ? (
              <div className="text-center p-12 text-foreground/40 border border-glass-border border-dashed rounded-lg bg-black/10">
                No active camera streams configured. Use the form to link a local webcam (0) or network RTSP stream.
              </div>
            ) : (
              <div className="space-y-4">
                {cameras.map(cam => (
                  <div 
                    key={cam.id}
                    className="glass-panel p-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border border-glass-border bg-black/20"
                  >
                    <div className="flex items-center gap-3">
                      <div 
                        className={`w-10 h-10 rounded-full flex items-center justify-center border ${
                          cam.status === "active" 
                            ? "bg-green-500/10 border-green-500/30 text-green-400" 
                            : "bg-danger/10 border-danger/30 text-danger"
                        }`}
                      >
                        <Camera className="w-5 h-5" />
                      </div>
                      <div>
                        <h4 className="font-semibold text-foreground flex items-center gap-2">
                          {cam.name}
                          <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full border ${
                            cam.status === "active"
                              ? "bg-green-500/10 border-green-500/20 text-green-400"
                              : "bg-danger/10 border-danger/20 text-danger"
                          }`}>
                            {cam.status === "active" ? "Active" : "Offline / Error"}
                          </span>
                        </h4>
                        <p className="text-xs text-foreground/50 mt-1 font-mono">Source: {cam.source}</p>
                        <p className="text-xs text-brand font-medium mt-1">
                          {cam.roi_points && cam.roi_points.length > 0 
                            ? `✓ Region of Interest (ROI) configured: ${cam.roi_points.length} points`
                            : "⚠️ No ROI configured. Entire screen monitored."}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 self-end md:self-auto">
                      <button
                        onClick={() => openRoiModal(cam)}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-brand/20 border border-brand/35 text-brand hover:bg-brand/30 rounded-lg text-xs font-semibold transition-colors cursor-pointer"
                      >
                        <Edit className="w-3.5 h-3.5" />
                        Configure ROI
                      </button>
                      <button
                        onClick={() => handleDeleteCamera(cam.id)}
                        className="p-2 text-foreground/50 hover:text-danger hover:bg-danger/10 rounded-lg transition-colors cursor-pointer"
                        title="Delete camera feed"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ROI Drawing Canvas Modal */}
      {selectedCam && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md transition-all duration-300">
          <div className="glass-panel w-full max-w-4xl overflow-hidden border border-glass-border shadow-2xl relative animate-in fade-in zoom-in-95 duration-200">
            {/* Modal Header */}
            <div className="glass-header px-6 py-4 flex justify-between items-center">
              <div>
                <h3 className="text-lg font-bold text-foreground">Interactive ROI Canvas Drawer</h3>
                <p className="text-xs text-foreground/60">{selectedCam.name} ({selectedCam.source})</p>
              </div>
              <button 
                onClick={() => setSelectedCam(null)}
                className="p-1.5 hover:bg-white/10 rounded-lg text-foreground/70 hover:text-foreground transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 space-y-4">
              <div className="p-3 bg-brand/10 border border-brand/20 text-brand rounded-lg text-xs flex items-start gap-2 leading-relaxed">
                <HelpCircle className="w-4 h-4 mt-0.5 shrink-0" />
                <span>
                  <strong>How to Draw ROI:</strong> Click on the live feed to place polygon points. The points will connect with lines in order. Place at least 3 points to define a loitering or restricted zone. When finished, click <strong>Save ROI coordinates</strong>.
                </span>
              </div>

              {/* Responsive 1280x720 Canvas */}
              <div className="aspect-video w-full bg-black/60 rounded-lg overflow-hidden border border-glass-border relative flex items-center justify-center">
                <canvas
                  ref={canvasRef}
                  width={1280}
                  height={720}
                  onClick={handleCanvasClick}
                  className="w-full h-auto aspect-video cursor-crosshair object-contain"
                />
              </div>

              <div className="flex justify-between items-center text-xs text-foreground/50 px-1">
                <span>Coordinates: [{roiPoints.map(p => `(${p[0]},${p[1]})`).join(", ")}]</span>
                <span>Points count: {roiPoints.length}</span>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="glass-header border-t border-b-0 px-6 py-4 flex justify-between items-center gap-4">
              <button 
                onClick={clearRoiPoints}
                className="px-4 py-2 border border-glass-border hover:bg-glass rounded-lg text-sm font-semibold transition-colors cursor-pointer text-foreground/80 hover:text-foreground"
              >
                Clear Points
              </button>
              
              <div className="flex items-center gap-2">
                <button 
                  onClick={() => setSelectedCam(null)}
                  className="px-4 py-2 border border-glass-border hover:bg-glass rounded-lg text-sm font-semibold transition-colors cursor-pointer text-foreground/80 hover:text-foreground"
                >
                  Cancel
                </button>
                <button 
                  onClick={handleSaveRoi}
                  className="px-6 py-2 bg-brand hover:bg-brand/90 text-white rounded-lg text-sm font-bold transition-colors cursor-pointer"
                >
                  Save ROI Coordinates
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
