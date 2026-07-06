"use client";

import { useEffect, useRef, useCallback } from "react";
import { WifiOff, AlertTriangle } from "lucide-react";

interface AlertData {
  id: string;
  message: string;
  timestamp: string;
  camera_id: string;
  image_path: string;
}

interface CameraFrame {
  id: string;
  name: string;
  frame: string;
  status: string;
}

interface WsPayload {
  type: string;
  cameras: CameraFrame[];
  alert: AlertData | null;
  audio: string | null;
}

/**
 * High-performance camera grid.
 *
 * CRITICAL performance design:
 * - Frame data is NEVER stored in React state — doing so causes a full
 *   React reconciliation + commit + paint on every frame (~25fps = 25 re-renders/s),
 *   which is what caused the "laggy" / "motion weirdness" you saw.
 * - Instead we keep a Map of img refs and update .src directly on the DOM node.
 *   This is imperatively equivalent to canvas.drawImage() for JPEG streams.
 * - Camera list (names, ids) is stored in state because it rarely changes.
 * - Alert overlay is stored in state because it's infrequent.
 */
export default function CameraGrid() {
  // Holds camera metadata (id, name). Updated rarely, OK in state.
  const camerasRef = useRef<CameraFrame[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);

  // Map of camera id → <img> DOM element for direct frame injection
  const imgRefs = useRef<Map<string, HTMLImageElement>>(new Map());

  // Alert overlay state — infrequent updates, fine to use state
  const alertRef = useRef<{
    cam_id: string;
    message: string;
    until: number;
  } | null>(null);
  const alertDivRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const alertBannerRef = useRef<Map<string, HTMLDivElement>>(new Map());

  // Connection status indicator
  const statusDotRef = useRef<HTMLSpanElement>(null);
  const statusTextRef = useRef<HTMLSpanElement>(null);

  const playSiren = useCallback(() => {
    try {
      const Ctx = window.AudioContext || (window as any).webkitAudioContext;
      if (!Ctx) return;
      const ctx = new Ctx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const now = ctx.currentTime;
      osc.type = "sine";
      osc.frequency.setValueAtTime(580, now);
      osc.frequency.linearRampToValueAtTime(950, now + 0.35);
      osc.frequency.linearRampToValueAtTime(580, now + 0.7);
      osc.frequency.linearRampToValueAtTime(950, now + 1.05);
      osc.frequency.linearRampToValueAtTime(580, now + 1.4);
      osc.connect(gain);
      gain.connect(ctx.destination);
      gain.gain.setValueAtTime(0.18, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 1.4);
      osc.start(now);
      osc.stop(now + 1.4);
    } catch {}
  }, []);

  // Directly update alert overlay DOM nodes (no React re-render)
  const applyAlertState = useCallback((camId: string, active: boolean, message?: string) => {
    alertDivRefs.current.forEach((div, id) => {
      const isThis = id === camId && active;
      div.style.display = isThis ? "block" : "none";
    });
    alertBannerRef.current.forEach((div, id) => {
      const isThis = id === camId && active;
      div.style.display = isThis ? "flex" : "none";
      if (isThis && message) {
        div.textContent = message;
      }
    });
    // Border flash on camera tile
    imgRefs.current.forEach((img, id) => {
      const tile = img.closest(".cam-tile") as HTMLElement | null;
      if (tile) {
        if (id === camId && active) {
          tile.classList.add("alert-pulse", "ring-2", "ring-danger/80");
        } else {
          tile.classList.remove("alert-pulse", "ring-2", "ring-danger/80");
        }
      }
    });
  }, []);

  // Render camera tiles imperatively into the container
  const renderCameras = useCallback((cameras: CameraFrame[]) => {
    const container = containerRef.current;
    if (!container) return;

    // Dynamically adjust grid container class based on number of cameras
    const count = cameras.length;
    if (count === 1) {
      container.className = "flex-1 flex items-center justify-center p-4 min-h-0";
      container.style.gridTemplateColumns = "";
    } else {
      container.className = "flex-1 grid gap-4 p-4 min-h-0 items-center justify-center";
      if (count === 2) {
        container.style.gridTemplateColumns = "repeat(auto-fit, minmax(400px, 1fr))";
      } else if (count <= 4) {
        container.style.gridTemplateColumns = "repeat(auto-fit, minmax(320px, 1fr))";
      } else {
        container.style.gridTemplateColumns = "repeat(auto-fit, minmax(280px, 1fr))";
      }
    }

    const existing = new Set(imgRefs.current.keys());
    const incoming = new Set(cameras.map((c) => c.id));

    // Remove tiles for cameras that disappeared
    existing.forEach((id) => {
      if (!incoming.has(id)) {
        const tile = container.querySelector(`[data-cam="${id}"]`);
        if (tile) container.removeChild(tile);
        imgRefs.current.delete(id);
        alertDivRefs.current.delete(id);
        alertBannerRef.current.delete(id);
      }
    });

    cameras.forEach((cam) => {
      if (!imgRefs.current.has(cam.id)) {
        // Build tile DOM manually — no React needed
        const tile = document.createElement("div");
        tile.dataset.cam = cam.id;
        
        // Single camera uses h-full, multiple cameras use w-full with aspect-video to fit grid cells
        if (count === 1) {
          tile.className =
            "cam-tile relative rounded-2xl overflow-hidden border border-glass-border bg-black aspect-video h-full max-w-full mx-auto shadow-2xl transition-all duration-300 hover:border-brand/40";
        } else {
          tile.className =
            "cam-tile relative rounded-2xl overflow-hidden border border-glass-border bg-black aspect-video w-full max-w-full max-h-full mx-auto shadow-lg transition-all duration-300 hover:border-brand/40";
        }
        tile.style.minHeight = "0";
        tile.style.minWidth = "0";

        // Camera name header
        const header = document.createElement("div");
        header.className =
          "absolute top-0 left-0 right-0 z-10 glass-header px-3 py-1.5 flex items-center justify-between";
        header.innerHTML = `
          <div class="flex items-center gap-1.5">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-3.5 h-3.5 text-brand" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>
            <span class="text-xs font-semibold text-foreground/90">${cam.name}</span>
          </div>
          <span class="text-[10px] text-foreground/50 font-mono flex items-center gap-1 font-semibold">
            <span class="w-1.5 h-1.5 rounded-full bg-brand animate-pulse"></span>
            LIVE
          </span>
        `;
        tile.appendChild(header);

        // The image — this is the performance-critical element
        const img = document.createElement("img");
        img.alt = cam.name;
        img.decoding = "async";
        img.style.cssText =
          "width:100%;height:100%;object-fit:cover;display:block;";
        tile.appendChild(img);
        imgRefs.current.set(cam.id, img);

        // Alert border overlay
        const alertBorder = document.createElement("div");
        alertBorder.style.cssText =
          "display:none;position:absolute;inset:0;border:3px solid rgba(244,63,94,0.8);pointer-events:none;border-radius:1rem;z-index:20;";
        tile.appendChild(alertBorder);
        alertDivRefs.current.set(cam.id, alertBorder);

        // Alert message banner
        const alertBanner = document.createElement("div");
        alertBanner.style.cssText =
          "display:none;position:absolute;bottom:0;left:0;right:0;z-index:30;background:rgba(244,63,94,0.9);color:#fff;font-size:11px;font-weight:700;padding:8px 12px;align-items:center;gap:6px;backdrop-filter:blur(4px);";
        tile.appendChild(alertBanner);
        alertBannerRef.current.set(cam.id, alertBanner);

        container.appendChild(tile);
      } else {
        // If the camera count changed, update class on existing tiles
        const tile = container.querySelector(`[data-cam="${cam.id}"]`);
        if (tile) {
          if (count === 1) {
            tile.className =
              "cam-tile relative rounded-2xl overflow-hidden border border-glass-border bg-black aspect-video h-full max-w-full mx-auto shadow-2xl transition-all duration-300 hover:border-brand/40";
          } else {
            tile.className =
              "cam-tile relative rounded-2xl overflow-hidden border border-glass-border bg-black aspect-video w-full max-w-full max-h-full mx-auto shadow-lg transition-all duration-300 hover:border-brand/40";
          }
        }
      }
    });
  }, []);

  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimeout: ReturnType<typeof setTimeout>;
    let alertClearTimeout: ReturnType<typeof setTimeout>;

    const setConnected = (ok: boolean) => {
      if (statusDotRef.current) {
        statusDotRef.current.style.backgroundColor = ok ? "#14b8a6" : "#f43f5e";
      }
      if (statusTextRef.current) {
        statusTextRef.current.textContent = ok
          ? "Connected"
          : "Disconnected — retrying…";
      }
    };

    const connect = () => {
      const wsUrl = process.env.NEXT_PUBLIC_API_URL
        ? process.env.NEXT_PUBLIC_API_URL.replace("http", "ws") + "/ws"
        : "ws://localhost:8000/ws";

      ws = new WebSocket(wsUrl);

      ws.onopen = () => setConnected(true);

      ws.onclose = () => {
        setConnected(false);
        reconnectTimeout = setTimeout(connect, 3000);
      };

      ws.onerror = () => ws.close();

      ws.onmessage = (event: MessageEvent) => {
        // Parse off the main thread via JSON.parse (fast for these payloads)
        let payload: WsPayload;
        try {
          payload = JSON.parse(event.data as string);
        } catch {
          return;
        }

        if (payload.type !== "multi_frame") return;

        const cameras = payload.cameras ?? [];

        // Render / update camera tiles (creates DOM nodes if new camera appeared)
        renderCameras(cameras);

        // Inject frames directly into img.src — bypasses React entirely
        cameras.forEach((cam) => {
          const img = imgRefs.current.get(cam.id);
          if (img && cam.frame) {
            img.src = `data:image/jpeg;base64,${cam.frame}`;
          }
        });

        // Handle alert overlay
        if (payload.alert) {
          const { camera_id, message } = payload.alert;
          applyAlertState(camera_id, true, message);
          playSiren();

          clearTimeout(alertClearTimeout);
          alertClearTimeout = setTimeout(() => {
            applyAlertState(camera_id, false);
          }, 4000);
        }
      };
    };

    connect();

    return () => {
      clearTimeout(reconnectTimeout);
      clearTimeout(alertClearTimeout);
      ws?.close();
    };
  }, [renderCameras, applyAlertState, playSiren]);

  return (
    <div className="flex flex-col h-full" style={{ minHeight: "inherit" }}>
      {/* Status bar */}
      <div className="flex items-center gap-2 px-4 py-2 text-xs text-foreground/50">
        <span
          ref={statusDotRef}
          className="w-2 h-2 rounded-full transition-colors"
          style={{ backgroundColor: "#f43f5e" }}
        />
        <span ref={statusTextRef}>Disconnected — retrying…</span>
      </div>

      {/* Camera grid — fills remaining space */}
      <div
        ref={containerRef}
        className="flex-1 flex flex-wrap gap-2 p-2"
        style={{ minHeight: 0 }}
      />
    </div>
  );
}
