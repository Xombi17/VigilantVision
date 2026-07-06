"use client";

import { useState, useEffect } from "react";
import { Search, Download, ExternalLink, Calendar, Loader2, Image as ImageIcon } from "lucide-react";

interface AlertHistory {
  id: string;
  message: string;
  timestamp: string;
  image_path: string;
}

export default function HistoryPage() {
  const [history, setHistory] = useState<AlertHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedType, setSelectedType] = useState("All Event Types");

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/history`);
        if (response.ok) {
          const data = await response.json();
          setHistory(data);
        }
      } catch (err) {
        console.error("Failed to fetch history:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  const formatTime = (ts: string) => {
    if (!ts || ts.length !== 15) return ts;
    const year = ts.slice(0, 4);
    const month = ts.slice(4, 6);
    const day = ts.slice(6, 8);
    const hour = ts.slice(9, 11);
    const min = ts.slice(11, 13);
    const sec = ts.slice(13, 15);
    return `${year}-${month}-${day} ${hour}:${min}:${sec}`;
  };

  // Advanced client-side filtering matching logic
  const filteredHistory = history.filter((event) => {
    const matchesSearch = 
      event.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      event.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
      event.image_path.toLowerCase().includes(searchQuery.toLowerCase());

    let matchesType = true;
    if (selectedType === "Suspicious Behavior") {
      matchesType = event.message.includes("SUSPICION") || event.message.includes("RESTRICTED") || event.message.includes("CRIMINAL");
    } else if (selectedType === "Blacklisted Face") {
      matchesType = event.message.includes("BLACKLIST");
    } else if (selectedType === "Item Concealment") {
      matchesType = event.message.includes("THEFT") || event.message.includes("Concealed");
    }

    return matchesSearch && matchesType;
  });

  // Client-side CSV export trigger
  const handleExportCSV = () => {
    if (filteredHistory.length === 0) return;
    const headers = ["Event ID", "Timestamp", "Alert Message", "Evidence Path"];
    const rows = filteredHistory.map(event => [
      event.id,
      formatTime(event.timestamp),
      event.message,
      event.image_path
    ]);
    
    const csvContent = "data:text/csv;charset=utf-8,\uFEFF" 
      + [headers.join(","), ...rows.map(r => r.map(val => `"${val.replace(/"/g, '""')}"`).join(","))].join("\n");
      
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `VigilantVision_Alerts_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="max-w-6xl mx-auto pb-10">
      <header className="mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold tracking-tight mb-2">Alert History</h2>
          <p className="text-foreground/60">Review past security events and export evidence.</p>
        </div>
        <div className="flex gap-3">
          <button className="flex items-center gap-2 px-4 py-2 bg-glass border border-glass-border rounded-lg hover:bg-white/5 transition-colors text-sm font-medium">
            <Calendar className="w-4 h-4" />
            Last 7 Days
          </button>
          <button 
            onClick={handleExportCSV}
            disabled={filteredHistory.length === 0}
            className="flex items-center gap-2 px-4 py-2 bg-brand/20 border border-brand/35 hover:bg-brand/30 text-brand rounded-lg transition-colors text-sm font-bold cursor-pointer disabled:opacity-50"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
        </div>
      </header>

      <div className="glass-panel overflow-hidden">
        <div className="p-4 border-b border-glass-border flex gap-4">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-foreground/50" />
            <input 
              type="text" 
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Search by event ID, type, or camera..." 
              className="w-full bg-black/40 border border-glass-border rounded-lg pl-9 pr-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-brand"
            />
          </div>
          <select 
            value={selectedType}
            onChange={e => setSelectedType(e.target.value)}
            className="bg-black/40 border border-glass-border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-brand text-foreground"
          >
            <option className="bg-[#0f111a]">All Event Types</option>
            <option className="bg-[#0f111a]">Suspicious Behavior</option>
            <option className="bg-[#0f111a]">Blacklisted Face</option>
            <option className="bg-[#0f111a]">Item Concealment</option>
          </select>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-black/20 border-b border-glass-border text-sm text-foreground/70">
                <th className="p-4 font-medium">Event ID</th>
                <th className="p-4 font-medium">Date & Time</th>
                <th className="p-4 font-medium">Detection Type</th>
                <th className="p-4 font-medium">Image Path</th>
                <th className="p-4 font-medium text-center">Snapshot</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              {loading ? (
                <tr>
                  <td colSpan={5} className="p-8 text-center text-foreground/60">
                    <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
                    Loading history...
                  </td>
                </tr>
              ) : filteredHistory.length === 0 ? (
                <tr>
                  <td colSpan={5} className="p-8 text-center text-foreground/60">
                    No alert history found matching search criteria.
                  </td>
                </tr>
              ) : (
                filteredHistory.map((event) => (
                  <tr key={event.id} className="border-b border-glass-border/50 hover:bg-white/[0.02] transition-colors">
                    <td className="p-4 font-mono text-brand text-xs">{event.id.slice(0, 8)}...</td>
                    <td className="p-4 text-foreground/80">{formatTime(event.timestamp)}</td>
                    <td className="p-4">
                      <span className={`inline-block px-2 py-1 rounded text-xs font-semibold ${
                        event.message.includes('THEFT') || event.message.includes('CRIMINAL') ? 'bg-danger/20 text-danger border border-danger/20' : 
                        event.message.includes('BLACKLIST') || event.message.includes('RESTRICTED') ? 'bg-orange-500/20 text-orange-400 border border-orange-500/20' :
                        'bg-blue-500/20 text-blue-400 border border-blue-500/20'
                      }`}>
                        {event.message}
                      </span>
                    </td>
                    <td className="p-4 font-mono text-xs text-foreground/60">{event.image_path}</td>
                    <td className="p-4 text-center">
                      <a 
                        href={`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/${event.image_path}`} 
                        target="_blank" 
                        rel="noreferrer" 
                        className="p-1.5 rounded hover:bg-white/10 text-foreground/70 hover:text-brand transition-colors inline-block cursor-pointer"
                      >
                        <ImageIcon className="w-4 h-4" />
                      </a>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
        
        {!loading && filteredHistory.length > 0 && (
          <div className="p-4 border-t border-glass-border flex items-center justify-between text-sm text-foreground/60">
            <div>Showing {filteredHistory.length} of {history.length} entries</div>
            <div className="flex gap-1">
              <button className="px-3 py-1 border border-glass-border rounded hover:bg-white/5 disabled:opacity-50" disabled>Prev</button>
              <button className="px-3 py-1 bg-brand text-white rounded font-bold">1</button>
              <button className="px-3 py-1 border border-glass-border rounded hover:bg-white/5 disabled:opacity-50" disabled>Next</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
