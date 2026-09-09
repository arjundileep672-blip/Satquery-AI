import React from 'react';
import { Satellite, Cpu, ShieldCheck, Activity } from 'lucide-react';
import type { HealthStatus } from '../types';

interface HeaderProps {
  health: HealthStatus | null;
  onOpenMetrics?: () => void;
}

export const Header: React.FC<HeaderProps> = ({ health, onOpenMetrics }) => {
  return (
    <header className="bg-slate-900 border-b border-slate-800 px-6 py-3.5 flex items-center justify-between shadow-md">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
          <Satellite className="w-5 h-5" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-bold text-slate-100 tracking-wide">
              SATQUERY <span className="text-emerald-400">AI</span>
            </h1>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 font-mono border border-slate-700">
              Phase 1 · SIH26167
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Interactive Vision-Language Assistant for Remote Sensing Image Analysis
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {onOpenMetrics && (
          <button
            type="button"
            onClick={onOpenMetrics}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 text-xs font-medium transition-all shadow-sm active:scale-95 cursor-pointer"
            title="Open Model Performance & Confusion Matrix"
          >
            <Activity className="w-3.5 h-3.5 text-emerald-400" />
            <span>Metrics & Matrix</span>
          </button>
        )}

        {health?.demo_mode && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-amber-500/10 border border-amber-500/30 text-amber-400 text-xs font-medium">
            <Cpu className="w-3.5 h-3.5" />
            <span>Demo Mode (Mock VLM)</span>
          </div>
        )}

        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-medium">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>Systems Operational</span>
        </div>
      </div>
    </header>
  );
};
