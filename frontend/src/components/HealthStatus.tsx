import React from 'react';
import { Cpu, Sparkles, RefreshCw } from 'lucide-react';
import type { HealthStatus as HealthStatusType } from '../types';

interface HealthStatusProps {
  health: HealthStatusType | null;
  isChecking: boolean;
  onRefresh: () => void;
}

export const HealthStatus: React.FC<HealthStatusProps> = ({
  health,
  isChecking,
  onRefresh,
}) => {
  const isOnline = !!health && (health.status === 'healthy' || health.status === 'ok');

  return (
    <div className="flex items-center gap-2 md:gap-3">
      {/* Backend Online / Offline Badge */}
      <div
        className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
          isOnline
            ? 'bg-emerald-950/60 border-emerald-500/40 text-emerald-300'
            : 'bg-red-950/60 border-red-500/40 text-red-300'
        }`}
      >
        <span
          className={`w-2 h-2 rounded-full ${
            isOnline ? 'bg-emerald-400 animate-pulse' : 'bg-red-500'
          }`}
        />
        <span className="font-semibold tracking-wide">
          {isOnline ? 'AI Backend Online' : 'AI Backend Offline'}
        </span>
      </div>

      {/* GPU Status Pill */}
      {isOnline && (
        <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs bg-slate-800/80 text-slate-300 border border-slate-700 font-mono">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span>{health.gpu_available ? 'GPU: Available' : 'GPU: Simulated (CPU)'}</span>
        </div>
      )}

      {/* Models Status Pill */}
      {isOnline && (
        <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs bg-slate-800/80 text-slate-300 border border-slate-700 font-mono">
          <Sparkles className="w-3.5 h-3.5 text-amber-400" />
          <span>Models: {health.models_status || 'Ready'}</span>
        </div>
      )}

      {/* Demo Mode Badge */}
      {health?.demo_mode && (
        <span className="hidden md:inline-block px-2 py-0.5 rounded text-[11px] font-mono bg-amber-500/10 border border-amber-500/30 text-amber-400">
          Demo Mode
        </span>
      )}

      {/* Refresh Button */}
      <button
        onClick={onRefresh}
        disabled={isChecking}
        title="Check Backend Health"
        className="p-1 rounded-md text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors disabled:opacity-50"
      >
        <RefreshCw className={`w-3.5 h-3.5 ${isChecking ? 'animate-spin text-cyan-400' : ''}`} />
      </button>
    </div>
  );
};
