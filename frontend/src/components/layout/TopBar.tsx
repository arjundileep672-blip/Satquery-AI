import React, { useEffect, useState } from 'react';
import { Activity, Cpu, Shield, UserRound } from 'lucide-react';
import type { HealthStatus } from '../../types';

const PLACEHOLDERS = [
  'Ask SatQuery anything about this imagery…',
  'Find all buildings',
  'What changed between 2024 and 2026?',
  'Show newly constructed roads',
  'Calculate vegetation loss',
];

interface TopBarProps {
  query: string;
  onQueryChange: (q: string) => void;
  onSubmit: () => void;
  hasImage: boolean;
  isLoading: boolean;
  health: HealthStatus | null;
  onOpenMetrics: () => void;
  onOpenSettings: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  query,
  onQueryChange,
  onSubmit,
  hasImage,
  isLoading,
  health,
  onOpenMetrics,
  onOpenSettings,
}) => {
  const [ph, setPh] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => setPh((p) => (p + 1) % PLACEHOLDERS.length), 3800);
    return () => window.clearInterval(id);
  }, []);

  const online = !!health && (health.status === 'healthy' || health.status === 'ok');

  return (
    <header className="h-12 shrink-0 sq-glass flex items-center gap-3 px-3">
      <div className="hidden md:block w-[168px] shrink-0">
        <div className="text-[11px] font-semibold tracking-[0.2em] text-slate-100">SATQUERY AI</div>
      </div>

      <form
        className="flex-1 max-w-3xl mx-auto"
        onSubmit={(e) => {
          e.preventDefault();
          onSubmit();
        }}
      >
        <input
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          disabled={!hasImage || isLoading}
          placeholder={PLACEHOLDERS[ph]}
          className="w-full h-8 bg-black/35 border border-white/10 px-3 text-[13px] text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-teal-300/50"
        />
      </form>

      <div className="flex items-center gap-2 shrink-0">
        <span className="hidden sm:flex items-center gap-1.5 px-2 h-7 border border-white/10 text-[10px] sq-mono text-slate-300">
          <span className={`w-1.5 h-1.5 rounded-full ${online ? 'bg-teal-400 sq-live' : 'bg-rose-400'}`} />
          {health?.offline_mode === false ? 'ONLINE' : 'OFFLINE'}
        </span>
        <span className="hidden md:flex items-center gap-1 px-2 h-7 border border-white/10 text-[10px] sq-mono text-slate-300">
          <Cpu className="w-3 h-3 text-teal-300" />
          {health?.gpu_available ? 'GPU READY' : 'CPU'}
        </span>
        <button
          type="button"
          onClick={onOpenMetrics}
          className="p-1.5 text-slate-400 hover:text-teal-200"
          title="Model metrics"
        >
          <Activity className="w-4 h-4" />
        </button>
        <button
          type="button"
          onClick={onOpenSettings}
          className="p-1.5 text-slate-400 hover:text-slate-100"
          title="Settings"
        >
          <UserRound className="w-4 h-4" />
        </button>
        <Shield className="w-3.5 h-3.5 text-slate-600 hidden lg:block" />
      </div>
    </header>
  );
};
