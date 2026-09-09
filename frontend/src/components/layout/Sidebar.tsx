import React from 'react';
import {
  LayoutGrid,
  ScanSearch,
  Columns2,
  Target,
  GitCompare,
  Layers,
  History,
  Settings,
  PanelLeftClose,
  PanelLeftOpen,
} from 'lucide-react';
import type { HealthStatus, NavView } from '../../types';

const NAV: { id: NavView; label: string; icon: React.ElementType }[] = [
  { id: 'overview', label: 'Overview', icon: LayoutGrid },
  { id: 'analyze', label: 'Analyze', icon: ScanSearch },
  { id: 'compare', label: 'Compare', icon: Columns2 },
  { id: 'detections', label: 'Detections', icon: Target },
  { id: 'change', label: 'Change Analysis', icon: GitCompare },
  { id: 'layers', label: 'Layers', icon: Layers },
  { id: 'history', label: 'History', icon: History },
  { id: 'settings', label: 'Settings', icon: Settings },
];

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  view: NavView;
  onChangeView: (v: NavView) => void;
  health: HealthStatus | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
  collapsed,
  onToggle,
  view,
  onChangeView,
  health,
}) => {
  const online = !!health && (health.status === 'healthy' || health.status === 'ok');
  const modelsReady = (health?.models_status || 'Ready').toLowerCase() !== 'degraded';

  return (
    <aside
      className={`h-full sq-glass flex flex-col shrink-0 transition-[width] duration-200 ${
        collapsed ? 'w-[56px]' : 'w-[216px]'
      }`}
    >
      <div className={`flex items-center ${collapsed ? 'justify-center px-1 py-3' : 'justify-between px-3 py-3'} border-b border-white/10`}>
        {!collapsed && (
          <div className="min-w-0">
            <div className="text-[12px] font-semibold tracking-[0.18em] text-slate-100">SATQUERY AI</div>
            <div className="text-[9px] sq-mono tracking-[0.16em] text-teal-300/80 mt-0.5">
              REMOTE SENSING INTELLIGENCE
            </div>
          </div>
        )}
        <button
          type="button"
          onClick={onToggle}
          className="p-1.5 text-slate-400 hover:text-slate-100"
          title={collapsed ? 'Expand navigation' : 'Collapse navigation'}
        >
          {collapsed ? <PanelLeftOpen className="w-4 h-4" /> : <PanelLeftClose className="w-4 h-4" />}
        </button>
      </div>

      <nav className="flex-1 py-2 px-1.5 space-y-0.5 overflow-y-auto">
        {NAV.map((item) => {
          const Icon = item.icon;
          const active = view === item.id;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onChangeView(item.id)}
              title={item.label}
              className={`w-full flex items-center gap-2.5 ${
                collapsed ? 'justify-center px-0' : 'px-2.5'
              } py-2 text-[12px] transition-colors ${
                active
                  ? 'bg-teal-400/10 text-teal-200 border-l-2 border-teal-300'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-white/5 border-l-2 border-transparent'
              }`}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {!collapsed && <span>{item.label}</span>}
            </button>
          );
        })}
      </nav>

      <div className={`border-t border-white/10 ${collapsed ? 'p-2' : 'p-3'}`}>
        {!collapsed && (
          <div className="text-[9px] sq-mono tracking-[0.16em] text-slate-500 mb-2">SYSTEM STATUS</div>
        )}
        <StatusRow collapsed={collapsed} label="AI ENGINE" ok={online} />
        <StatusRow collapsed={collapsed} label="VISION MODELS" ok={modelsReady && online} />
        <StatusRow collapsed={collapsed} label="OFFLINE MODE" ok={health?.offline_mode !== false} />
      </div>
    </aside>
  );
};

const StatusRow: React.FC<{ collapsed: boolean; label: string; ok: boolean }> = ({
  collapsed,
  label,
  ok,
}) => (
  <div className={`flex items-center ${collapsed ? 'justify-center' : 'justify-between'} py-1`}>
    <span className="flex items-center gap-2">
      <span className={`w-1.5 h-1.5 rounded-full ${ok ? 'bg-teal-400 sq-live' : 'bg-rose-400'}`} />
      {!collapsed && <span className="text-[10px] sq-mono text-slate-400">{label}</span>}
    </span>
    {!collapsed && (
      <span className={`text-[10px] sq-mono ${ok ? 'text-teal-300' : 'text-rose-300'}`}>
        {ok ? 'ONLINE' : 'OFFLINE'}
      </span>
    )}
  </div>
);
