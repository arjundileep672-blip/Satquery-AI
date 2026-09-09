import React from 'react';
import type { AnalysisResult } from '../../types';

interface DemoOverlayProps {
  result: AnalysisResult | null;
  query: string;
  onViewChanges: () => void;
  onExit: () => void;
}

export const DemoOverlay: React.FC<DemoOverlayProps> = ({ result, query, onViewChanges, onExit }) => {
  if (!result) return null;
  const stats = result.statistics || {};
  const params = result.multitemporal_report?.parameters || [];
  const dates = result.multitemporal_report?.dates || [];

  return (
    <div className="absolute top-3 left-3 z-20 sq-glass px-5 py-4 max-w-sm pointer-events-auto">
      <div className="text-[10px] sq-mono tracking-[0.22em] text-teal-300">SATQUERY AI</div>
      <div className="text-[13px] tracking-[0.12em] text-slate-100 mt-1">
        {result.multitemporal_report ? 'MULTITEMPORAL ANALYSIS' : (result.task || 'SATELLITE ANALYSIS').toUpperCase()}
      </div>
      {dates.length >= 2 && (
        <div className="text-[12px] sq-mono text-slate-400 mt-1">
          {dates[0]} → {dates[dates.length - 1]}
        </div>
      )}
      <p className="text-[11px] text-slate-400 mt-2 truncate">“{query}”</p>
      <div className="mt-3 space-y-1.5">
        {stats.objects_detected != null && <DemoRow label="OBJECTS" value={String(stats.objects_detected)} />}
        {stats.buildings_detected != null && <DemoRow label="BUILDINGS" value={String(stats.buildings_detected)} />}
        {stats.changed_buildings != null && <DemoRow label="CHANGED BUILDINGS" value={String(stats.changed_buildings)} />}
        {stats.changed_percentage != null && (
          <DemoRow label="CHANGED AREA" value={`${stats.changed_percentage}%`} />
        )}
        {params.slice(0, 4).map((p) => (
          <DemoRow key={p.parameter} label={p.parameter.toUpperCase()} value={p.percentage_change_text || '—'} />
        ))}
      </div>
      <div className="flex gap-2 mt-4">
        <button
          type="button"
          onClick={onViewChanges}
          className="flex-1 h-8 border border-teal-300/50 text-[11px] sq-mono text-teal-100 bg-teal-400/10"
        >
          VIEW CHANGES
        </button>
        <button type="button" onClick={onExit} className="h-8 px-3 border border-white/15 text-[11px] text-slate-400">
          EXIT
        </button>
      </div>
    </div>
  );
};

const DemoRow: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="flex justify-between gap-4 text-[12px]">
    <span className="sq-mono text-slate-500 tracking-wide">{label}</span>
    <span className="sq-mono text-teal-200">{value}</span>
  </div>
);
