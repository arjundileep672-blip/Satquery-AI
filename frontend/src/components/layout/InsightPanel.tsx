import React from 'react';
import { X } from 'lucide-react';
import type { AnalysisResult, HistoryEntry, LayerVisibilityState, NavView, ObjectChangeItem } from '../../types';
import { ResultPanel } from '../ResultPanel';
import { StatisticsPanel } from '../StatisticsPanel';
import { MultitemporalSummaryPanel } from '../MultitemporalSummaryPanel';
import { ModelInfo } from '../ModelInfo';
import { ChangeInspector } from './ChangeInspector';
import { ImageUploader } from '../ImageUploader';

interface InsightPanelProps {
  open: boolean;
  onClose: () => void;
  view: NavView;
  result: AnalysisResult | null;
  lastQuery: string;
  selectedObject: ObjectChangeItem | null;
  onSelectObject: (obj: ObjectChangeItem | null) => void;
  onHighlight: (filter: LayerVisibilityState['parameterFilter']) => void;
  history: HistoryEntry[];
  onRestoreHistory: (entry: HistoryEntry) => void;
  onOpenMetrics: () => void;
  modelsUsed: string[];
  uploadProps: React.ComponentProps<typeof ImageUploader>;
  demoMode: boolean;
}

function metricCards(result: AnalysisResult | null) {
  if (!result) return [];
  const cards: { key: string; value: string; label: string; filter: LayerVisibilityState['parameterFilter'] }[] = [];
  const stats = result.statistics || {};
  const report = result.multitemporal_report;

  if (stats.objects_detected != null || (result.detections && result.detections.length)) {
    cards.push({
      key: 'obj',
      value: String(stats.objects_detected ?? result.detections?.length ?? 0),
      label: 'OBJECTS',
      filter: 'all',
    });
  }
  if (stats.buildings_detected != null || (result.masks && result.masks.length)) {
    cards.push({
      key: 'bldg',
      value: String(stats.buildings_detected ?? result.masks?.length ?? 0),
      label: 'BUILDINGS',
      filter: 'buildings',
    });
  }
  if (stats.changed_buildings != null) {
    cards.push({
      key: 'chg',
      value: String(stats.changed_buildings),
      label: 'CHANGED BLDG',
      filter: 'buildings',
    });
  }
  if (stats.changed_percentage != null) {
    cards.push({
      key: 'area',
      value: `${stats.changed_percentage}%`,
      label: 'CHANGED AREA',
      filter: 'all',
    });
  }

  report?.parameters.forEach((p) => {
    if (p.percentage_change_text && p.percentage_change_text !== '—') {
      const cat = (['buildings', 'roads', 'vegetation', 'water', 'land_cover'] as const).includes(
        p.category as any
      )
        ? (p.category as LayerVisibilityState['parameterFilter'])
        : 'all';
      cards.push({
        key: p.parameter,
        value: p.percentage_change_text,
        label: p.parameter.toUpperCase(),
        filter: cat,
      });
    }
  });

  const seen = new Set<string>();
  return cards.filter((c) => {
    if (seen.has(c.label)) return false;
    seen.add(c.label);
    return true;
  }).slice(0, 6);
}

export const InsightPanel: React.FC<InsightPanelProps> = ({
  open,
  onClose,
  view,
  result,
  lastQuery,
  selectedObject,
  onSelectObject,
  onHighlight,
  history,
  onRestoreHistory,
  onOpenMetrics,
  modelsUsed,
  uploadProps,
  demoMode,
}) => {
  if (!open) return null;
  const metrics = metricCards(result);

  return (
    <aside className="w-full lg:w-[380px] h-full sq-glass flex flex-col shrink-0 border-l border-white/10">
      <div className="h-10 px-3 flex items-center justify-between border-b border-white/10">
        <div className="text-[10px] sq-mono tracking-[0.2em] text-teal-200">
          {view === 'history' ? 'RECENT ANALYSIS' : view === 'settings' ? 'AI PIPELINE' : 'AI ANALYSIS'}
        </div>
        <button type="button" onClick={onClose} className="p-1 text-slate-500 hover:text-slate-200 lg:hidden">
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {view === 'overview' || view === 'analyze' || view === 'compare' || view === 'layers' ? (
          <ImageUploader {...uploadProps} />
        ) : null}

        {view === 'history' && (
          <div className="space-y-1.5">
            {history.length === 0 && (
              <p className="text-[12px] text-slate-500">No completed analyses in this session.</p>
            )}
            {history.map((h) => (
              <button
                key={h.id}
                type="button"
                onClick={() => onRestoreHistory(h)}
                className="w-full text-left px-3 py-2 border border-white/10 hover:border-teal-300/40"
              >
                <div className="text-[12px] text-slate-200 truncate">“{h.query}”</div>
                <div className="text-[10px] sq-mono text-slate-500 mt-1 flex justify-between">
                  <span>{new Date(h.at).toLocaleString()}</span>
                  <span className="text-teal-300">COMPLETE</span>
                </div>
              </button>
            ))}
          </div>
        )}

        {view === 'settings' && (
          <ModelInfo modelsUsed={modelsUsed} onOpenMetrics={onOpenMetrics} />
        )}

        {result && view !== 'history' && (
          <>
            <div>
              <div className="text-[9px] sq-mono tracking-[0.16em] text-slate-500 mb-1">QUERY</div>
              <p className="text-[13px] text-slate-200 leading-snug">“{lastQuery}”</p>
            </div>
            <div>
              <div className="text-[9px] sq-mono tracking-[0.16em] text-slate-500 mb-1">SUMMARY</div>
              <p className="text-[12.5px] text-slate-300 leading-relaxed">
                {result.multitemporal_report?.executive_summary || result.answer}
              </p>
            </div>
            {metrics.length > 0 && (
              <div className={`grid ${demoMode ? 'grid-cols-1' : 'grid-cols-2'} gap-2`}>
                {metrics.map((m) => (
                  <button
                    key={m.key}
                    type="button"
                    onClick={() => onHighlight(m.filter)}
                    className="text-left px-2.5 py-2 border border-white/10 hover:border-teal-300/40"
                    title="Highlight corresponding map features"
                  >
                    <div className={`${demoMode ? 'text-3xl' : 'text-lg'} font-medium sq-mono text-teal-200`}>
                      {m.value}
                    </div>
                    <div className="text-[9px] sq-mono tracking-[0.12em] text-slate-500 mt-0.5">{m.label}</div>
                  </button>
                ))}
              </div>
            )}
            {selectedObject && (
              <ChangeInspector
                obj={selectedObject}
                onClose={() => onSelectObject(null)}
              />
            )}
            {!demoMode && <StatisticsPanel result={result} />}
            {!demoMode && result.multitemporal_report && (
              <MultitemporalSummaryPanel
                report={result.multitemporal_report}
                selectedObjectId={selectedObject?.object_id}
                onSelectObject={onSelectObject}
                onParameterSelect={(cat) => {
                  const mapped =
                    cat === 'built_up'
                      ? 'buildings'
                      : (['buildings', 'roads', 'vegetation', 'water', 'land_cover'] as const).includes(cat as any)
                      ? (cat as LayerVisibilityState['parameterFilter'])
                      : 'all';
                  onHighlight(mapped);
                }}
              />
            )}
            {!demoMode && <ResultPanel result={result} />}
            {!demoMode && view !== 'settings' && (
              <ModelInfo modelsUsed={modelsUsed} onOpenMetrics={onOpenMetrics} />
            )}
          </>
        )}
      </div>
    </aside>
  );
};
