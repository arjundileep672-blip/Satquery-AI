import React from 'react';
import { BarChart3, Target, Building2, TrendingUp, Compass, ShieldCheck, Layers } from 'lucide-react';
import type { AnalysisResult, PatchLabel, PatchSceneClassification } from '../types';

// ── Class colour map (EuroSAT 10 classes) ─────────────────────────────────────
const CLASS_COLORS: Record<string, { bg: string; bar: string; text: string }> = {
  AnnualCrop:            { bg: 'bg-amber-950/40',   bar: 'bg-amber-500',   text: 'text-amber-300' },
  Forest:                { bg: 'bg-emerald-950/40', bar: 'bg-emerald-600', text: 'text-emerald-300' },
  HerbaceousVegetation:  { bg: 'bg-green-950/40',   bar: 'bg-green-400',   text: 'text-green-300' },
  Highway:               { bg: 'bg-slate-800/60',   bar: 'bg-slate-400',   text: 'text-slate-300' },
  Industrial:            { bg: 'bg-orange-950/40',  bar: 'bg-orange-500',  text: 'text-orange-300' },
  Pasture:               { bg: 'bg-lime-950/40',    bar: 'bg-lime-400',    text: 'text-lime-300' },
  PermanentCrop:         { bg: 'bg-yellow-950/40',  bar: 'bg-yellow-500',  text: 'text-yellow-300' },
  Residential:           { bg: 'bg-purple-950/40',  bar: 'bg-purple-400',  text: 'text-purple-300' },
  River:                 { bg: 'bg-cyan-950/40',    bar: 'bg-cyan-400',    text: 'text-cyan-300' },
  SeaLake:               { bg: 'bg-blue-950/40',    bar: 'bg-blue-500',    text: 'text-blue-300' },
};

interface StatisticsPanelProps {
  result: AnalysisResult | null;
}

export const StatisticsPanel: React.FC<StatisticsPanelProps> = ({ result }) => {
  if (!result) return null;

  const stats = result.statistics || {};
  const detections = result.detections || [];
  const masks = result.masks || [];

  // Extract real numbers provided by backend
  const objectsDetected =
    stats.objects_detected !== undefined
      ? stats.objects_detected
      : detections.length > 0
      ? detections.length
      : null;

  const buildingsDetected =
    stats.buildings_detected !== undefined
      ? stats.buildings_detected
      : masks.length > 0
      ? masks.length
      : null;

  const changedBuildings =
    stats.changed_buildings !== undefined
      ? stats.changed_buildings
      : null;

  const changedAreaPercentage =
    stats.changed_percentage !== undefined
      ? stats.changed_percentage
      : null;

  const changedAreaHectares =
    stats.changed_area_hectares !== undefined
      ? stats.changed_area_hectares
      : stats.total_hectares !== undefined
      ? stats.total_hectares
      : null;

  const changedAreaM2 =
    stats.changed_area_m2 !== undefined
      ? stats.changed_area_m2
      : stats.total_area_m2 !== undefined
      ? stats.total_area_m2
      : null;

  const meanConfidence =
    stats.mean_confidence !== undefined
      ? stats.mean_confidence
      : result.confidence !== null && result.confidence !== undefined
      ? result.confidence
      : null;

  const patchScene: PatchSceneClassification | null =
    stats.patch_scene_classification ?? null;
  const singleScene = stats.scene_classification ?? null;

  // If no stats present, return null
  const hasAnyStat =
    objectsDetected !== null ||
    buildingsDetected !== null ||
    changedBuildings !== null ||
    changedAreaPercentage !== null ||
    changedAreaHectares !== null ||
    meanConfidence !== null ||
    patchScene !== null ||
    singleScene !== null;

  if (!hasAnyStat) return null;

  return (
    <div className="sq-glass p-4 space-y-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-200">
          <BarChart3 className="w-4 h-4 text-emerald-400" />
          <span>Analytical Statistics</span>
        </div>
        <span className="text-[10px] font-mono text-slate-500">Derived Metrics</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {/* Objects Detected */}
        {objectsDetected !== null && (
          <div className="bg-slate-950/70 border border-slate-800/80 rounded-lg p-3 flex flex-col">
            <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
              <Target className="w-3.5 h-3.5 text-emerald-400" />
              <span>Objects Detected</span>
            </div>
            <div className="text-xl md:text-2xl font-bold text-slate-100 font-mono">
              {objectsDetected}
            </div>
          </div>
        )}

        {/* Buildings Detected */}
        {buildingsDetected !== null && (
          <div className="bg-slate-950/70 border border-slate-800/80 rounded-lg p-3 flex flex-col">
            <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
              <Building2 className="w-3.5 h-3.5 text-cyan-400" />
              <span>Buildings Detected</span>
            </div>
            <div className="text-xl md:text-2xl font-bold text-slate-100 font-mono">
              {buildingsDetected}
            </div>
          </div>
        )}

        {/* Changed Buildings */}
        {changedBuildings !== null && (
          <div className="bg-slate-950/70 border border-red-900/40 rounded-lg p-3 flex flex-col">
            <div className="flex items-center gap-1.5 text-red-400 text-xs mb-1">
              <TrendingUp className="w-3.5 h-3.5 text-red-400" />
              <span>Changed Buildings</span>
            </div>
            <div className="text-xl md:text-2xl font-bold text-red-300 font-mono">
              {changedBuildings}
            </div>
          </div>
        )}

        {/* Changed Area */}
        {changedAreaPercentage !== null && (
          <div className="bg-slate-950/70 border border-amber-900/40 rounded-lg p-3 flex flex-col">
            <div className="flex items-center gap-1.5 text-amber-400 text-xs mb-1">
              <Compass className="w-3.5 h-3.5 text-amber-400" />
              <span>Changed Area</span>
            </div>
            <div className="text-xl md:text-2xl font-bold text-amber-300 font-mono">
              {changedAreaPercentage}%
            </div>
            {changedAreaHectares && (
              <div className="text-[10px] text-slate-400 font-mono mt-0.5">
                {changedAreaHectares} ha ({changedAreaM2 ? `${Math.round(changedAreaM2).toLocaleString()} m²` : ''})
              </div>
            )}
          </div>
        )}

        {/* Confidence Score */}
        {meanConfidence !== null && (
          <div className="bg-slate-950/70 border border-slate-800/80 rounded-lg p-3 flex flex-col">
            <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
              <ShieldCheck className="w-3.5 h-3.5 text-purple-400" />
              <span>Model Confidence</span>
            </div>
            <div className="text-xl md:text-2xl font-bold text-purple-300 font-mono">
              {(meanConfidence * 100).toFixed(0)}%
            </div>
          </div>
        )}
      </div>

      {/* ── Patch-Based Scene Composition ──────────────────────────────────── */}
      {patchScene && patchScene.available && patchScene.labels.length > 0 && (
        <div className="border-t border-slate-800 pt-3 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-300">
              <Layers className="w-3.5 h-3.5 text-teal-400" />
              <span>Scene Composition</span>
              <span className="text-[10px] font-mono font-normal text-slate-500 normal-case">
                ({patchScene.total_tiles} tiles · {patchScene.tile_size}px · {patchScene.stride}px stride)
              </span>
            </div>
            <span className="text-[10px] font-mono text-slate-500">
              {patchScene.inference_ms.toFixed(0)} ms
            </span>
          </div>
          <div className="space-y-1.5">
            {patchScene.labels.filter((l) => l.coverage_pct >= 3).map((label: PatchLabel) => {
              const colors = CLASS_COLORS[label.class_name] ?? {
                bg: 'bg-slate-800/60',
                bar: 'bg-slate-500',
                text: 'text-slate-300',
              };
              return (
                <div
                  key={label.class_name}
                  className={`${colors.bg} rounded-lg px-3 py-2`}
                  title={`${label.class_name}: ${label.coverage_pct}% of tiles (mean conf ${(label.mean_confidence * 100).toFixed(1)}%)`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className={`text-xs font-semibold ${colors.text}`}>
                      {label.class_name}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-mono text-slate-400">
                        {label.tile_count} tiles
                      </span>
                      <span className={`text-xs font-bold font-mono ${colors.text}`}>
                        {label.coverage_pct}%
                      </span>
                    </div>
                  </div>
                  {/* Coverage progress bar */}
                  <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${colors.bar} rounded-full transition-all duration-500`}
                      style={{ width: `${Math.min(label.coverage_pct, 100)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          <p className="text-[10px] text-slate-500 font-mono">
            Classes &lt; 3% coverage suppressed for readability.
            EfficientNet-B0 trained on ESA Sentinel-2 EuroSAT.
          </p>
        </div>
      )}

      {/* ── Single-label fallback badge (small images) ──────────────────────── */}
      {!patchScene && singleScene && (
        <div className="border-t border-slate-800 pt-3">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Layers className="w-3.5 h-3.5 text-teal-400 shrink-0" />
            <span className="font-semibold text-slate-300">Scene Type:</span>
            <span className="font-mono text-teal-300">{singleScene.top_class}</span>
            <span className="ml-auto text-[10px] font-mono text-slate-500">
              {(singleScene.confidence * 100).toFixed(1)}% conf
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
