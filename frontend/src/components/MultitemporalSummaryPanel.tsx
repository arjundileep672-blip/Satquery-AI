import React, { useState } from 'react';
import {
  Calendar,
  Building2,
  GitCommit,
  TrendingUp,
  Layers,
  Droplets,
  Trees,
  MapPin,
  ShieldAlert,
} from 'lucide-react';
import type {
  MultitemporalReport,
  ObjectChangeItem,
} from '../types';

interface MultitemporalSummaryPanelProps {
  report: MultitemporalReport;
  selectedObjectId?: string | null;
  onSelectObject?: (obj: ObjectChangeItem | null) => void;
  onParameterSelect?: (category: string) => void;
}

export const MultitemporalSummaryPanel: React.FC<MultitemporalSummaryPanelProps> = ({
  report,
  selectedObjectId,
  onSelectObject,
  onParameterSelect,
}) => {
  const [activeTab, setActiveTab] = useState<'parameters' | 'objects' | 'transitions' | 'ranking'>(
    'parameters'
  );
  const [objectFilter, setObjectFilter] = useState<string>('ALL');

  const {
    executive_summary,
    dates,
    registration_quality,
    parameters,
    object_changes,
    transition_matrix,
    change_ranking,
    limitations_and_disclaimers,
    spatial_summary,
  } = report;

  const date1 = dates[0] || 'Date 1 (Baseline)';
  const date2 = dates[dates.length - 1] || 'Date 2 (Current)';

  const filteredObjects = object_changes.filter((obj) => {
    if (objectFilter === 'ALL') return true;
    return obj.status.toUpperCase() === objectFilter.toUpperCase();
  });

  const getStatusColor = (status: string) => {
    switch (status.toUpperCase()) {
      case 'NEW':
        return 'text-emerald-400 bg-emerald-950/60 border-emerald-800/80';
      case 'REMOVED':
        return 'text-rose-400 bg-rose-950/60 border-rose-800/80';
      case 'EXPANDED':
        return 'text-amber-400 bg-amber-950/60 border-amber-800/80';
      case 'CONTRACTED':
        return 'text-cyan-400 bg-cyan-950/60 border-cyan-800/80';
      case 'INCREASED':
        return 'text-emerald-400';
      case 'DECREASED':
        return 'text-rose-400';
      default:
        return 'text-slate-400 bg-slate-800 border-slate-700';
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'built_up':
        return <Building2 className="w-3.5 h-3.5 text-amber-400" />;
      case 'roads':
        return <GitCommit className="w-3.5 h-3.5 text-cyan-400" />;
      case 'vegetation':
        return <Trees className="w-3.5 h-3.5 text-emerald-400" />;
      case 'water':
        return <Droplets className="w-3.5 h-3.5 text-blue-400" />;
      default:
        return <Layers className="w-3.5 h-3.5 text-purple-400" />;
    }
  };

  return (
    <div className="sq-glass overflow-hidden space-y-4">
      {/* ── Header: Multitemporal Analysis Title & Dates ──────────────────────── */}
      <div className="bg-slate-950/80 px-5 py-4 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Calendar className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
              Multitemporal Change Parameter Analysis
            </h2>
            <div className="flex items-center gap-2 text-xs font-mono text-slate-400 mt-0.5">
              <span className="text-slate-300 font-semibold">{date1}</span>
              <span className="text-cyan-400">→</span>
              <span className="text-emerald-400 font-semibold">{date2}</span>
            </div>
          </div>
        </div>

        {/* Alignment Quality Badge */}
        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-500 font-mono text-[11px]">Alignment:</span>
          <span
            className={`px-2.5 py-1 rounded-full text-[11px] font-mono border ${
              registration_quality?.status === 'EXCELLENT'
                ? 'bg-emerald-950/60 text-emerald-300 border-emerald-800'
                : registration_quality?.status === 'ADEQUATE'
                ? 'bg-amber-950/60 text-amber-300 border-amber-800'
                : 'bg-rose-950/60 text-rose-300 border-rose-800'
            }`}
          >
            {registration_quality?.status || 'UNALIGNED'}
            {registration_quality?.inliers ? ` (${registration_quality.inliers} inliers)` : ''}
          </span>
        </div>
      </div>

      <div className="px-5 space-y-4">
        {/* ── Executive Summary ─────────────────────────────────────────────────── */}
        <div className="bg-cyan-950/20 border border-cyan-800/40 rounded-xl p-3.5 space-y-1.5">
          <div className="flex items-center gap-2 text-xs font-semibold text-cyan-400 uppercase tracking-wider">
            <TrendingUp className="w-4 h-4" />
            <span>Executive Change Summary</span>
          </div>
          <p className="text-xs text-slate-200 leading-relaxed font-normal">
            {executive_summary}
          </p>
        </div>

        {/* ── Tab Navigation ────────────────────────────────────────────────────── */}
        <div className="flex flex-wrap gap-2 border-b border-slate-800 pb-2 text-xs">
          <button
            type="button"
            onClick={() => setActiveTab('parameters')}
            className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
              activeTab === 'parameters'
                ? 'bg-cyan-600 text-white'
                : 'bg-slate-800/80 text-slate-300 hover:bg-slate-700'
            }`}
          >
            Parameter Comparisons ({parameters.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('objects')}
            className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
              activeTab === 'objects'
                ? 'bg-cyan-600 text-white'
                : 'bg-slate-800/80 text-slate-300 hover:bg-slate-700'
            }`}
          >
            Tracked Objects ({object_changes.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('transitions')}
            className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
              activeTab === 'transitions'
                ? 'bg-cyan-600 text-white'
                : 'bg-slate-800/80 text-slate-300 hover:bg-slate-700'
            }`}
          >
            Transition Matrix ({transition_matrix.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('ranking')}
            className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
              activeTab === 'ranking'
                ? 'bg-cyan-600 text-white'
                : 'bg-slate-800/80 text-slate-300 hover:bg-slate-700'
            }`}
          >
            Top Change Ranking
          </button>
        </div>

        {/* ── TAB 1: Parameter Comparison Table ─────────────────────────────────── */}
        {activeTab === 'parameters' && (
          <div className="space-y-3">
            <div className="overflow-x-auto rounded-lg border border-slate-800">
              <table className="w-full text-xs text-left">
                <thead className="bg-slate-950 text-slate-400 font-mono text-[11px] uppercase border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-3">Domain & Parameter</th>
                    <th className="py-2.5 px-3">{date1}</th>
                    <th className="py-2.5 px-3">{date2}</th>
                    <th className="py-2.5 px-3">Absolute Change</th>
                    <th className="py-2.5 px-3">Percentage Change</th>
                    <th className="py-2.5 px-3">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 bg-slate-900/50">
                  {parameters.map((param, idx) => (
                    <tr
                      key={idx}
                      className="hover:bg-white/5 transition-colors cursor-pointer"
                      onClick={() => onParameterSelect?.(param.category)}
                    >
                      <td className="py-2 px-3 font-medium text-slate-200">
                        <div className="flex items-center gap-2">
                          {getCategoryIcon(param.category)}
                          <span>{param.parameter}</span>
                        </div>
                        {param.notes && (
                          <div className="text-[10px] text-amber-400/90 font-mono mt-0.5">
                            {param.notes}
                          </div>
                        )}
                      </td>
                      <td className="py-2 px-3 font-mono text-slate-300">
                        {param.value_1 !== null && param.value_1 !== undefined
                          ? `${param.value_1} ${param.unit}`
                          : 'Not available'}
                      </td>
                      <td className="py-2 px-3 font-mono text-slate-300">
                        {param.value_2 !== null && param.value_2 !== undefined
                          ? `${param.value_2} ${param.unit}`
                          : 'Not available'}
                      </td>
                      <td className="py-2 px-3 font-mono">
                        {param.absolute_change !== null && param.absolute_change !== undefined ? (
                          <span
                            className={
                              param.absolute_change > 0
                                ? 'text-emerald-400'
                                : param.absolute_change < 0
                                ? 'text-rose-400'
                                : 'text-slate-400'
                            }
                          >
                            {param.absolute_change > 0 ? `+${param.absolute_change}` : param.absolute_change}{' '}
                            {param.unit}
                          </span>
                        ) : (
                          <span className="text-slate-500">—</span>
                        )}
                      </td>
                      <td className="py-2 px-3 font-mono">
                        <span
                          className={
                            param.percentage_change && param.percentage_change > 0
                              ? 'text-emerald-400'
                              : param.percentage_change && param.percentage_change < 0
                              ? 'text-rose-400'
                              : 'text-slate-300'
                          }
                        >
                          {param.percentage_change_text || '—'}
                        </span>
                      </td>
                      <td className="py-2 px-3">
                        <span
                          className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase font-semibold ${getStatusColor(
                            param.status
                          )}`}
                        >
                          {param.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── TAB 2: Object-Level Change Analysis ───────────────────────────────── */}
        {activeTab === 'objects' && (
          <div className="space-y-3">
            {/* Filter pills */}
            <div className="flex flex-wrap items-center gap-1.5 text-xs">
              <span className="text-slate-400 text-[11px] mr-1">Filter Status:</span>
              {['ALL', 'NEW', 'REMOVED', 'EXPANDED', 'CONTRACTED', 'UNCHANGED'].map((st) => (
                <button
                  key={st}
                  type="button"
                  onClick={() => setObjectFilter(st)}
                  className={`px-2.5 py-1 rounded text-[11px] font-mono font-medium transition-colors ${
                    objectFilter === st
                      ? 'bg-cyan-600 text-white'
                      : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                  }`}
                >
                  {st}
                </button>
              ))}
            </div>

            <div className="max-h-80 overflow-y-auto space-y-2 pr-1">
              {filteredObjects.length === 0 ? (
                <div className="text-xs text-slate-500 py-6 text-center font-mono">
                  No objects matched the filter '{objectFilter}'.
                </div>
              ) : (
                filteredObjects.map((obj) => {
                  const isSelected = selectedObjectId === obj.object_id;
                  return (
                    <div
                      key={obj.object_id}
                      onClick={() => onSelectObject && onSelectObject(isSelected ? null : obj)}
                      className={`p-3 rounded-lg border transition-all cursor-pointer text-xs ${
                        isSelected
                          ? 'bg-cyan-950/40 border-cyan-500 shadow-md'
                          : 'bg-slate-950/40 border-slate-800 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-bold text-slate-200">
                            {obj.object_id}
                          </span>
                          <span className="text-slate-400 text-[11px]">
                            {obj.object_class}
                          </span>
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-mono border ${getStatusColor(
                              obj.status
                            )}`}
                          >
                            {obj.status}
                          </span>
                        </div>

                        {obj.region_sector && (
                          <div className="flex items-center gap-1 text-[11px] text-cyan-400 font-mono">
                            <MapPin className="w-3 h-3" />
                            <span>{obj.region_sector}</span>
                          </div>
                        )}
                      </div>

                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-2 pt-2 border-t border-slate-800/60 font-mono text-[11px] text-slate-400">
                        <div>
                          <span className="text-slate-500">Area {date1}: </span>
                          <span className="text-slate-200">{obj.area_date1 ?? 0} m²</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Area {date2}: </span>
                          <span className="text-slate-200">{obj.area_date2 ?? 0} m²</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Change: </span>
                          <span
                            className={
                              (obj.area_change || 0) > 0
                                ? 'text-emerald-400'
                                : (obj.area_change || 0) < 0
                                ? 'text-rose-400'
                                : 'text-slate-300'
                            }
                          >
                            {obj.area_change !== null && obj.area_change !== undefined
                              ? `${obj.area_change > 0 ? `+${obj.area_change}` : obj.area_change} m²`
                              : '—'}
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-500">%: </span>
                          <span className="text-slate-300">
                            {obj.percentage_change_text || '—'}
                          </span>
                        </div>
                      </div>

                      {isSelected && (
                        <div className="mt-2.5 p-2 bg-slate-900 rounded border border-cyan-900/60 text-[11px] font-mono text-slate-300 space-y-1">
                          <div className="text-cyan-400 font-semibold">Detailed Object Properties:</div>
                          {obj.centroid_geo && (
                            <div>
                              Geo Coordinate: [{obj.centroid_geo.lat.toFixed(5)}, {obj.centroid_geo.lon.toFixed(5)}]
                            </div>
                          )}
                          {obj.centroid_pixel && (
                            <div>
                              Pixel Centroid: [{obj.centroid_pixel[0]}, {obj.centroid_pixel[1]}]
                            </div>
                          )}
                          {obj.perimeter_px && <div>Perimeter: {obj.perimeter_px} px</div>}
                          {obj.shape_compactness && (
                            <div>Compactness: {obj.shape_compactness}</div>
                          )}
                          {obj.confidence && (
                            <div>Confidence: {(obj.confidence * 100).toFixed(0)}%</div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}

        {/* ── TAB 3: Land-Cover Transition Matrix ────────────────────────────────── */}
        {activeTab === 'transitions' && (
          <div className="space-y-3">
            {transition_matrix.length === 0 ? (
              <div className="text-xs text-slate-500 py-6 text-center font-mono">
                No significant class-level land-cover transitions detected.
              </div>
            ) : (
              <div className="overflow-x-auto rounded-lg border border-slate-800">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-950 text-slate-400 font-mono text-[11px] uppercase border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-3">Previous Class ({date1})</th>
                      <th className="py-2.5 px-3">Current Class ({date2})</th>
                      <th className="py-2.5 px-3">Area Changed (ha)</th>
                      <th className="py-2.5 px-3">Area Changed (m²)</th>
                      <th className="py-2.5 px-3">% of Total Change</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 bg-slate-900/50">
                    {transition_matrix.map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                        <td className="py-2 px-3 font-semibold text-slate-300">
                          {row.previous_class}
                        </td>
                        <td className="py-2 px-3 font-semibold text-cyan-400">
                          {row.current_class}
                        </td>
                        <td className="py-2 px-3 font-mono text-slate-200">
                          {row.area_changed_hectares ?? 0} ha
                        </td>
                        <td className="py-2 px-3 font-mono text-slate-400">
                          {row.area_changed_m2?.toLocaleString() ?? 0} m²
                        </td>
                        <td className="py-2 px-3 font-mono font-medium text-emerald-400">
                          {row.percentage_of_total_change}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* ── TAB 4: Change Ranking ─────────────────────────────────────────────── */}
        {activeTab === 'ranking' && (
          <div className="space-y-2">
            <div className="text-xs text-slate-400 font-medium mb-1">
              Top changes ranked by relative magnitude:
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {change_ranking.map((item) => (
                <div
                  key={item.rank}
                  className="bg-slate-950/40 border border-slate-800 rounded-lg p-3 flex items-center justify-between"
                >
                  <div className="flex items-center gap-2.5">
                    <span className="w-5 h-5 rounded-full bg-cyan-950 border border-cyan-700/60 text-cyan-400 text-xs font-mono flex items-center justify-center shrink-0">
                      {item.rank}
                    </span>
                    <div>
                      <div className="text-xs font-semibold text-slate-200">
                        {item.parameter}
                      </div>
                      <div className="text-[11px] text-slate-400 font-mono">
                        {item.change_summary}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <span
                      className={`text-xs font-mono font-bold ${
                        item.direction === 'increase'
                          ? 'text-emerald-400'
                          : item.direction === 'decrease'
                          ? 'text-rose-400'
                          : 'text-slate-400'
                      }`}
                    >
                      {item.direction.toUpperCase()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Quality Control & Scientific Disclaimers ─────────────────────────── */}
        {limitations_and_disclaimers.length > 0 && (
          <div className="bg-amber-950/20 border border-amber-800/40 rounded-lg p-3 text-xs space-y-1">
            <div className="flex items-center gap-1.5 text-amber-400 font-semibold">
              <ShieldAlert className="w-3.5 h-3.5" />
              <span>Data & Scientific Limitations / Quality Control</span>
            </div>
            <ul className="list-disc list-inside text-amber-200/80 space-y-0.5 text-[11px]">
              {limitations_and_disclaimers.map((disc, idx) => (
                <li key={idx}>{disc}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="bg-slate-950/60 px-5 py-2.5 border-t border-slate-800 text-[11px] text-slate-500 font-mono flex items-center justify-between">
        <span>Grounded Analysis Engine • No fabricated parameters</span>
        <span>
          Primary Change Sector: <strong className="text-slate-300">{spatial_summary?.primary_change_sector || 'Central'}</strong>
        </span>
      </div>
    </div>
  );
};
