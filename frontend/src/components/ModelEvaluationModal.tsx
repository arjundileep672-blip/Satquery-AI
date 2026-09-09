import React, { useState, useEffect } from 'react';
import {
  X,
  Activity,
  BarChart2,
  CheckCircle2,
  Layers,
  Database,
  Info,
  RefreshCw,
  Award,
  Filter,
  Timer,
} from 'lucide-react';
import type { FleetMetricsResponse, ModelMetricsResponse } from '../types';
import { fetchFleetMetrics, fetchModelMetrics } from '../api/satquery';

interface ModelEvaluationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ModelEvaluationModal: React.FC<ModelEvaluationModalProps> = ({
  isOpen,
  onClose,
}) => {
  const [metrics, setMetrics] = useState<ModelMetricsResponse | null>(null);
  const [fleet, setFleet] = useState<FleetMetricsResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [matrixMode, setMatrixMode] = useState<'normalized' | 'counts'>('normalized');
  const [hoveredCell, setHoveredCell] = useState<{
    trueClass: string;
    predClass: string;
    count: number;
    pct: number;
    i: number;
    j: number;
  } | null>(null);
  const [selectedClass, setSelectedClass] = useState<string | null>(null);

  const loadMetrics = React.useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [data, fleetData] = await Promise.all([
        fetchModelMetrics(),
        fetchFleetMetrics(),
      ]);
      setMetrics(data);
      setFleet(fleetData);
    } catch (err: any) {
      setError(err?.message || 'Failed to load model metrics');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);

    if (!metrics) {
      loadMetrics();
    }

    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, metrics, loadMetrics, onClose]);

  if (!isOpen) return null;

  const classes = metrics?.class_names || [];
  const matrix =
    matrixMode === 'normalized'
      ? metrics?.normalized_confusion_matrix || []
      : metrics?.confusion_matrix || [];

  const getCellColor = (val: number, isDiagonal: boolean, maxVal: number) => {
    if (val === 0) return 'bg-slate-900/40 text-slate-600';
    if (matrixMode === 'normalized') {
      if (val >= 80) return 'bg-emerald-500/80 text-white font-bold';
      if (val >= 70) return 'bg-emerald-600/70 text-emerald-100 font-semibold';
      if (val >= 50) return 'bg-emerald-700/60 text-emerald-200';
      if (val >= 25) return 'bg-emerald-800/50 text-emerald-300';
      if (val >= 10) return 'bg-amber-500/30 text-amber-200';
      return 'bg-slate-800/60 text-slate-400';
    } else {
      const ratio = maxVal > 0 ? val / maxVal : 0;
      if (isDiagonal) {
        if (ratio >= 0.75) return 'bg-emerald-500/80 text-white font-bold';
        if (ratio >= 0.5) return 'bg-emerald-600/70 text-emerald-100 font-semibold';
        return 'bg-emerald-700/50 text-emerald-200';
      }
      if (ratio >= 0.15) return 'bg-amber-500/30 text-amber-200';
      return 'bg-slate-800/60 text-slate-400';
    }
  };

  // Find max value in matrix for relative scaling
  let maxMatrixVal = 1;
  if (matrix.length > 0) {
    maxMatrixVal = Math.max(...matrix.map((row) => Math.max(...row)));
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 overflow-y-auto">
      <div
        className="relative w-full max-w-5xl bg-slate-900 border border-slate-700/80 rounded-2xl shadow-2xl overflow-hidden max-h-[92vh] flex flex-col animate-in fade-in zoom-in-95 duration-150"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="px-6 py-4 border-b border-slate-800 bg-slate-900/90 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-slate-100 tracking-wide">
                  Model Performance & Confusion Matrix
                </h2>
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-700">
                  Fleet + EuroSAT
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Native-benchmark scores, local smoke-test latency, and the EuroSAT confusion matrix
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={loadMetrics}
              disabled={isLoading}
              className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors"
              title="Refresh Metrics"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-2 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded-lg transition-colors"
              title="Close (Esc)"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Modal Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {isLoading && !metrics ? (
            <div className="h-64 flex flex-col items-center justify-center gap-3 text-slate-400">
              <RefreshCw className="w-8 h-8 animate-spin text-emerald-400" />
              <p className="text-sm font-medium">Computing evaluation statistics & confusion matrix...</p>
            </div>
          ) : error ? (
            <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-center justify-between">
              <span>{error}</span>
              <button
                onClick={loadMetrics}
                className="px-3 py-1 bg-rose-600 hover:bg-rose-500 text-white rounded text-xs"
              >
                Retry
              </button>
            </div>
          ) : metrics ? (
            <>
              {fleet && (
                <FleetPerformanceCharts fleet={fleet} />
              )}

              {/* 1. KPI Metric Summary Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 text-center space-y-1">
                  <div className="flex items-center justify-center gap-1.5 text-xs text-slate-400">
                    <Award className="w-3.5 h-3.5 text-emerald-400" />
                    <span>Top-1 Accuracy</span>
                  </div>
                  <div className="text-2xl font-extrabold text-emerald-400 font-mono">
                    {metrics.overall_accuracy.toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-slate-500">Overall Benchmark</div>
                </div>

                <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 text-center space-y-1">
                  <div className="flex items-center justify-center gap-1.5 text-xs text-slate-400">
                    <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Top-3 Accuracy</span>
                  </div>
                  <div className="text-2xl font-extrabold text-cyan-400 font-mono">
                    {metrics.top_3_accuracy.toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-slate-500">Candidate Pool</div>
                </div>

                <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 text-center space-y-1">
                  <div className="text-xs text-slate-400">Macro F1</div>
                  <div className="text-2xl font-bold text-slate-100 font-mono">
                    {metrics.macro_f1.toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-slate-500">Unweighted Mean</div>
                </div>

                <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 text-center space-y-1">
                  <div className="text-xs text-slate-400">Precision</div>
                  <div className="text-2xl font-bold text-slate-100 font-mono">
                    {metrics.macro_precision.toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-slate-500">Macro Average</div>
                </div>

                <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 text-center space-y-1">
                  <div className="text-xs text-slate-400">Recall</div>
                  <div className="text-2xl font-bold text-slate-100 font-mono">
                    {metrics.macro_recall.toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-slate-500">Sensitivity</div>
                </div>

                <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 text-center space-y-1">
                  <div className="flex items-center justify-center gap-1.5 text-xs text-slate-400">
                    <Database className="w-3.5 h-3.5 text-purple-400" />
                    <span>Validation Set</span>
                  </div>
                  <div className="text-2xl font-bold text-purple-400 font-mono">
                    {metrics.total_samples.toLocaleString()}
                  </div>
                  <div className="text-[10px] text-slate-500">Sentinel-2 Samples</div>
                </div>
              </div>

              {/* 2. Confusion Matrix Heatmap Section */}
              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
                  <div className="flex items-center gap-2">
                    <BarChart2 className="w-4 h-4 text-emerald-400" />
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                      Confusion Matrix Heatmap (10 × 10)
                    </h3>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="text-[11px] text-slate-400 font-medium">Display Value:</span>
                    <div className="flex items-center bg-slate-900 border border-slate-700/80 rounded-lg p-0.5 text-xs">
                      <button
                        onClick={() => setMatrixMode('normalized')}
                        className={`px-2.5 py-1 rounded-md transition-all ${
                          matrixMode === 'normalized'
                            ? 'bg-emerald-600 text-white font-medium shadow-sm'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Normalized (%)
                      </button>
                      <button
                        onClick={() => setMatrixMode('counts')}
                        className={`px-2.5 py-1 rounded-md transition-all ${
                          matrixMode === 'counts'
                            ? 'bg-emerald-600 text-white font-medium shadow-sm'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        Raw Counts
                      </button>
                    </div>
                  </div>
                </div>

                {/* Matrix Interactive View */}
                <div className="overflow-x-auto pt-2 pb-1">
                  <div className="inline-block min-w-[650px]">
                    {/* Column Labels (Predicted) */}
                    <div className="text-center text-[11px] font-bold text-slate-400 mb-2 uppercase tracking-wider">
                      Predicted Class (Model Output)
                    </div>

                    <div className="grid grid-cols-[140px_repeat(10,minmax(46px,1fr))] gap-1 items-center">
                      {/* Top-left corner */}
                      <div className="text-[11px] font-mono text-slate-500 text-right pr-2">
                        True ↓ \ Pred →
                      </div>

                      {/* Header columns */}
                      {classes.map((cls, idx) => (
                        <div
                          key={cls}
                          className={`text-[10px] font-mono text-center truncate px-1 py-1 rounded ${
                            selectedClass === cls
                              ? 'bg-emerald-500/20 text-emerald-300 font-bold'
                              : 'text-slate-400'
                          }`}
                          title={`Class ${idx + 1}: ${cls}`}
                        >
                          {cls.slice(0, 4)}..
                        </div>
                      ))}

                      {/* Matrix Rows */}
                      {matrix.map((row, i) => {
                        const trueCls = classes[i];
                        const isSelectedRow = selectedClass === trueCls;

                        return (
                          <React.Fragment key={trueCls}>
                            {/* Row label */}
                            <button
                              type="button"
                              onClick={() =>
                                setSelectedClass(selectedClass === trueCls ? null : trueCls)
                              }
                              className={`text-[11px] font-mono text-right pr-2 truncate text-left hover:text-emerald-400 transition-colors flex items-center justify-between ${
                                isSelectedRow
                                  ? 'text-emerald-300 font-bold bg-emerald-950/40 rounded px-1'
                                  : 'text-slate-300'
                              }`}
                              title={`Click to filter: ${trueCls}`}
                            >
                              <span className="truncate">{trueCls}</span>
                              <span className="text-[9px] text-slate-500 ml-1">#{i + 1}</span>
                            </button>

                            {/* Row cells */}
                            {row.map((val, j) => {
                              const predCls = classes[j];
                              const isDiagonal = i === j;
                              const cellStyle = getCellColor(val, isDiagonal, maxMatrixVal);
                              const isHovered =
                                hoveredCell?.i === i && hoveredCell?.j === j;

                              return (
                                <div
                                  key={`${i}-${j}`}
                                  onMouseEnter={() =>
                                    setHoveredCell({
                                      trueClass: trueCls,
                                      predClass: predCls,
                                      count: metrics.confusion_matrix[i]?.[j] || 0,
                                      pct: metrics.normalized_confusion_matrix[i]?.[j] || 0,
                                      i,
                                      j,
                                    })
                                  }
                                  onMouseLeave={() => setHoveredCell(null)}
                                  className={`h-9 flex items-center justify-center rounded text-[11px] font-mono cursor-pointer transition-transform duration-75 relative select-none ${cellStyle} ${
                                    isHovered
                                      ? 'ring-2 ring-cyan-400 scale-105 z-10'
                                      : isDiagonal
                                      ? 'border border-emerald-400/30'
                                      : ''
                                  }`}
                                >
                                  {matrixMode === 'normalized'
                                    ? `${val.toFixed(0)}%`
                                    : val}
                                </div>
                              );
                            })}
                          </React.Fragment>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Interactive Tooltip Status Bar */}
                <div className="h-9 px-3 bg-slate-900/90 border border-slate-800 rounded-lg flex items-center justify-between text-xs text-slate-300 font-mono">
                  {hoveredCell ? (
                    <div className="flex items-center gap-3">
                      <span className="text-slate-400">Cell Inspection:</span>
                      <span>
                        True: <strong className="text-emerald-400">{hoveredCell.trueClass}</strong>
                      </span>
                      <span>→</span>
                      <span>
                        Predicted: <strong className="text-cyan-400">{hoveredCell.predClass}</strong>
                      </span>
                      <span className="text-slate-500">|</span>
                      <span>
                        Count: <strong className="text-white">{hoveredCell.count}</strong> samples
                      </span>
                      <span>
                        Accuracy / Rate:{' '}
                        <strong className="text-emerald-300">
                          {hoveredCell.pct.toFixed(1)}%
                        </strong>
                      </span>
                      {hoveredCell.i === hoveredCell.j && (
                        <span className="text-[10px] px-1.5 py-0.2 rounded bg-emerald-950 text-emerald-400 border border-emerald-700">
                          True Positive
                        </span>
                      )}
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-slate-500 text-[11px]">
                      <Info className="w-3.5 h-3.5 text-slate-400" />
                      <span>
                        Hover over any grid cell to inspect class misclassifications, confusion rates, and sample counts.
                      </span>
                    </div>
                  )}

                  {selectedClass && (
                    <button
                      onClick={() => setSelectedClass(null)}
                      className="flex items-center gap-1 text-[11px] text-amber-400 hover:underline"
                    >
                      <Filter className="w-3 h-3" />
                      Clear Filter: {selectedClass}
                    </button>
                  )}
                </div>
              </div>

              {/* 3. Per-Class Performance Breakdown Table */}
              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-emerald-400" />
                    <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                      Per-Class Classification Metrics (Precision / Recall / F1)
                    </h3>
                  </div>
                  <span className="text-[11px] text-slate-400">
                    10 Earth Observation Surface Classes
                  </span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs font-mono">
                    <thead>
                      <tr className="text-slate-400 border-b border-slate-800 text-[11px]">
                        <th className="pb-2 font-semibold">Class Name</th>
                        <th className="pb-2 font-semibold text-right">Support</th>
                        <th className="pb-2 font-semibold pl-4">Precision (%)</th>
                        <th className="pb-2 font-semibold pl-4">Recall (%)</th>
                        <th className="pb-2 font-semibold pl-4">F1-Score (%)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {classes
                        .filter((cls) => !selectedClass || selectedClass === cls)
                        .map((cls) => {
                          const pcm = metrics.per_class_metrics?.[cls] || {
                            precision: 0,
                            recall: 0,
                            f1: 0,
                            support: 0,
                          };

                          return (
                            <tr
                              key={cls}
                              className={`hover:bg-slate-900/50 transition-colors ${
                                selectedClass === cls ? 'bg-emerald-950/30' : ''
                              }`}
                            >
                              <td className="py-2.5 font-sans font-medium text-slate-200 flex items-center gap-2">
                                <span className="w-2 h-2 rounded-full bg-emerald-400" />
                                {cls}
                              </td>
                              <td className="py-2.5 text-right text-slate-400">
                                {pcm.support}
                              </td>
                              <td className="py-2.5 pl-4">
                                <div className="flex items-center gap-2">
                                  <div className="w-20 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                                    <div
                                      className="bg-emerald-400 h-full rounded-full"
                                      style={{ width: `${Math.min(100, pcm.precision)}%` }}
                                    />
                                  </div>
                                  <span className="w-12 text-slate-200">
                                    {pcm.precision.toFixed(1)}%
                                  </span>
                                </div>
                              </td>
                              <td className="py-2.5 pl-4">
                                <div className="flex items-center gap-2">
                                  <div className="w-20 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                                    <div
                                      className="bg-cyan-400 h-full rounded-full"
                                      style={{ width: `${Math.min(100, pcm.recall)}%` }}
                                    />
                                  </div>
                                  <span className="w-12 text-slate-200">
                                    {pcm.recall.toFixed(1)}%
                                  </span>
                                </div>
                              </td>
                              <td className="py-2.5 pl-4">
                                <div className="flex items-center gap-2">
                                  <div className="w-20 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                                    <div
                                      className="bg-purple-400 h-full rounded-full"
                                      style={{ width: `${Math.min(100, pcm.f1)}%` }}
                                    />
                                  </div>
                                  <span className="w-12 text-purple-200 font-bold">
                                    {pcm.f1.toFixed(1)}%
                                  </span>
                                </div>
                              </td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          ) : null}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-slate-900/95 flex items-center justify-between text-xs text-slate-400 font-mono">
          <div className="flex items-center gap-2">
            <span>Offline Artifact:</span>
            <span className="text-slate-300">
              fleet_metrics.json · smoke_test_results.json · eurosat/evaluation_metrics.json
            </span>
          </div>

          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

const FleetPerformanceCharts: React.FC<{ fleet: FleetMetricsResponse }> = ({ fleet }) => {
  const maxScore = 100;
  const tests = fleet.smoke?.tests || [];
  const maxLatency = Math.max(1, ...tests.map((t) => t.latency_ms || 0));
  const smokePassed = tests.filter((t) => t.passed).length;
  const pytest = fleet.smoke?.pytest;

  return (
    <div className="space-y-4">
      <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-3">
        <div className="flex items-center justify-between gap-3 border-b border-slate-800/80 pb-2">
          <div className="flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Headline score by model
            </h3>
          </div>
          <span className="text-[10px] text-slate-500">Native benchmark · not one shared test set</span>
        </div>
        <div className="space-y-2.5">
          {fleet.models.map((row) => (
            <div key={row.id} className="grid grid-cols-[140px_1fr_auto] gap-3 items-center">
              <div className="text-[11px] font-medium text-slate-200 truncate" title={row.name}>
                {row.short_name}
              </div>
              <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${row.local_eval ? 'bg-emerald-400' : 'bg-cyan-400'}`}
                  style={{ width: `${Math.min(100, (row.score / maxScore) * 100)}%` }}
                />
              </div>
              <div className="text-[11px] font-mono text-slate-200 w-28 text-right">
                {row.score.toFixed(1)}%
                <span className="text-slate-500 ml-1">{row.metric.split(' ')[0]}</span>
              </div>
            </div>
          ))}
        </div>
        <p className="text-[10px] text-slate-500 leading-relaxed">{fleet.disclaimer}</p>
      </div>

      <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-2">
          <div className="flex items-center gap-2">
            <Timer className="w-4 h-4 text-amber-400" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
              Local smoke tests
            </h3>
          </div>
          <div className="flex items-center gap-2 text-[10px] font-mono">
            {fleet.smoke ? (
              <>
                <span className="px-2 py-0.5 rounded-full bg-emerald-950 text-emerald-300 border border-emerald-700">
                  {smokePassed}/{tests.length} smoke pass
                </span>
                {pytest && (
                  <span className="px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-600">
                    pytest {pytest.passed} passed / {pytest.failed} failed
                  </span>
                )}
                <span className="text-slate-500">
                  {fleet.smoke.device} · {new Date(fleet.smoke.run_at).toLocaleString()}
                </span>
              </>
            ) : (
              <span className="text-amber-400">
                No smoke run yet — python scripts/smoke_test_models.py
              </span>
            )}
          </div>
        </div>

        {tests.length > 0 && (
          <div className="space-y-2.5">
            {tests.map((row) => (
              <div key={row.id} className="grid grid-cols-[160px_1fr_auto_auto] gap-3 items-center">
                <div className="text-[11px] font-medium text-slate-200 truncate">{row.name}</div>
                <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${row.passed ? 'bg-amber-400' : 'bg-rose-500'}`}
                    style={{
                      width: `${Math.min(100, ((row.latency_ms || 0) / maxLatency) * 100)}%`,
                    }}
                  />
                </div>
                <div className="text-[11px] font-mono text-slate-200 w-20 text-right">
                  {row.latency_ms != null ? `${row.latency_ms.toFixed(0)} ms` : '—'}
                </div>
                <span
                  className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                    row.passed
                      ? 'bg-emerald-950 text-emerald-300 border border-emerald-700'
                      : 'bg-rose-950 text-rose-300 border border-rose-700'
                  }`}
                >
                  {row.passed ? 'PASS' : 'FAIL'}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
