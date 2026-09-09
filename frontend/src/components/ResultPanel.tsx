import React from 'react';
import {
  Brain,
  CheckCircle2,
  AlertTriangle,
  FileCheck,
  Info,
} from 'lucide-react';
import type { AnalysisResult } from '../types';

interface ResultPanelProps {
  result: AnalysisResult | null;
}

export const ResultPanel: React.FC<ResultPanelProps> = ({ result }) => {
  if (!result) return null;

  return (
    <div className="sq-glass p-4 space-y-5">
      {/* 1. Header Bar: Task & Request ID */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <FileCheck className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm md:text-base font-bold text-slate-100">
              Analysis Results
            </h3>
            <p className="text-[11px] text-slate-400 font-mono">
              Request ID: {result.request_id}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Task Badge */}
          <span className="px-3 py-1 rounded-full text-xs font-bold bg-cyan-950/70 text-cyan-300 border border-cyan-800/70 font-mono shadow-sm">
            Task: {result.task || result.operation.replace(/_/g, ' ')}
          </span>

          {/* Confidence Badge */}
          <span className="px-2.5 py-1 rounded-full text-xs font-medium bg-slate-800 text-slate-300 border border-slate-700 font-mono flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-amber-400" />
            <span>
              {result.confidence !== null && result.confidence !== undefined
                ? `Confidence: ${(result.confidence * 100).toFixed(0)}%`
                : 'Confidence: Qualitative (VLM)'}
            </span>
          </span>
        </div>
      </div>

      {/* 2. Large Natural Language Answer */}
      <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4.5 space-y-2.5 shadow-inner">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold text-emerald-400 flex items-center gap-1.5 uppercase tracking-wider">
            <Brain className="w-4 h-4" />
            <span>AI Analytical Synthesis</span>
          </span>
          <span className="text-[10px] text-slate-500 font-mono">
            {result.models_used.length > 0 ? result.models_used.join(' · ') : 'Vision-Language Engine'}
          </span>
        </div>
        <div className="text-sm md:text-base text-slate-200 leading-relaxed whitespace-pre-line font-normal">
          {result.answer}
        </div>
      </div>

      {/* 3. Analysis Trace (Execution Milestones) */}
      {result.analysis_trace && result.analysis_trace.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-bold text-slate-300 flex items-center gap-2 uppercase tracking-wider">
            <CheckCircle2 className="w-4 h-4 text-cyan-400" />
            <span>Analysis Trace / Execution Milestones</span>
          </h4>
          <div className="bg-slate-950/50 border border-slate-800/80 rounded-lg p-3">
            <ol className="space-y-2 text-xs">
              {result.analysis_trace.map((step, idx) => (
                <li key={idx} className="flex items-start gap-2.5 text-slate-300">
                  <span className="w-4 h-4 rounded-full bg-cyan-950 border border-cyan-700/60 text-cyan-400 text-[10px] font-mono flex items-center justify-center shrink-0 mt-0.5 font-bold">
                    {idx + 1}
                  </span>
                  <span className="font-mono text-slate-300 text-[11px] md:text-xs">{step}</span>
                </li>
              ))}
            </ol>
          </div>
        </div>
      )}

      {/* 4. Operational & Scientific Warnings */}
      {result.warnings && result.warnings.length > 0 && (
        <div className="bg-amber-950/20 border border-amber-800/40 rounded-lg p-3 text-xs space-y-1">
          <div className="flex items-center gap-1.5 text-amber-400 font-semibold">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>Operational & Ground Advisory</span>
          </div>
          <ul className="list-disc list-inside text-amber-200/80 space-y-0.5 text-[11px]">
            {result.warnings.map((warn, i) => (
              <li key={i}>{warn}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
