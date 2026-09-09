import React, { useEffect, useState } from 'react';
import { Satellite, Radio, CheckCircle2 } from 'lucide-react';

interface ProcessingStatusProps {
  isLoading: boolean;
  hasSecondImage: boolean;
}

export const ProcessingStatus: React.FC<ProcessingStatusProps> = ({
  isLoading,
  hasSecondImage,
}) => {
  const [currentStageIndex, setCurrentStageIndex] = useState(0);

  const stages = [
    'Uploading image raster...',
    'Understanding query & classifying intent...',
    'Selecting remote sensing model pipeline...',
    hasSecondImage
      ? 'Registering bi-temporal imagery via ORB + RANSAC...'
      : 'Calibrating radiometric indices & CRS...',
    'Running detector & spatial inference...',
    hasSecondImage ? 'Detecting changes with ChangeFormer...' : 'Generating segmentation masks...',
    'Generating natural-language answer & explanation...',
  ];

  useEffect(() => {
    if (!isLoading) {
      setCurrentStageIndex(0);
      return;
    }

    const interval = setInterval(() => {
      setCurrentStageIndex((prev) => (prev < stages.length - 1 ? prev + 1 : prev));
    }, 1100);

    return () => clearInterval(interval);
  }, [isLoading, stages.length]);

  if (!isLoading) return null;

  return (
    <div className="bg-slate-900 border border-emerald-500/40 rounded-xl p-5 shadow-xl space-y-4 animate-fade-in ring-1 ring-emerald-500/20">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 relative overflow-hidden">
            <Radio className="w-5 h-5 animate-pulse" />
            <span className="absolute inset-0 bg-emerald-400/10 animate-ping rounded-lg" />
          </div>
          <div>
            <h4 className="text-sm font-bold text-slate-100 flex items-center gap-2">
              <span>Vision-Language Inference in Progress</span>
            </h4>
            <p className="text-xs text-emerald-400 font-mono flex items-center gap-1.5 mt-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
              <span>{stages[currentStageIndex]}</span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-1 text-xs text-slate-400 font-mono">
          <Satellite className="w-4 h-4 text-cyan-400 animate-bounce" />
          <span>Multimodal Pipeline</span>
        </div>
      </div>

      {/* Indeterminate Animated Progress Bar (No fake percentages) */}
      <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800 relative">
        <div className="h-full bg-gradient-to-r from-emerald-500 via-teal-400 to-cyan-400 rounded-full animate-indeterminate" />
      </div>

      {/* Pipeline Milestone Checklist */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2 pt-1 text-[11px] font-mono">
        {stages.slice(0, 6).map((stage, idx) => {
          const isDone = idx < currentStageIndex;
          const isCurrent = idx === currentStageIndex;

          return (
            <div
              key={idx}
              className={`flex items-center gap-2 p-1.5 rounded border transition-colors ${
                isDone
                  ? 'bg-emerald-950/20 border-emerald-800/40 text-emerald-300'
                  : isCurrent
                  ? 'bg-cyan-950/30 border-cyan-700/60 text-cyan-200'
                  : 'bg-slate-950/40 border-slate-800/50 text-slate-500'
              }`}
            >
              <CheckCircle2
                className={`w-3.5 h-3.5 shrink-0 ${
                  isDone
                    ? 'text-emerald-400'
                    : isCurrent
                    ? 'text-cyan-400 animate-pulse'
                    : 'text-slate-600'
                }`}
              />
              <span className="truncate">{stage.replace('...', '')}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
