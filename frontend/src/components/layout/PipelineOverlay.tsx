import React, { useEffect, useState } from 'react';

const STAGES_SINGLE = [
  'QUERY RECEIVED',
  'QUERY UNDERSTANDING',
  'TASK SELECTION',
  'VISION ANALYSIS',
  'SPATIAL GROUNDING',
  'RESULT GENERATED',
];

const STAGES_PAIR = [
  'QUERY RECEIVED',
  'QUERY UNDERSTANDING',
  'TASK SELECTION',
  'IMAGE REGISTRATION',
  'VISION ANALYSIS',
  'CHANGE EXTRACTION',
  'SPATIAL GROUNDING',
  'RESULT GENERATED',
];

interface PipelineOverlayProps {
  active: boolean;
  hasSecondImage: boolean;
}

export const PipelineOverlay: React.FC<PipelineOverlayProps> = ({ active, hasSecondImage }) => {
  const stages = hasSecondImage ? STAGES_PAIR : STAGES_SINGLE;
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    if (!active) {
      setIdx(0);
      return;
    }
    const t = window.setInterval(() => {
      setIdx((i) => (i < stages.length - 1 ? i + 1 : i));
    }, 900);
    return () => window.clearInterval(t);
  }, [active, stages.length]);

  if (!active) return null;

  return (
    <div className="absolute inset-0 z-40 bg-[#070b12]/72 backdrop-blur-[2px] flex items-center justify-center">
      <div className="sq-glass px-8 py-6 min-w-[280px]">
        <div className="text-[10px] sq-mono tracking-[0.22em] text-teal-300 mb-4">ANALYZING</div>
        <div className="h-[2px] bg-white/10 overflow-hidden mb-5">
          <div className="h-full w-1/3 bg-teal-300 sq-indeterminate" />
        </div>
        <ol className="space-y-1.5">
          {stages.map((s, i) => {
            const done = i < idx;
            const cur = i === idx;
            return (
              <li
                key={s}
                className={`flex items-center gap-3 text-[11px] sq-mono ${
                  done ? 'text-teal-200' : cur ? 'text-slate-100' : 'text-slate-600'
                }`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${done ? 'bg-teal-400' : cur ? 'bg-teal-200 sq-live' : 'bg-slate-700'}`} />
                {s}
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
};
