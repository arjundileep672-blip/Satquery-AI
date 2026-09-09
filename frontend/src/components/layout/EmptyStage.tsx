import React from 'react';
import { Upload } from 'lucide-react';

interface EmptyStageProps {
  onUpload: () => void;
  onPresetQuery: (q: string) => void;
  onDemo: (n: 1 | 2 | 3 | 4) => void;
}

export const EmptyStage: React.FC<EmptyStageProps> = ({ onUpload, onPresetQuery, onDemo }) => {
  return (
    <div className="absolute inset-0 sq-grid-bg sq-scan flex items-center justify-center">
      <div className="text-center px-6 max-w-lg sq-fade">
        <div className="text-[11px] sq-mono tracking-[0.28em] text-teal-300/80">SATQUERY AI</div>
        <h2 className="mt-3 text-[22px] font-medium tracking-[0.08em] text-slate-100">
          ASK YOUR SATELLITE IMAGE
        </h2>
        <p className="mt-3 text-[13px] text-slate-400 leading-relaxed">
          Upload imagery or launch a guided analysis. Natural-language queries drive detection,
          segmentation, and multitemporal change on the map.
        </p>
        <button
          type="button"
          onClick={onUpload}
          className="mt-6 inline-flex items-center gap-2 h-9 px-4 border border-teal-300/40 bg-teal-400/10 text-teal-100 text-[12px] tracking-wide hover:bg-teal-400/20"
        >
          <Upload className="w-4 h-4" />
          Upload Image
        </button>
        <div className="mt-8 text-[10px] sq-mono tracking-[0.16em] text-slate-500">TRY ASKING</div>
        <div className="mt-2 flex flex-wrap justify-center gap-2">
          {['Find all buildings', 'What changed?', 'Show new construction'].map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => onPresetQuery(q === 'What changed?' ? 'What changed between these images?' : q === 'Show new construction' ? 'Which buildings have changed?' : 'Find and locate all buildings')}
              className="text-[11px] px-2.5 py-1 border border-white/10 text-slate-300 hover:border-teal-300/40"
            >
              “{q}”
            </button>
          ))}
        </div>
        <div className="mt-8 grid grid-cols-2 gap-2 text-left">
          {[
            { n: 1 as const, t: 'Vehicle detection' },
            { n: 2 as const, t: 'Building segmentation' },
            { n: 3 as const, t: 'Change detection' },
            { n: 4 as const, t: 'Changed buildings' },
          ].map((d) => (
            <button
              key={d.n}
              type="button"
              onClick={() => onDemo(d.n)}
              className="px-3 py-2 border border-white/10 hover:border-teal-300/30 text-[11px] text-slate-300"
            >
              <span className="sq-mono text-teal-300/80 mr-2">0{d.n}</span>
              {d.t}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
