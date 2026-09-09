import React from 'react';
import { Terminal, Sparkles, AlertCircle } from 'lucide-react';

interface QueryBoxProps {
  query: string;
  setQuery: (q: string) => void;
  hasImage1: boolean;
  hasImage2: boolean;
  isLoading: boolean;
}

export const QueryBox: React.FC<QueryBoxProps> = ({
  query,
  setQuery,
  hasImage1,
  hasImage2,
  isLoading,
}) => {
  const quickQueries = [
    { label: 'Detect objects', query: 'Detect all objects in this satellite scene' },
    { label: 'Find buildings', query: 'Find and locate all buildings' },
    { label: 'Count vehicles', query: 'Detect and count all vehicles' },
    { label: 'Segment buildings', query: 'Find and segment all buildings' },
    { label: 'What changed?', query: 'What changed between these images?', requiresTwoImages: true },
    { label: 'Which buildings changed?', query: 'Which buildings have changed?', requiresTwoImages: true },
    { label: 'Calculate changed area', query: 'Calculate changed area between these images', requiresTwoImages: true },
  ];

  const isChangeQuery =
    query.toLowerCase().includes('change') ||
    query.toLowerCase().includes('between these') ||
    query.toLowerCase().includes('difference');

  const showMissingImage2Warning = isChangeQuery && hasImage1 && !hasImage2;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm space-y-3">
      <div className="flex items-center justify-between">
        <label htmlFor="satquery-input" className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
          <Terminal className="w-4 h-4 text-emerald-400" />
          <span>Ask SatQuery AI:</span>
        </label>
        <span className="text-[11px] font-mono text-slate-500">Natural Language Multimodal Query</span>
      </div>

      <div className="relative">
        <input
          id="satquery-input"
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={
            hasImage1
              ? 'Ask a question or enter an analysis command (e.g. "Which buildings have changed?")...'
              : 'Please upload an image above before submitting query...'
          }
          disabled={!hasImage1 || isLoading}
          className="w-full bg-slate-950 border border-slate-700/80 rounded-lg px-4 py-3 text-sm md:text-base text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-400 focus:ring-2 focus:ring-emerald-400/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-inner"
        />
      </div>

      {/* Validation warning when change query is entered with only 1 image */}
      {showMissingImage2Warning && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-amber-950/30 border border-amber-600/40 text-amber-300 text-xs animate-fade-in">
          <AlertCircle className="w-4 h-4 text-amber-400 shrink-0" />
          <span>
            <strong>Two images needed:</strong> Change detection requires both Image 1 and Image 2. Please upload Image 2 above.
          </span>
        </div>
      )}

      {/* Quick Example Query Buttons */}
      <div className="pt-1">
        <div className="flex items-center gap-1.5 text-[11px] text-slate-400 mb-2">
          <Sparkles className="w-3.5 h-3.5 text-amber-400" />
          <span className="font-medium">Quick Query Presets:</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {quickQueries.map((item, idx) => {
            const isSelected = query.trim() === item.query;
            return (
              <button
                key={idx}
                type="button"
                onClick={() => setQuery(item.query)}
                disabled={isLoading || !hasImage1}
                className={`text-xs px-2.5 py-1 rounded-md border transition-all ${
                  isSelected
                    ? 'bg-emerald-600/30 text-emerald-300 border-emerald-500/60 font-medium'
                    : 'bg-slate-950 hover:bg-slate-800 text-slate-300 border-slate-800 hover:border-slate-700'
                } disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1`}
              >
                <span>{item.label}</span>
                {item.requiresTwoImages && (
                  <span className="text-[10px] text-cyan-400/80 font-mono">2×img</span>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
