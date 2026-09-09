import React from 'react';
import { Send, Loader2, MessageSquare, Terminal } from 'lucide-react';

interface QueryPanelProps {
  query: string;
  setQuery: (q: string) => void;
  onAnalyze: () => void;
  isLoading: boolean;
  hasImage: boolean;
}

export const QueryPanel: React.FC<QueryPanelProps> = ({
  query,
  setQuery,
  onAnalyze,
  isLoading,
  hasImage,
}) => {
  const sampleQueries = [
    'What objects are visible in this satellite image?',
    'Is there a river or water channel in the western part?',
    'Describe the scene land cover composition and infrastructure.',
    'Are there commercial vessels or container ships berthed?',
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!isLoading && hasImage && query.trim()) {
      onAnalyze();
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm">
      <div className="flex items-center justify-between mb-2.5">
        <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
          <Terminal className="w-4 h-4 text-emerald-400" />
          <span>Ask SatQuery</span>
        </h3>
        <span className="text-[11px] text-slate-500 font-mono">Natural Language Query</span>
      </div>

      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={
              hasImage
                ? 'Ask a question or enter an analysis query...'
                : 'Upload an image above before querying...'
            }
            disabled={!hasImage || isLoading}
            className="w-full bg-slate-950 border border-slate-700/80 rounded-lg pl-3 pr-24 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500/40 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          />
          <button
            type="submit"
            disabled={!hasImage || !query.trim() || isLoading}
            className="absolute right-1.5 top-1.5 bottom-1.5 px-4 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-md flex items-center gap-1.5 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Analyzing</span>
              </>
            ) : (
              <>
                <Send className="w-3.5 h-3.5" />
                <span>Analyze</span>
              </>
            )}
          </button>
        </div>

        {/* Quick Suggestion Chips */}
        <div>
          <div className="flex items-center gap-1.5 text-[11px] text-slate-400 mb-1.5">
            <MessageSquare className="w-3 h-3 text-emerald-400" />
            <span>Suggested Queries:</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {sampleQueries.map((sq, i) => (
              <button
                key={i}
                type="button"
                onClick={() => setQuery(sq)}
                disabled={isLoading || !hasImage}
                className="text-[11px] text-left px-2.5 py-1 rounded bg-slate-950/60 hover:bg-slate-800 text-slate-300 border border-slate-800 hover:border-slate-700 transition-colors disabled:opacity-50"
              >
                {sq}
              </button>
            ))}
          </div>
        </div>
      </form>
    </div>
  );
};
