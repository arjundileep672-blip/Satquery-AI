import React, { useEffect, useState } from 'react';
import { ArrowUpRight, Sparkles } from 'lucide-react';

const HINTS = [
  'Find all buildings',
  'What changed between 2024 and 2026?',
  'Show newly constructed roads',
  'Calculate vegetation loss',
  'Detect all vehicles',
];

interface CommandBarProps {
  query: string;
  setQuery: (q: string) => void;
  onSubmit: () => void;
  disabled: boolean;
  isLoading: boolean;
  hasImage2: boolean;
}

export const CommandBar: React.FC<CommandBarProps> = ({
  query,
  setQuery,
  onSubmit,
  disabled,
  isLoading,
  hasImage2,
}) => {
  const [hint, setHint] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => setHint((h) => (h + 1) % HINTS.length), 4200);
    return () => window.clearInterval(id);
  }, []);

  const presets = [
    { q: 'Detect all objects in this satellite scene', two: false },
    { q: 'Find and locate all buildings', two: false },
    { q: 'Find and segment all buildings', two: false },
    { q: 'What changed between these images?', two: true },
    { q: 'Which buildings have changed?', two: true },
  ];

  return (
    <div className="pointer-events-none absolute left-4 right-4 bottom-4 z-30 flex justify-center">
      <div className="pointer-events-auto w-full max-w-3xl sq-glass px-3 py-2 shadow-[0_8px_40px_rgba(0,0,0,0.45)]">
        <form
          className="flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            onSubmit();
          }}
        >
          <Sparkles className="w-4 h-4 text-teal-300 shrink-0" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={disabled || isLoading}
            placeholder={`Ask about this image…  “${HINTS[hint]}”`}
            className="flex-1 bg-transparent text-[13.5px] text-slate-100 placeholder:text-slate-500 focus:outline-none disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={disabled || isLoading || !query.trim()}
            className="h-8 w-8 flex items-center justify-center bg-teal-400/15 border border-teal-300/40 text-teal-200 hover:bg-teal-400/25 disabled:opacity-40"
            title="Run analysis"
          >
            <ArrowUpRight className="w-4 h-4" />
          </button>
        </form>
        <div className="flex flex-wrap gap-1.5 mt-2">
          {presets
            .filter((p) => !p.two || hasImage2)
            .map((p) => (
              <button
                key={p.q}
                type="button"
                disabled={disabled || isLoading}
                onClick={() => setQuery(p.q)}
                className={`text-[10px] sq-mono px-2 py-0.5 border ${
                  query === p.q
                    ? 'border-teal-300/50 text-teal-200 bg-teal-400/10'
                    : 'border-white/10 text-slate-400 hover:text-slate-200'
                }`}
              >
                {p.q}
              </button>
            ))}
        </div>
      </div>
    </div>
  );
};
