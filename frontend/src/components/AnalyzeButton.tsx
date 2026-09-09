import React from 'react';
import { Play, Loader2, Sparkles } from 'lucide-react';

interface AnalyzeButtonProps {
  onAnalyze: () => void;
  isLoading: boolean;
  disabled: boolean;
}

export const AnalyzeButton: React.FC<AnalyzeButtonProps> = ({
  onAnalyze,
  isLoading,
  disabled,
}) => {
  return (
    <button
      type="button"
      onClick={onAnalyze}
      disabled={disabled || isLoading}
      className={`w-full py-3.5 px-6 rounded-xl font-bold text-base md:text-lg uppercase tracking-wider flex items-center justify-center gap-3 transition-all shadow-lg select-none ${
        disabled || isLoading
          ? 'bg-slate-800/80 text-slate-500 border border-slate-700/50 cursor-not-allowed'
          : 'bg-gradient-to-r from-emerald-600 via-teal-600 to-emerald-500 hover:from-emerald-500 hover:to-teal-500 text-white border border-emerald-400/40 shadow-emerald-950/50 hover:shadow-emerald-900/60 active:scale-[0.99] cursor-pointer'
      }`}
    >
      {isLoading ? (
        <>
          <Loader2 className="w-5 h-5 animate-spin text-emerald-200" />
          <span>Analyzing Remote Sensing Imagery...</span>
        </>
      ) : (
        <>
          <Play className="w-5 h-5 fill-current text-white" />
          <span>[ ANALYZE SATELLITE IMAGERY ]</span>
          <Sparkles className="w-4 h-4 text-emerald-200" />
        </>
      )}
    </button>
  );
};
