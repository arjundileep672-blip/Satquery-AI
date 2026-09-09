import React from 'react';
import type { LayerVisibilityState } from '../types';

interface LayerControlsProps {
  layers: LayerVisibilityState;
  setLayers: React.Dispatch<React.SetStateAction<LayerVisibilityState>>;
  hasDetections: boolean;
  hasOriented: boolean;
  hasMasks: boolean;
  hasChanges: boolean;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onResetZoom: () => void;
  scale: number;
  compact?: boolean;
}

export const LayerControls: React.FC<LayerControlsProps> = ({
  layers,
  setLayers,
  hasDetections,
  hasOriented,
  hasMasks,
  hasChanges,
  compact,
}) => {
  const toggle = (key: keyof LayerVisibilityState) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const rows: { key: keyof LayerVisibilityState; label: string; enabled: boolean }[] = [
    { key: 'original', label: 'Satellite Imagery', enabled: true },
    { key: 'buildings', label: 'Buildings', enabled: hasMasks || hasChanges },
    { key: 'roads', label: 'Roads', enabled: hasChanges },
    { key: 'vegetation', label: 'Vegetation', enabled: hasChanges },
    { key: 'water', label: 'Water', enabled: hasChanges },
    { key: 'changeMask', label: 'Change Detection', enabled: hasChanges },
    { key: 'detections', label: 'Bounding Boxes', enabled: hasDetections },
    { key: 'oriented', label: 'Oriented Boxes', enabled: hasOriented },
    { key: 'segmentation', label: 'Segmentation', enabled: hasMasks },
    { key: 'grid', label: 'Geographic Grid', enabled: true },
    { key: 'changedBuildings', label: 'Multitemporal Objects', enabled: hasChanges },
  ];

  return (
    <div className={`sq-glass ${compact ? 'p-2' : 'p-3'} min-w-[200px]`}>
      <div className="text-[9px] sq-mono tracking-[0.18em] text-slate-500 mb-2">LAYERS</div>
      <div className="space-y-1">
        {rows.map((row) => (
          <label
            key={String(row.key)}
            className={`flex items-center gap-2 text-[11px] ${
              row.enabled ? 'text-slate-300' : 'text-slate-600'
            }`}
          >
            <input
              type="checkbox"
              disabled={!row.enabled}
              checked={Boolean(layers[row.key])}
              onChange={() => {
                toggle(row.key);
                if (row.key === 'roads') setLayers((p) => ({ ...p, parameterFilter: p.roads ? 'all' : 'roads' }));
                if (row.key === 'vegetation')
                  setLayers((p) => ({ ...p, parameterFilter: p.vegetation ? 'all' : 'vegetation' }));
                if (row.key === 'water') setLayers((p) => ({ ...p, parameterFilter: p.water ? 'all' : 'water' }));
                if (row.key === 'buildings')
                  setLayers((p) => ({ ...p, parameterFilter: p.buildings ? p.parameterFilter : 'buildings' }));
              }}
              className="accent-teal-300"
            />
            {row.label}
          </label>
        ))}
      </div>
      <div className="mt-3 pt-2 border-t border-white/10">
        <div className="text-[9px] sq-mono text-slate-500 mb-1">OPACITY {Math.round(layers.opacity * 100)}%</div>
        <input
          type="range"
          min="0.1"
          max="1"
          step="0.05"
          value={layers.opacity}
          onChange={(e) => setLayers((p) => ({ ...p, opacity: parseFloat(e.target.value) }))}
          className="sq-range w-full"
        />
      </div>
    </div>
  );
};
