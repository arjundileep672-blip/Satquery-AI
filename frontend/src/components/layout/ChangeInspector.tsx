import React from 'react';
import type { ObjectChangeItem } from '../../types';

interface ChangeInspectorProps {
  obj: ObjectChangeItem;
  onClose: () => void;
  onZoom?: () => void;
}

export const ChangeInspector: React.FC<ChangeInspectorProps> = ({ obj, onClose, onZoom }) => {
  return (
    <div className="border border-white/10 p-3 space-y-2 sq-fade">
      <div className="flex items-center justify-between">
        <div className="text-[10px] sq-mono tracking-[0.18em] text-teal-200">{obj.object_id}</div>
        <button type="button" onClick={onClose} className="text-[10px] text-slate-500 hover:text-slate-200">
          CLOSE
        </button>
      </div>
      <Row k="CLASS" v={obj.object_class} />
      <Row k="STATUS" v={obj.status} />
      <Row k="DATE 1" v={obj.date_first_detected || '—'} />
      <Row k="DATE 2" v={obj.date_last_detected || '—'} />
      <Row
        k="AREA"
        v={
          obj.area_date2 != null
            ? `${obj.area_date2} m²`
            : obj.area_date1 != null
            ? `${obj.area_date1} m²`
            : '—'
        }
      />
      <Row
        k="LOCATION"
        v={
          obj.centroid_geo
            ? `${obj.centroid_geo.lat.toFixed(4)}° N  ${obj.centroid_geo.lon.toFixed(4)}° E`
            : obj.centroid_pixel
            ? `px ${obj.centroid_pixel[0]}, ${obj.centroid_pixel[1]}`
            : '—'
        }
      />
      <Row k="CONFIDENCE" v={obj.confidence != null ? `${Math.round(obj.confidence * 100)}%` : '—'} />
      {onZoom && (
        <button
          type="button"
          onClick={onZoom}
          className="w-full mt-1 h-7 border border-teal-300/40 text-[10px] sq-mono text-teal-200"
        >
          ZOOM TO CHANGE
        </button>
      )}
    </div>
  );
};

const Row: React.FC<{ k: string; v: string }> = ({ k, v }) => (
  <div className="flex justify-between gap-3 text-[11px]">
    <span className="sq-mono text-slate-500 tracking-wide">{k}</span>
    <span className="text-slate-200 text-right">{v}</span>
  </div>
);
