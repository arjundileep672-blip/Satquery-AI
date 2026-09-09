import React, { useState, useRef, useEffect } from 'react';
import { Crosshair, Expand, LocateFixed, Minus, Plus } from 'lucide-react';
import type {
  AnalysisResult,
  CompareMode,
  DetectedObject,
  ImageMetadata,
  LayerVisibilityState,
} from '../types';

interface ImageViewerProps {
  image1Src: string | null;
  image2Src: string | null;
  metadata?: ImageMetadata | null;
  result?: AnalysisResult | null;
  layers: LayerVisibilityState;
  scale: number;
  setScale: React.Dispatch<React.SetStateAction<number>>;
  selectedObjectId?: string | null;
  onSelectObject?: (obj: any | null) => void;
  compareMode: CompareMode;
  onCompareMode: (m: CompareMode) => void;
  sceneLabel?: string;
  sceneDate?: string;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  image1Src,
  image2Src,
  metadata,
  result,
  layers,
  scale,
  setScale,
  selectedObjectId,
  onSelectObject,
  compareMode,
  onCompareMode,
  sceneLabel,
  sceneDate,
}) => {
  const [position, setPosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoveredObject, setHoveredObject] = useState<DetectedObject | null>(null);
  const [swipe, setSwipe] = useState(52);
  const [blinkShowT2, setBlinkShowT2] = useState(false);
  const [isSwipeDrag, setIsSwipeDrag] = useState(false);


  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!image2Src && (compareMode === 'secondary' || compareMode === 'side_by_side' || compareMode === 'swipe' || compareMode === 'blink')) {
      onCompareMode('primary');
    }
  }, [image2Src, compareMode, onCompareMode]);

  useEffect(() => {
    if (compareMode !== 'blink' || !image2Src) return;
    const id = window.setInterval(() => setBlinkShowT2((v) => !v), 700);
    return () => window.clearInterval(id);
  }, [compareMode, image2Src]);

  // Reset viewport when primary image changes
  useEffect(() => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  }, [image1Src, setScale]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!image1Src || isSwipeDrag) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    setScale((prev) => Math.min(Math.max(prev * factor, 0.25), 8));
  };

  const detections = result?.detections || [];
  const masks = result?.masks || [];
  const changes = result?.changes || [];

  const imgRef = useRef<HTMLImageElement>(null);
  const [imgNaturalSize, setImgNaturalSize] = useState<{ w: number; h: number }>({ w: 512, h: 512 });

  const handleImgLoad = () => {
    if (imgRef.current) {
      setImgNaturalSize({
        w: imgRef.current.naturalWidth,
        h: imgRef.current.naturalHeight,
      });
    }
  };

  // Render SVG Vector Overlays — coordinates are in pixel space from backend
  const renderOverlays = (_isSecondaryView: boolean = false) => {
    if (!result) return null;
    const { w, h } = imgNaturalSize;

    return (
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="absolute inset-0 w-full h-full pointer-events-none"
        style={{ opacity: layers.opacity }}
        preserveAspectRatio="none"
      >
        <defs>
          {/* Change Mask Striped Pattern */}
          <pattern id="changeStripe" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="8" stroke="#ef4444" strokeWidth="2.5" />
          </pattern>
        </defs>

        {/* 1. Segmentation Masks & Building Footprints */}
        {(layers.segmentation || layers.buildings) &&
          masks.map((maskItem: any, idx: number) => {
            // Support both geometry.coordinates[0] (GeoJSON) and direct polygon [[x,y],...]
            let ring: [number, number][] | null = null;
            try {
              const geom = maskItem.geometry;
              if (geom?.coordinates?.[0]) {
                ring = geom.coordinates[0] as [number, number][];
              } else if (Array.isArray(maskItem.polygon) && maskItem.polygon.length >= 3) {
                ring = maskItem.polygon as [number, number][];
              }
            } catch {
              return null;
            }

            if (!ring || ring.length < 3) return null;
            const pointsStr = ring.map((pt) => `${pt[0]},${pt[1]}`).join(' ');

            return (
              <polygon
                key={`mask-${idx}`}
                points={pointsStr}
                fill="#8b5cf6"
                fillOpacity="0.35"
                stroke="#a78bfa"
                strokeWidth="2"
                strokeDasharray="4 2"
                className="transition-all hover:fill-opacity-60"
              />
            );
          })}

        {/* 2. Change Mask Overlay */}
        {layers.changeMask &&
          changes.map((chg: any, idx: number) => {
            let ring: [number, number][] | null = null;
            try {
              const coords = chg.geometry_geojson?.coordinates?.[0] || chg.coords?.[0];
              if (coords) ring = coords as [number, number][];
            } catch {
              return null;
            }
            if (!ring || ring.length < 3) return null;
            const pointsStr = ring.map((pt: any) => `${pt[0]},${pt[1]}`).join(' ');

            return (
              <g key={`change-${idx}`}>
                <polygon
                  points={pointsStr}
                  fill="url(#changeStripe)"
                  fillOpacity="0.6"
                  stroke="#ef4444"
                  strokeWidth="2.5"
                />
                <polygon
                  points={pointsStr}
                  fill="#ef4444"
                  fillOpacity="0.25"
                />
              </g>
            );
          })}

        {/* 3. Dynamic Multitemporal Object-Level Changes */}
        {(layers.buildings || layers.changedBuildings || layers.multitemporalChanges !== false) &&
          result.multitemporal_report?.object_changes.map((obj: any, idx: number) => {
            const filter = layers.parameterFilter || 'all';
            if (filter !== 'all') {
              const cls = obj.object_class?.toLowerCase() || '';
              if (filter === 'buildings' && !cls.includes('building')) return null;
              if (filter === 'roads' && !cls.includes('road')) return null;
              if (filter === 'water' && !cls.includes('water')) return null;
              if (filter === 'vegetation' && !cls.includes('veg')) return null;
            }

            const isSelected = selectedObjectId === obj.object_id;
            const st = obj.status?.toUpperCase() || '';

            // Status colors: NEW=Emerald, REMOVED=Rose, EXPANDED=Amber, CONTRACTED=Cyan, UNCHANGED=Slate
            let strokeColor = '#64748b';
            let fillColor = 'rgba(100, 116, 139, 0.2)';
            if (st === 'NEW') {
              strokeColor = '#10b981';
              fillColor = 'rgba(16, 185, 129, 0.35)';
            } else if (st === 'REMOVED') {
              strokeColor = '#f43f5e';
              fillColor = 'rgba(244, 63, 94, 0.35)';
            } else if (st === 'EXPANDED') {
              strokeColor = '#f59e0b';
              fillColor = 'rgba(245, 158, 11, 0.35)';
            } else if (st === 'CONTRACTED') {
              strokeColor = '#06b6d4';
              fillColor = 'rgba(6, 182, 212, 0.35)';
            }

            if (isSelected) {
              strokeColor = '#38bdf8';
              fillColor = 'rgba(56, 189, 248, 0.5)';
            }

            const poly = obj.polygon_pixel;
            const hasPoly = Array.isArray(poly) && poly.length >= 3;
            const pointsStr = hasPoly ? poly.map((pt: any) => `${pt[0]},${pt[1]}`).join(' ') : null;

            const bb = obj.bbox_pixel || [0, 0, 0, 0];
            const bw = bb[2] - bb[0];
            const bh = bb[3] - bb[1];

            return (
              <g
                key={`multi-obj-${obj.object_id || idx}`}
                className="cursor-pointer pointer-events-auto"
                onClick={() => onSelectObject && onSelectObject(isSelected ? null : obj)}
              >
                {pointsStr ? (
                  <polygon
                    points={pointsStr}
                    fill={fillColor}
                    stroke={strokeColor}
                    strokeWidth={isSelected ? 3.5 : 2}
                  />
                ) : (
                  <rect
                    x={bb[0]}
                    y={bb[1]}
                    width={bw}
                    height={bh}
                    fill={fillColor}
                    stroke={strokeColor}
                    strokeWidth={isSelected ? 3.5 : 2}
                    rx="2"
                  />
                )}
                {/* Badge Label */}
                <text
                  x={bb[0] + 2}
                  y={Math.max(12, bb[1] - 3)}
                  fill={strokeColor}
                  fontSize="10"
                  fontFamily="monospace"
                  fontWeight="bold"
                  className="select-none"
                >
                  {obj.object_id} ({st})
                </text>
              </g>
            );
          })}


        {layers.grid && (
          <g opacity="0.35">
            {Array.from({ length: 12 }).map((_, i) => (
              <line
                key={`gv-${i}`}
                x1={(w / 12) * i}
                y1={0}
                x2={(w / 12) * i}
                y2={h}
                stroke="#94a3b8"
                strokeWidth="0.6"
              />
            ))}
            {Array.from({ length: 12 }).map((_, i) => (
              <line
                key={`gh-${i}`}
                x1={0}
                y1={(h / 12) * i}
                x2={w}
                y2={(h / 12) * i}
                stroke="#94a3b8"
                strokeWidth="0.6"
              />
            ))}
          </g>
        )}

        {/* 5. Detections: Bounding Boxes (Horizontal & Oriented) */}
        {(layers.detections || layers.boundingBoxes !== false) &&
          layers.detections &&
          detections.map((obj, idx) => {
            // bbox_pixel: [x1, y1, x2, y2] - set by backend conversion
            const bboxPixel: [number, number, number, number] = obj.bbox_pixel
              ? obj.bbox_pixel
              : Array.isArray(obj.bbox)
              ? obj.bbox
              : obj.bbox
              ? [obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2]
              : [0, 0, 0, 0];
            const [x1, y1, x2, y2] = bboxPixel;
            const bw = x2 - x1;
            const bh = y2 - y1;
            const obb = obj.oriented_bbox;

            const isHovered = hoveredObject?.id === obj.id;

            return (
              <g
                key={obj.id || idx}
                className="cursor-pointer pointer-events-auto"
                onMouseEnter={() => setHoveredObject(obj)}
                onMouseLeave={() => setHoveredObject(null)}
              >
                {/* Horizontal Bounding Box */}
                <rect
                  x={x1}
                  y={y1}
                  width={bw}
                  height={bh}
                  fill={isHovered ? 'rgba(16, 185, 129, 0.25)' : 'rgba(16, 185, 129, 0.1)'}
                  stroke={isHovered ? '#34d399' : '#10b981'}
                  strokeWidth={isHovered ? 2.5 : 1.75}
                  rx="3"
                />

                {/* Oriented BBox if present and toggled */}
                {layers.oriented && obb && (
                  <rect
                    x={x1}
                    y={y1}
                    width={obb.width_px || bw}
                    height={obb.height_px || bh}
                    fill="none"
                    stroke="#06b6d4"
                    strokeWidth="2"
                    strokeDasharray="3 2"
                    transform={`rotate(${obb.angle_degrees || 0}, ${x1 + bw / 2}, ${y1 + bh / 2})`}
                  />
                )}

                {/* Label Badge */}
                <g transform={`translate(${x1}, ${Math.max(y1 - 18, 2)})`}>
                  <rect
                    x="0"
                    y="0"
                    width={Math.max(obj.label.length * 6.5 + (obj.confidence ? 38 : 10), 50)}
                    height="16"
                    fill="rgba(15, 23, 42, 0.85)"
                    stroke={isHovered ? '#34d399' : '#059669'}
                    strokeWidth="1"
                    rx="3"
                  />
                  <text
                    x="4"
                    y="12"
                    fill="#a7f3d0"
                    fontSize="9"
                    fontFamily="monospace"
                    fontWeight="bold"
                  >
                    {obj.label}
                    {obj.confidence !== undefined && ` ${(obj.confidence * 100).toFixed(0)}%`}
                  </text>
                </g>
              </g>
            );
          })}
      </svg>
    );
  };


  const enterFullscreen = () => {
    containerRef.current?.requestFullscreen?.();
  };

  const showT2 = compareMode === 'secondary' || (compareMode === 'blink' && blinkShowT2);
  const showSwipe = compareMode === 'swipe' && !!image2Src;
  const showSplit = compareMode === 'side_by_side' && !!image2Src;
  const showChangeMap = compareMode === 'change_map';

  return (
    <div className="relative w-full h-full min-h-0 bg-[#070b12] overflow-hidden flex flex-col select-none">
      {image1Src && (
        <div className="absolute top-3 left-3 z-20 sq-glass px-3 py-2 pointer-events-none max-w-sm">
          <div className="text-[9px] sq-mono tracking-[0.2em] text-teal-300">SATELLITE ANALYSIS</div>
          <div className="text-[13px] text-slate-100 mt-0.5">{sceneLabel || metadata?.filename || 'Loaded scene'}</div>
          <div className="text-[10px] sq-mono text-slate-400 mt-0.5">
            {sceneDate ? `Image Date: ${sceneDate}` : null}
            {metadata ? ` · ${metadata.width}×${metadata.height} px` : ''}
            {metadata?.crs ? ` · ${metadata.crs}` : ''}
          </div>
        </div>
      )}

      {image2Src && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20 pointer-events-auto flex sq-glass p-0.5 text-[10px] sq-mono">
          {([
            ['swipe', 'Swipe'],
            ['side_by_side', 'Side-by-Side'],
            ['blink', 'Blink'],
            ['change_map', 'Change Map'],
          ] as const).map(([id, label]) => (
            <button
              key={id}
              type="button"
              onClick={() => onCompareMode(id)}
              className={`px-2.5 py-1 ${compareMode === id ? 'bg-teal-400/20 text-teal-100' : 'text-slate-400 hover:text-slate-200'}`}
            >
              {label}
            </button>
          ))}
        </div>
      )}

      <div className="absolute top-3 right-3 z-20 flex flex-col gap-1 pointer-events-auto">
        <MapBtn title="Zoom in" onClick={() => setScale((s) => Math.min(s * 1.25, 8))}>
          <Plus className="w-3.5 h-3.5" />
        </MapBtn>
        <MapBtn title="Zoom out" onClick={() => setScale((s) => Math.max(s / 1.25, 0.25))}>
          <Minus className="w-3.5 h-3.5" />
        </MapBtn>
        <MapBtn title="Recenter" onClick={() => { setScale(1); setPosition({ x: 0, y: 0 }); }}>
          <LocateFixed className="w-3.5 h-3.5" />
        </MapBtn>
        <MapBtn title="Fullscreen" onClick={enterFullscreen}>
          <Expand className="w-3.5 h-3.5" />
        </MapBtn>
        <MapBtn title="Crosshair">
          <Crosshair className="w-3.5 h-3.5" />
        </MapBtn>
      </div>

      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={(e) => {
          handleMouseMove(e);
          if (isSwipeDrag && containerRef.current) {
            const r = containerRef.current.getBoundingClientRect();
            setSwipe(Math.min(92, Math.max(8, ((e.clientX - r.left) / r.width) * 100)));
          }
        }}
        onMouseUp={() => {
          handleMouseUp();
          setIsSwipeDrag(false);
        }}
        onMouseLeave={() => {
          handleMouseUp();
          setIsSwipeDrag(false);
        }}
        onWheel={handleWheel}
        className={`w-full flex-1 flex items-center justify-center ${
          isDragging ? 'cursor-grabbing' : 'cursor-grab'
        } overflow-hidden relative sq-grid-bg`}
      >
        {image1Src ? (
          <div
            style={{
              transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
              transition: isDragging ? 'none' : 'transform 0.08s ease-out',
            }}
            className="relative max-w-full max-h-full"
          >
            {showSplit ? (
              <div className="flex items-stretch gap-px bg-white/10">
                <div className="relative">
                  <div className="absolute top-2 left-2 z-10 px-2 py-0.5 sq-glass text-[10px] sq-mono text-teal-200">T1</div>
                  <img src={image1Src} alt="T1" className="max-h-[calc(100vh-160px)] max-w-[42vw] object-contain" />
                </div>
                <div className="relative">
                  <div className="absolute top-2 left-2 z-10 px-2 py-0.5 sq-glass text-[10px] sq-mono text-amber-200">T2</div>
                  <img src={image2Src || ''} alt="T2" className="max-h-[calc(100vh-160px)] max-w-[42vw] object-contain" />
                  {renderOverlays(true)}
                </div>
              </div>
            ) : showSwipe ? (
              <div className="relative inline-block">
                <img
                  ref={imgRef}
                  src={image1Src}
                  alt="T1"
                  onLoad={handleImgLoad}
                  className="max-h-[calc(100vh-140px)] max-w-[min(1200px,92vw)] object-contain"
                />
                <div className="absolute inset-0 overflow-hidden" style={{ width: `${swipe}%` }}>
                  <img
                    src={image2Src || ''}
                    alt="T2"
                    className="h-full w-auto max-w-none object-cover"
                    style={{ width: '100%', minWidth: `${10000 / swipe}%`, maxHeight: 'none' }}
                  />
                </div>
                <div
                  className="absolute top-0 bottom-0 w-px bg-teal-200 cursor-ew-resize z-10"
                  style={{ left: `${swipe}%` }}
                  onMouseDown={(e) => {
                    e.stopPropagation();
                    setIsSwipeDrag(true);
                  }}
                >
                  <div className="absolute top-1/2 -translate-y-1/2 -left-2 w-4 h-8 bg-teal-300/80" />
                </div>
                {renderOverlays(false)}
              </div>
            ) : (
              <div className="relative inline-block">
                <img
                  ref={imgRef}
                  src={showT2 && image2Src ? image2Src : image1Src}
                  alt="Satellite scene"
                  onLoad={handleImgLoad}
                  className="max-h-[calc(100vh-140px)] max-w-[min(1400px,96vw)] object-contain"
                />
                {(compareMode !== 'secondary' || showChangeMap) && renderOverlays(showT2)}
              </div>
            )}
          </div>
        ) : null}

        {hoveredObject && (
          <div className="absolute bottom-16 left-4 z-30 sq-glass p-3 text-xs sq-mono text-slate-200 pointer-events-none space-y-1">
            <div className="text-teal-200 font-medium">{hoveredObject.label}</div>
            {hoveredObject.confidence !== undefined && (
              <div>{(hoveredObject.confidence * 100).toFixed(1)}% conf</div>
            )}
          </div>
        )}

        {result?.multitemporal_report && (compareMode === 'change_map' || layers.changedBuildings) && (
          <div className="absolute bottom-12 right-4 z-30 sq-glass p-2.5 text-[10px] sq-mono space-y-1">
            <div className="text-slate-500 tracking-[0.14em] mb-1">CHANGE MAP</div>
            <Legend c="bg-emerald-500" t="NEW" />
            <Legend c="bg-amber-500" t="EXPANDED" />
            <Legend c="bg-rose-500" t="REMOVED" />
            <Legend c="bg-cyan-500" t="CONTRACTED" />
            <Legend c="bg-slate-500" t="UNCHANGED" />
          </div>
        )}
      </div>

      <div className="h-8 px-3 flex items-center justify-between text-[10px] sq-mono text-slate-500 border-t border-white/10">
        <div className="flex items-center gap-3">
          <span className="w-5 h-5 border border-white/20 flex items-center justify-center text-slate-300">N</span>
          <span>Scale {Math.round(scale * 100)}%</span>
          {detections.length > 0 && <span className="text-teal-300">{detections.length} objects</span>}
          {changes.length > 0 && <span className="text-amber-300">{changes.length} change zones</span>}
        </div>
        <div>Imagery · © SatQuery AI</div>
      </div>
    </div>
  );
};

const MapBtn: React.FC<{ title: string; onClick?: () => void; children: React.ReactNode }> = ({
  title,
  onClick,
  children,
}) => (
  <button
    type="button"
    title={title}
    onClick={onClick}
    className="w-8 h-8 sq-glass flex items-center justify-center text-slate-300 hover:text-teal-200"
  >
    {children}
  </button>
);

const Legend: React.FC<{ c: string; t: string }> = ({ c, t }) => (
  <div className="flex items-center gap-1.5 text-slate-300">
    <span className={`w-2.5 h-2.5 ${c}`} />
    {t}
  </div>
);
