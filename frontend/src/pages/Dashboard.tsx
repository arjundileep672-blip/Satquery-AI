import React, { useState, useEffect, useRef } from 'react';
import { AlertCircle, Presentation } from 'lucide-react';
import { ImageViewer } from '../components/ImageViewer';
import { LayerControls } from '../components/LayerControls';
import { ModelEvaluationModal } from '../components/ModelEvaluationModal';
import { Sidebar } from '../components/layout/Sidebar';
import { TopBar } from '../components/layout/TopBar';
import { CommandBar } from '../components/layout/CommandBar';
import { InsightPanel } from '../components/layout/InsightPanel';
import { EmptyStage } from '../components/layout/EmptyStage';
import { PipelineOverlay } from '../components/layout/PipelineOverlay';
import { DemoOverlay } from '../components/layout/DemoOverlay';
import { healthCheck, analyzeImage } from '../api/satquery';
import type {
  AnalysisResult,
  CompareMode,
  HealthStatus as HealthStatusType,
  HistoryEntry,
  ImageMetadata,
  LayerVisibilityState,
  NavView,
  ObjectChangeItem,
} from '../types';

export const Dashboard: React.FC = () => {
  const [health, setHealth] = useState<HealthStatusType | null>(null);
  const [isMetricsModalOpen, setIsMetricsModalOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [view, setView] = useState<NavView>('overview');
  const [insightOpen, setInsightOpen] = useState(true);
  const [layersOpen, setLayersOpen] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const uploadRef = useRef<HTMLInputElement>(null);

  const [image1, setImage1] = useState<File | null>(null);
  const [image2, setImage2] = useState<File | null>(null);
  const [previewUrl1, setPreviewUrl1] = useState<string | null>(null);
  const [previewUrl2, setPreviewUrl2] = useState<string | null>(null);
  const [dim1, setDim1] = useState<{ width: number; height: number } | null>(null);
  const [dim2, setDim2] = useState<{ width: number; height: number } | null>(null);

  const [query, setQuery] = useState('Detect all vehicles');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [selectedObject, setSelectedObject] = useState<ObjectChangeItem | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [compareMode, setCompareMode] = useState<CompareMode>('primary');

  const [scale, setScale] = useState(1);
  const [layers, setLayers] = useState<LayerVisibilityState>({
    original: true,
    detections: true,
    oriented: true,
    segmentation: true,
    changeMask: true,
    buildings: true,
    changedBuildings: true,
    roads: false,
    vegetation: false,
    water: false,
    grid: false,
    boundingBoxes: true,
    opacity: 0.85,
  });

  const fetchHealth = async () => {
    try {
      const data = await healthCheck();
      setHealth(data);
    } catch {
      setHealth(null);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  useEffect(() => {
    if (typeof window !== 'undefined' && window.innerWidth < 1024) {
      setSidebarCollapsed(true);
      setInsightOpen(false);
    }
  }, []);

  const extractDimensions = (file: File): Promise<{ width: number; height: number }> => {
    return new Promise((resolve) => {
      const img = new Image();
      const url = URL.createObjectURL(file);
      img.onload = () => {
        resolve({ width: img.naturalWidth || 512, height: img.naturalHeight || 512 });
        URL.revokeObjectURL(url);
      };
      img.onerror = () => {
        resolve({ width: 512, height: 512 });
        URL.revokeObjectURL(url);
      };
      img.src = url;
    });
  };

  const handleSelectImage1 = async (file: File) => {
    setImage1(file);
    setPreviewUrl1(URL.createObjectURL(file));
    setDim1(await extractDimensions(file));
    setError(null);
    setView('analyze');
  };

  const handleSelectImage2 = async (file: File) => {
    setImage2(file);
    setPreviewUrl2(URL.createObjectURL(file));
    setDim2(await extractDimensions(file));
    setError(null);
  };

  const handleRemoveImage1 = () => {
    setImage1(null);
    setPreviewUrl1(null);
    setDim1(null);
    setResult(null);
  };

  const handleRemoveImage2 = () => {
    setImage2(null);
    setPreviewUrl2(null);
    setDim2(null);
  };

  const loadDemoFile = async (path: string, filename: string, mime: string): Promise<File> => {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`Could not fetch demo file ${path}`);
    const blob = await res.blob();
    return new File([blob], filename, { type: mime });
  };

  const handleLaunchHeroDemo = async (demoNumber: 1 | 2 | 3 | 4) => {
    setError(null);
    setResult(null);
    try {
      if (demoNumber === 1) {
        const f1 = await loadDemoFile('/data/demo/sample_harbor_optical.png', 'sample_harbor_optical.png', 'image/png');
        await handleSelectImage1(f1);
        handleRemoveImage2();
        setQuery('Detect all vehicles');
      } else if (demoNumber === 2) {
        const f1 = await loadDemoFile('/data/demo/sample_harbor_optical.png', 'sample_harbor_optical.png', 'image/png');
        await handleSelectImage1(f1);
        handleRemoveImage2();
        setQuery('Find and segment all buildings');
      } else if (demoNumber === 3) {
        const f1 = await loadDemoFile('/data/demo/image1_before.png', 'image1_before.png', 'image/png');
        const f2 = await loadDemoFile('/data/demo/image2_after.png', 'image2_after.png', 'image/png');
        await handleSelectImage1(f1);
        await handleSelectImage2(f2);
        setQuery('What changed between these images?');
        setCompareMode('swipe');
      } else {
        const f1 = await loadDemoFile('/data/demo/image1_before.png', 'image1_before.png', 'image/png');
        const f2 = await loadDemoFile('/data/demo/image2_after.png', 'image2_after.png', 'image/png');
        await handleSelectImage1(f1);
        await handleSelectImage2(f2);
        setQuery('Which buildings have changed?');
        setCompareMode('change_map');
      }
      setInsightOpen(true);
    } catch {
      setError('Unable to load demo dataset. You can manually upload an image instead.');
    }
  };

  const handleAnalyze = async () => {
    if (!image1) {
      setError('Upload primary imagery before analysis.');
      setInsightOpen(true);
      return;
    }
    const cleanQuery = query.trim();
    if (!cleanQuery) {
      setError('Enter a query.');
      return;
    }
    const isChangeQuery =
      cleanQuery.toLowerCase().includes('change') ||
      cleanQuery.toLowerCase().includes('difference') ||
      cleanQuery.toLowerCase().includes('between these');
    if (isChangeQuery && !image2) {
      setError('Change detection requires two images.');
      setInsightOpen(true);
      return;
    }

    setIsLoading(true);
    setError(null);
    setSelectedObject(null);
    try {
      const data = await analyzeImage(image1, cleanQuery, image2);
      setResult(data);
      setHistory((prev) => [
        {
          id: data.request_id,
          query: cleanQuery,
          at: new Date().toISOString(),
          result: data,
          hasImage2: !!image2,
        },
        ...prev,
      ].slice(0, 12));
      setInsightOpen(true);
      if (image2) setCompareMode((m) => (m === 'primary' ? 'change_map' : m));
    } catch (err: any) {
      setError(err.message || 'Analysis failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const onChangeView = (v: NavView) => {
    setView(v);
    if (v === 'layers') setLayersOpen(true);
    if (v === 'compare' && image2) setCompareMode('side_by_side');
    if (v === 'detections') {
      setLayers((p) => ({ ...p, detections: true, boundingBoxes: true }));
      setInsightOpen(true);
    }
    if (v === 'change') {
      setLayers((p) => ({ ...p, changeMask: true, changedBuildings: true }));
      if (image2) setCompareMode('change_map');
      setInsightOpen(true);
    }
    if (v === 'history' || v === 'settings' || v === 'analyze' || v === 'overview') {
      setInsightOpen(true);
    }
    if (typeof window !== 'undefined' && window.innerWidth < 1024) {
      setInsightOpen(true);
    }
  };

  const highlight = (filter: LayerVisibilityState['parameterFilter']) => {
    setLayers((p) => ({
      ...p,
      parameterFilter: filter || 'all',
      detections: true,
      buildings: filter === 'buildings' || filter === 'all' ? true : p.buildings,
      changeMask: true,
      changedBuildings: true,
      vegetation: filter === 'vegetation',
      roads: filter === 'roads',
      water: filter === 'water',
    }));
  };

  const currentMetadata: ImageMetadata | null = result?.metadata || null;
  const sceneDate = image1 ? new Date(image1.lastModified).toLocaleDateString() : undefined;

  const mapTools = (
    <div className="absolute top-[72px] left-3 z-20 pointer-events-auto flex gap-1">
      {['Layers', 'Compare', 'Detect', 'Segment'].map((label) => (
        <button
          key={label}
          type="button"
          onClick={() => {
            if (label === 'Layers') setLayersOpen((v) => !v);
            if (label === 'Compare') {
              setCompareMode(image2 ? 'swipe' : 'primary');
              setView('compare');
            }
            if (label === 'Detect') {
              setQuery('Detect all objects in this satellite scene');
              setView('detections');
            }
            if (label === 'Segment') setQuery('Find and segment all buildings');
          }}
          className="h-7 px-2.5 sq-glass text-[10px] sq-mono tracking-wide text-slate-300 hover:text-teal-200"
        >
          {label}
        </button>
      ))}
      <button
        type="button"
        onClick={() => setDemoMode((d) => !d)}
        className={`h-7 px-2.5 sq-glass text-[10px] sq-mono tracking-wide flex items-center gap-1 ${
          demoMode ? 'text-teal-200' : 'text-slate-400'
        }`}
      >
        <Presentation className="w-3 h-3" />
        Demo
      </button>
    </div>
  );

  return (
    <div className="h-screen w-screen overflow-hidden bg-[#070b12] text-slate-100 flex flex-col">
      <TopBar
        query={query}
        onQueryChange={setQuery}
        onSubmit={handleAnalyze}
        hasImage={!!image1}
        isLoading={isLoading}
        health={health}
        onOpenMetrics={() => setIsMetricsModalOpen(true)}
        onOpenSettings={() => onChangeView('settings')}
      />

      <div className="flex-1 flex min-h-0">
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggle={() => setSidebarCollapsed((c) => !c)}
          view={view}
          onChangeView={onChangeView}
          health={health}
        />

        <div className="flex-1 relative min-w-0 min-h-0">
          {!previewUrl1 && (
            <EmptyStage
              onUpload={() => uploadRef.current?.click()}
              onPresetQuery={setQuery}
              onDemo={handleLaunchHeroDemo}
            />
          )}

          {previewUrl1 && (
            <>
              <ImageViewer
                image1Src={previewUrl1}
                image2Src={previewUrl2}
                metadata={currentMetadata}
                result={result}
                layers={layers}
                scale={scale}
                setScale={setScale}
                selectedObjectId={selectedObject?.object_id}
                onSelectObject={setSelectedObject}
                compareMode={compareMode}
                onCompareMode={setCompareMode}
                sceneLabel={image1?.name}
                sceneDate={sceneDate}
              />
              {mapTools}
              {layersOpen && (
                <div className="absolute top-[108px] left-3 z-30">
                  <LayerControls
                    layers={layers}
                    setLayers={setLayers}
                    hasDetections={Boolean(result?.detections && result.detections.length > 0)}
                    hasOriented={Boolean(result?.detections && result.detections.some((d) => d.oriented_bbox))}
                    hasMasks={Boolean(result?.masks && result.masks.length > 0)}
                    hasChanges={Boolean(
                      (result?.changes && result.changes.length > 0) || result?.multitemporal_report
                    )}
                    onZoomIn={() => setScale((s) => Math.min(s * 1.25, 8))}
                    onZoomOut={() => setScale((s) => Math.max(s / 1.25, 0.25))}
                    onResetZoom={() => setScale(1)}
                    scale={scale}
                    compact
                  />
                </div>
              )}
              {!demoMode && (
                <CommandBar
                  query={query}
                  setQuery={setQuery}
                  onSubmit={handleAnalyze}
                  disabled={!image1}
                  isLoading={isLoading}
                  hasImage2={!!image2}
                />
              )}
              {demoMode && result && (
                <DemoOverlay
                  result={result}
                  query={query}
                  onViewChanges={() => {
                    setLayers((p) => ({ ...p, changeMask: true, changedBuildings: true }));
                    setCompareMode(image2 ? 'change_map' : 'primary');
                  }}
                  onExit={() => setDemoMode(false)}
                />
              )}
            </>
          )}

          <PipelineOverlay active={isLoading} hasSecondImage={!!image2} />

          {error && (
            <div className="absolute top-3 right-14 z-40 sq-glass px-3 py-2 max-w-sm text-[12px] text-rose-200 flex gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              {error}
            </div>
          )}

          <input
            ref={uploadRef}
            type="file"
            accept=".png,.jpg,.jpeg,.tif,.tiff"
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.[0]) handleSelectImage1(e.target.files[0]);
            }}
          />
        </div>

        {(insightOpen || view === 'history' || view === 'settings') && (
          <div className="hidden lg:flex h-full">
            <InsightPanel
              open
              onClose={() => setInsightOpen(false)}
              view={view}
              result={result}
              lastQuery={query}
              selectedObject={selectedObject}
              onSelectObject={setSelectedObject}
              onHighlight={highlight}
              history={history}
              onRestoreHistory={(entry) => {
                setResult(entry.result);
                setQuery(entry.query);
                setInsightOpen(true);
              }}
              onOpenMetrics={() => setIsMetricsModalOpen(true)}
              modelsUsed={result?.models_used || []}
              demoMode={demoMode}
              uploadProps={{
                image1,
                image2,
                previewUrl1,
                previewUrl2,
                dimensions1: dim1,
                dimensions2: dim2,
                isLoading,
                onSelectImage1: handleSelectImage1,
                onSelectImage2: handleSelectImage2,
                onRemoveImage1: handleRemoveImage1,
                onRemoveImage2: handleRemoveImage2,
              }}
            />
          </div>
        )}

        {insightOpen && (
          <div className="lg:hidden absolute inset-x-0 bottom-0 top-12 z-40 bg-[#070b12]/80">
            <InsightPanel
              open
              onClose={() => setInsightOpen(false)}
              view={view}
              result={result}
              lastQuery={query}
              selectedObject={selectedObject}
              onSelectObject={setSelectedObject}
              onHighlight={highlight}
              history={history}
              onRestoreHistory={(entry) => {
                setResult(entry.result);
                setQuery(entry.query);
              }}
              onOpenMetrics={() => setIsMetricsModalOpen(true)}
              modelsUsed={result?.models_used || []}
              demoMode={demoMode}
              uploadProps={{
                image1,
                image2,
                previewUrl1,
                previewUrl2,
                dimensions1: dim1,
                dimensions2: dim2,
                isLoading,
                onSelectImage1: handleSelectImage1,
                onSelectImage2: handleSelectImage2,
                onRemoveImage1: handleRemoveImage1,
                onRemoveImage2: handleRemoveImage2,
              }}
            />
          </div>
        )}
      </div>

      <ModelEvaluationModal isOpen={isMetricsModalOpen} onClose={() => setIsMetricsModalOpen(false)} />
    </div>
  );
};

export default Dashboard;
