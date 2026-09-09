import React, { useState } from 'react';
import { Cpu, ChevronDown, ChevronUp, Database, Activity } from 'lucide-react';

interface ModelInfoProps {
  modelsUsed: string[];
  onOpenMetrics?: () => void;
}

export const ModelInfo: React.FC<ModelInfoProps> = ({ modelsUsed, onOpenMetrics }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  // Model catalog with verified attributes
  const modelCatalog: Record<
    string,
    { name: string; role: string; dataset?: string; description: string }
  > = {
    'EuroSAT Scene Classifier': {
      name: 'EuroSAT EfficientNet-B0',
      role: 'Land-Use / Land-Cover Classification',
      dataset: 'EuroSAT (Sentinel-2 10-class)',
      description: 'Fine-tuned deep convolutional network for classifying Earth observation scenes.',
    },
    'YOLO26n-OBB': {
      name: 'YOLO26n-OBB',
      role: 'Remote-Sensing Oriented Detection',
      dataset: 'DOTA-v1 / DIOR-R',
      description: 'Generates angle-aware oriented bounding boxes for maritime and aerial objects.',
    },
    'YOLO12n': {
      name: 'YOLO12n',
      role: 'Fast Object Detection',
      description: 'Standard horizontal bounding box extraction across optical channels.',
    },
    'SpaceNet 7': {
      name: 'SpaceNet 7',
      role: 'Building Footprint Extraction',
      dataset: 'SpaceNet Multi-Temporal Urban',
      description: 'Deep neural network trained for polygonizing building structures.',
    },
    'SAM 2.1': {
      name: 'SAM 2.1',
      role: 'Promptable Segmentation',
      description: 'Zero-shot foundation model for boundary delineation and zone masking.',
    },
    'ChangeFormer': {
      name: 'ChangeFormer',
      role: 'Multi-Temporal Change Detection',
      dataset: 'LEVIR-CD / WHU-CD',
      description: 'Bi-temporal transformer comparing feature differences across aligned timestamps.',
    },
    'ORB + RANSAC': {
      name: 'ORB + RANSAC',
      role: 'Image Co-Registration',
      description: 'Calculates affine/homography transforms to align rasters with sub-pixel precision.',
    },
    'mock-rs-vlm-v1': {
      name: 'Mock Remote Sensing VLM',
      role: 'Vision-Language Understanding',
      description: 'Domain-adapted multimodal reasoning engine for qualitative question answering.',
    },
  };

  const displayModels = modelsUsed;

  return (
    <div className="sq-glass overflow-hidden">
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-slate-850 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-emerald-400" />
          <span className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Models & Adapters Deployed ({displayModels.length})
          </span>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span className="font-mono text-[11px]">
            {displayModels.join(' · ')}
          </span>
          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      {isExpanded && (
        <div className="p-4 pt-1 border-t border-white/10 bg-transparent space-y-3 sq-fade">
          {displayModels.length === 0 && (
            <p className="text-xs text-slate-500">No models recorded yet. Run an analysis to populate the pipeline.</p>
          )}
          <p className="text-xs text-slate-400 leading-relaxed">
            SatQuery AI leverages a modular multi-agent pipeline routing each query to specialized remote sensing computer vision networks:
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
            {displayModels.map((mName, idx) => {
              const matched =
                modelCatalog[mName] || {
                  name: mName,
                  role: 'Analytical Model',
                  description: 'Active pipeline tool registered in the SatQuery AI system.',
                };

              return (
                <div
                  key={idx}
                  className="bg-slate-900/90 border border-slate-800 rounded-lg p-3 text-xs space-y-1"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-slate-200 font-mono text-sm text-emerald-300">
                      {matched.name}
                    </span>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                      {matched.role}
                    </span>
                  </div>
                  <p className="text-slate-400 text-[11px] leading-relaxed">
                    {matched.description}
                  </p>
                  {matched.dataset && (
                    <div className="flex items-center gap-1 text-[10px] text-slate-500 font-mono pt-1">
                      <Database className="w-3 h-3 text-cyan-400" />
                      <span>Trained on: {matched.dataset}</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {onOpenMetrics && (
            <div className="pt-2 border-t border-slate-800/80 flex justify-end">
              <button
                type="button"
                onClick={onOpenMetrics}
                className="px-3 py-1.5 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 text-xs font-medium flex items-center gap-2 transition-colors cursor-pointer"
              >
                <Activity className="w-3.5 h-3.5 text-emerald-400" />
                <span>View Performance Metrics & Confusion Matrix</span>
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
