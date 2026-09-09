// ── Patch-Based Multi-Label Scene Classification ──────────────────────────────

export interface PatchLabel {
  class_name: string;
  coverage_pct: number;
  tile_count: number;
  mean_confidence: number;
  rank: number;
}

export interface PatchSceneClassification {
  labels: PatchLabel[];
  total_tiles: number;
  tile_size: number;
  stride: number;
  inference_ms: number;
  model: string;
  available: boolean;
}

export interface ImageMetadata {
  filename: string;
  format: string;
  width: number;
  height: number;
  channels: number;
  size_bytes: number;
  is_geotiff: boolean;
  crs?: string | null;
  transform?: number[] | null;
  bounds?: number[] | null;
  nodata?: number | null;
  dtype?: string | null;
  band_count?: number | null;
  tags?: Record<string, any>;
}

export interface Evidence {
  evidence_id: string;
  source_asset: string;
  observation: string;
  spatial_extent?: number[] | null;
  confidence?: number | null;
  model_tool_used: string;
  processing_steps: string[];
  measurements?: Record<string, any> | null;
  geometry?: Record<string, any> | null;
}

export interface OrientedBoundingBox {
  center_geo?: [number, number];
  width_px: number;
  height_px: number;
  angle_degrees: number;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DetectedObject {
  id: string;
  label: string;
  confidence?: number;
  bbox?: BoundingBox | [number, number, number, number];
  bbox_pixel?: [number, number, number, number]; // [x1, y1, x2, y2]
  bbox_geo?: [number, number, number, number];
  oriented_bbox?: OrientedBoundingBox;
  area_m2?: number;
  type?: string;
  properties?: Record<string, any>;
}

export interface ChangeRegion {
  region_id: string;
  change_type: string;
  area_m2?: number;
  confidence?: number;
  geometry_geojson?: {
    type: string;
    coordinates: number[][][] | number[][][][];
  };
  coords?: number[][][];
  description?: string;
}

export interface GeoJSONFeature {
  type: string;
  id?: string;
  geometry: {
    type: string;
    coordinates: any;
  };
  properties?: Record<string, any>;
}

export interface ParameterComparison {
  parameter: string;
  category: 'built_up' | 'roads' | 'vegetation' | 'water' | 'land_cover' | string;
  date_1: string;
  date_2: string;
  value_1?: number | null;
  value_2?: number | null;
  absolute_change?: number | null;
  percentage_change?: number | null;
  percentage_change_text: string;
  unit: string;
  status: 'INCREASED' | 'DECREASED' | 'UNCHANGED' | 'NOT_AVAILABLE' | string;
  confidence?: number | null;
  notes?: string | null;
}

export interface ObjectChangeItem {
  object_id: string;
  object_class: string;
  status: 'NEW' | 'REMOVED' | 'EXPANDED' | 'CONTRACTED' | 'UNCHANGED' | 'UNCERTAIN' | string;
  date_first_detected: string;
  date_last_detected: string;
  area_date1?: number | null;
  area_date2?: number | null;
  area_change?: number | null;
  percentage_change?: number | null;
  percentage_change_text?: string | null;
  centroid_pixel?: [number, number] | null;
  centroid_geo?: { lat: number; lon: number } | null;
  bbox_pixel?: [number, number, number, number] | null;
  polygon_pixel?: number[][] | null;
  perimeter_px?: number | null;
  shape_compactness?: number | null;
  region_sector?: string | null;
  confidence?: number | null;
}

export interface TransitionMatrixRow {
  previous_class: string;
  current_class: string;
  area_changed_m2?: number | null;
  area_changed_hectares?: number | null;
  area_changed_pixels: number;
  percentage_of_total_change: number;
}

export interface ChangeRankingItem {
  rank: number;
  parameter: string;
  category: string;
  change_summary: string;
  direction: 'increase' | 'decrease' | 'stable';
  magnitude_type: 'percentage' | 'area' | 'count' | 'length';
  raw_magnitude: number;
}

export interface TimeSeriesStep {
  date: string;
  parameters: Record<string, any>;
}

export interface MultitemporalReport {
  executive_summary: string;
  dates: string[];
  registration_quality: {
    status: string;
    inlier_ratio?: number;
    inliers?: number;
    matches_good?: number;
    registration_ms?: number;
  };
  parameters: ParameterComparison[];
  object_changes: ObjectChangeItem[];
  transition_matrix: TransitionMatrixRow[];
  change_ranking: ChangeRankingItem[];
  time_series: TimeSeriesStep[];
  limitations_and_disclaimers: string[];
  spatial_summary: Record<string, any>;
}

export interface AnalysisResult {
  request_id: string;
  answer: string;
  operation: string;
  task?: string;
  confidence?: number | null;
  detections?: DetectedObject[];
  masks?: GeoJSONFeature[] | any[];
  changes?: ChangeRegion[] | any[];
  multitemporal_report?: MultitemporalReport | null;
  statistics?: Record<string, any>;
  visualizations?: Array<{
    id?: string;
    name?: string;
    layer?: string;
    type?: string;
    count?: number;
  }>;
  evidence: Evidence[];
  tools_used: string[];
  models_used: string[];
  analysis_trace: string[];
  metadata: ImageMetadata;
  warnings: string[];
}

export interface HealthStatus {
  status: string;
  service: string;
  version: string;
  demo_mode: boolean;
  gpu_available?: boolean;
  models_status?: string;
  tools_registered: string[];
  offline_mode?: boolean;
  device?: string;
  models?: Record<string, string>;
}

export type NavView =
  | 'overview'
  | 'analyze'
  | 'compare'
  | 'detections'
  | 'change'
  | 'layers'
  | 'history'
  | 'settings';

export type CompareMode = 'primary' | 'secondary' | 'side_by_side' | 'swipe' | 'blink' | 'change_map';

export interface HistoryEntry {
  id: string;
  query: string;
  at: string;
  result: AnalysisResult;
  hasImage2: boolean;
}

export interface LayerVisibilityState {
  original: boolean;
  detections: boolean;
  oriented: boolean;
  segmentation: boolean;
  changeMask: boolean;
  buildings: boolean;
  changedBuildings: boolean;
  roads?: boolean;
  vegetation?: boolean;
  water?: boolean;
  grid?: boolean;
  boundingBoxes?: boolean;
  multitemporalChanges?: boolean;
  parameterFilter?: 'all' | 'buildings' | 'roads' | 'vegetation' | 'water' | 'land_cover';
  opacity: number; // 0 to 1
}


export interface PerClassMetric {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ModelMetricsResponse {
  model: string;
  architecture: string;
  dataset: string;
  overall_accuracy: number;
  top_3_accuracy: number;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  weighted_precision: number;
  weighted_recall: number;
  weighted_f1: number;
  total_samples: number;
  class_names: string[];
  confusion_matrix: number[][];
  normalized_confusion_matrix: number[][];
  per_class_metrics: Record<string, PerClassMetric>;
  checkpoint_epoch?: number;
  note?: string;
}

export interface FleetModelMetric {
  id: string;
  name: string;
  short_name: string;
  task: string;
  dataset: string;
  metric: string;
  score: number;
  secondary_metric?: string | null;
  secondary_score?: number | null;
  provenance: string;
  local_eval: boolean;
  registry_key: string;
}

export interface SmokeTestResult {
  id: string;
  name: string;
  passed: boolean;
  latency_ms: number | null;
  load_status: string;
  output_summary: string;
  error: string | null;
}

export interface SmokeSuiteResult {
  run_at: string;
  device: string;
  tests: SmokeTestResult[];
  pytest?: {
    exit_code: number;
    passed: number;
    failed: number;
    skipped: number;
    total: number;
    latency_ms: number;
    summary_line: string;
  };
}

export interface FleetMetricsResponse {
  disclaimer: string;
  models: FleetModelMetric[];
  smoke?: SmokeSuiteResult | null;
}

