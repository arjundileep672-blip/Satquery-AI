/**
 * SatQuery AI - Centralized API Client
 * SIH 2026 - ISRO - SIH26167
 *
 * Provides typed functions for multi-modal remote sensing analysis,
 * object detection, segmentation, and change detection.
 */

import type {
  AnalysisResult,
  FleetMetricsResponse,
  HealthStatus,
  ModelMetricsResponse,
} from '../types';

// Centralized API Base: configurable via VITE_API_URL, fallback to /api/v1 (proxied by Vite)
export const API_BASE = (import.meta.env.VITE_API_URL || '/api/v1').replace(/\/$/, '');

/**
 * Normalizes backend response so UI never crashes on missing optional fields.
 */
function normalizeAnalysisResult(data: any): AnalysisResult {
  return {
    request_id: data.request_id || `req_${Math.random().toString(36).slice(2, 10)}`,
    answer: data.answer || 'Analysis complete.',
    operation: data.operation || data.task || 'visual_question_answering',
    task: data.task || (data.operation ? data.operation.replace(/_/g, ' ') : 'Remote Sensing Analysis'),
    confidence: data.confidence !== undefined ? data.confidence : null,
    detections: Array.isArray(data.detections) ? data.detections : [],
    masks: Array.isArray(data.masks) ? data.masks : [],
    changes: Array.isArray(data.changes) ? data.changes : [],
    multitemporal_report: data.multitemporal_report || null,
    statistics: data.statistics && typeof data.statistics === 'object' ? data.statistics : {},
    visualizations: Array.isArray(data.visualizations) ? data.visualizations : [],
    evidence: Array.isArray(data.evidence) ? data.evidence : [],
    tools_used: Array.isArray(data.tools_used) ? data.tools_used : [],
    models_used: Array.isArray(data.models_used) ? data.models_used : [],
    analysis_trace: Array.isArray(data.analysis_trace) ? data.analysis_trace : [],
    metadata: data.metadata || {
      filename: 'image',
      format: 'PNG',
      width: 512,
      height: 512,
      channels: 3,
      size_bytes: 0,
      is_geotiff: false,
    },
    warnings: Array.isArray(data.warnings) ? data.warnings : [],
  };
}

/**
 * Parses user-friendly error messages from backend responses without exposing raw stack traces.
 */
async function parseErrorResponse(res: Response, fallbackMessage: string): Promise<string> {
  try {
    const errorJson = await res.json();
    if (errorJson.detail) {
      if (typeof errorJson.detail === 'string') {
        return errorJson.detail;
      }
      if (Array.isArray(errorJson.detail)) {
        return errorJson.detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ');
      }
    }
  } catch {
    // Response was not JSON
  }
  return `${fallbackMessage} (Status: ${res.status})`;
}

/**
 * Checks backend health, version, demo mode status, GPU availability, and model registry.
 */
export async function healthCheck(): Promise<HealthStatus> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) {
      throw new Error(`Health check returned status ${res.status}`);
    }
    return await res.json();
  } catch (err: any) {
    throw new Error(err.message || 'AI backend is currently unreachable.');
  }
}

/**
 * Primary Analysis integration:
 * Sends image1, optional image2, and natural language query.
 */
export async function analyzeImage(
  image1: File,
  query: string,
  image2?: File | null
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('image1', image1);
  formData.append('image', image1); // Alias for compatibility with single-image backend
  formData.append('query', query);

  if (image2) {
    formData.append('image2', image2);
  }

  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const detail = await parseErrorResponse(res, 'Analysis request failed');
    throw new Error(detail);
  }

  const data = await res.json();
  return normalizeAnalysisResult(data);
}

/**
 * Dedicated Object Detection endpoint
 */
export async function detectObjects(image: File, query?: string): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('image1', image);
  formData.append('image', image);
  if (query) {
    formData.append('query', query);
  }

  const res = await fetch(`${API_BASE}/detect`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const detail = await parseErrorResponse(res, 'Object detection failed');
    throw new Error(detail);
  }

  const data = await res.json();
  return normalizeAnalysisResult(data);
}

/**
 * Dedicated Segmentation endpoint
 */
export async function segmentImage(image: File, query?: string): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('image1', image);
  formData.append('image', image);
  if (query) {
    formData.append('query', query);
  }

  const res = await fetch(`${API_BASE}/segment`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const detail = await parseErrorResponse(res, 'Segmentation failed');
    throw new Error(detail);
  }

  const data = await res.json();
  return normalizeAnalysisResult(data);
}

/**
 * Dedicated Bi-Temporal Change Detection endpoint
 */
export async function detectChanges(
  image1: File,
  image2: File,
  query?: string
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('image1', image1);
  formData.append('image2', image2);
  if (query) {
    formData.append('query', query);
  }

  const res = await fetch(`${API_BASE}/change-detection`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const detail = await parseErrorResponse(res, 'Change detection failed');
    throw new Error(detail);
  }

  const data = await res.json();
  return normalizeAnalysisResult(data);
}

/**
 * Dedicated Changed Objects endpoint
 */
export async function detectChangedObjects(
  image1: File,
  image2: File,
  query?: string
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('image1', image1);
  formData.append('image2', image2);
  if (query) {
    formData.append('query', query);
  }

  const res = await fetch(`${API_BASE}/changed-objects`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const detail = await parseErrorResponse(res, 'Changed object detection failed');
    throw new Error(detail);
  }

  const data = await res.json();
  return normalizeAnalysisResult(data);
}

/**
 * Fetches EuroSAT scene classification performance metrics & confusion matrix
 */
export async function fetchModelMetrics(): Promise<ModelMetricsResponse> {
  const res = await fetch(`${API_BASE}/metrics/eurosat`);
  if (!res.ok) {
    const detail = await parseErrorResponse(res, 'Failed to fetch model evaluation metrics');
    throw new Error(detail);
  }
  return await res.json();
}

export async function fetchFleetMetrics(): Promise<FleetMetricsResponse> {
  const res = await fetch(`${API_BASE}/metrics/models`);
  if (!res.ok) {
    const detail = await parseErrorResponse(res, 'Failed to fetch fleet model metrics');
    throw new Error(detail);
  }
  return await res.json();
}

// Re-export as satquery default object
export default {
  healthCheck,
  analyzeImage,
  detectObjects,
  segmentImage,
  detectChanges,
  detectChangedObjects,
  fetchModelMetrics,
  fetchFleetMetrics,
};

