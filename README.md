# SatQuery AI (SIH26167)

**Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis Through Text Queries**

SatQuery AI is an agentic geospatial intelligence platform developed for the Smart India Hackathon (SIH 2026). It enables analysts and non-specialists to analyze satellite and aerial remote-sensing imagery using natural-language queries through a modular, auditable agentic architecture.

---

## 1. Project Overview

Analyzing remote sensing (RS) imagery traditionally demands specialized GIS software (QGIS, ArcGIS, SNAP) and deep domain expertise in spectral bands, coordinate reference systems (CRS), and radar physics. SatQuery AI bridges this gap with an AI-powered assistant that interprets natural language queries, classifies domain intent, dispatches schema-validated tools, normalizes evidence, and returns explainable results accompanied by user-safe execution traces.

### Phase 1 Vertical Slice
This release implements **Phase 1** of the SatQuery AI architecture:
- **Operations Supported**:
  1. `image_understanding`: Scene-level classification, land use/land cover overview, and contextual description.
  2. `visual_question_answering`: Targeted question answering grounded in visual and spatial raster features.
- **Rasters Supported**: Standard imagery (PNG, JPEG) and geospatial imagery (TIFF, GeoTIFF) with CRS, bounds, transform, and band metadata extraction via Rasterio.
- **Model Adapter Architecture**: Provider-independent `VisionLanguageModel` interface with `MockVLMAdapter` for zero-key local testing and `GeminiVLMAdapter` for Google Gemini.
- **No Chain-of-Thought**: Replaces opaque internal reasoning with a transparent, milestone-based **Analysis Trace**.
- **Evidence Provenance**: Calibrated separation of AI interpretation from verified, measured metadata.

---

## 2. Architecture Summary

```
User / Analyst
      │
      ▼
React WebGIS Interface (Interactive Viewport, Pan/Zoom, Query Panel)
      │  (REST API / multipart/form-data)
      ▼
FastAPI Route Controller (/api/v1/analyze)
      │
      ▼
SatQuery Planner (Intent Classification & Milestone Plan Construction)
      │
      ▼
Controlled Tool Registry (Capabilities, Limits, Schema Validation)
      │
      ├─────────────────────────────┬─────────────────────────────┐
      ▼                             ▼                             ▼
ImageUnderstandingTool     VisualQuestionAnsweringTool     Future Specialized Tools
      │                             │                             (Phase 2 - 7)
      └──────────────┬──────────────┘
                     ▼
          VisionLanguageModel Interface
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     MockVLMAdapter      GeminiVLMAdapter
     (DEMO_MODE=true)   (Live Google Gemini)
                     │
                     ▼
          Evidence Normalization Layer
                     │
                     ▼
          Structured AnalysisResult
          - Request ID
          - AI Interpretation (Answer)
          - Operation & Confidence (Null if uncalibrated)
          - Measured & Derived Evidence
          - Tools & Models Used
          - Analysis Trace (Safe Milestones)
          - Image Metadata (Width, Height, Bands, CRS)
          - Operational Warnings
```

---

## 3. Requirements

- **Operating System**: Windows, Linux, or macOS
- **Python**: 3.11+ (Tested and verified on Python 3.14)
- **Node.js**: 20+ (Tested and verified on Node 24)
- **Package Managers**: `pip`, `npm`

---

## 4. Installation

### 1. Clone or Open Workspace
```bash
cd SatQuery
```

### 2. Backend Dependencies
```bash
python -m pip install -r backend/requirements.txt
```
*Core packages installed:* `fastapi`, `uvicorn`, `pydantic`, `rasterio`, `numpy`, `pillow`, `scipy`, `shapely`, `python-multipart`, `httpx`, `pytest`.

### 3. Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## 5. Environment Setup

Copy the example environment template:
```bash
copy .env.example .env
```
*(On Linux/macOS: `cp .env.example .env`)*

---

## 6. Running Backend

Start the FastAPI application:
```bash
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```
- API Base: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/api/v1/health`

---

## 7. Running Frontend

In a separate terminal, start the Vite development server:
```bash
cd frontend
npm run dev
```
- Access the web interface at: `http://localhost:5173`

---

## 8. Running Automated Tests

Run the comprehensive 14-point test suite without requiring a Gemini API key:
```bash
python -m pytest -p no:cacheprovider -v
```

All 14 tests verify:
1. Health endpoint (`GET /api/v1/health`)
2. Valid image metadata extraction (PNG & GeoTIFF)
3. Invalid image rejection (corrupted byte streams)
4. Unsupported file type rejection (415)
5. Oversized file handling (>25MB)
6. Empty query handling (400)
7. Planner intent classification
8. Controlled tool registry access
9. VQA tool execution
10. Image understanding tool execution
11. Mock VLM deterministic output
12. Evidence schema normalization
13. AnalysisResult schema validation
14. End-to-end API response contract (`POST /api/v1/analyze`)

---

## 9. Demo Mode

SatQuery AI includes a built-in `DEMO_MODE` that enables complete local execution and jury demonstration without external API keys or cloud dependencies.

When `DEMO_MODE=true` in `.env` (or by default when `GEMINI_API_KEY` is not set):
- The system automatically engages `MockVLMAdapter`.
- Domain-aware remote-sensing answers are generated deterministically.
- Uncalibrated confidence is preserved (`confidence: null`), adhering strictly to scientific integrity guidelines.

---

## 10. Gemini Configuration

To use live Google Gemini models:
1. Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
2. Edit `.env`:
   ```dotenv
   DEMO_MODE=false
   GEMINI_API_KEY=AIzaSy...
   GEMINI_MODEL=gemini-2.5-flash
   ```
3. Restart the backend server.

---

## 11. Example Queries

| Intent Category | Example Natural-Language Query | Selected Tool |
| :--- | :--- | :--- |
| **Object Identification** | *"What objects are visible in this satellite image?"* | `visual_question_answering` |
| **Presence Verification** | *"Is there a river or open water channel in the western part?"* | `visual_question_answering` |
| **Scene Understanding** | *"Describe the scene land cover composition and infrastructure."* | `image_understanding` |
| **Maritime Activity** | *"Are there commercial vessels or container ships berthed at the pier?"* | `visual_question_answering` |
| **Urban Landscape** | *"Provide an overview of the built-up structures and road network."* | `image_understanding` |

---

## 12. Current Limitations (Phase 1)

1. **No Spatial Vectorization**: Phase 1 provides qualitative textual grounding and image metadata; pixel-accurate bounding boxes and segmentation polygons are scheduled for Phase 2.
2. **Uncalibrated Confidence**: Raw vision-language model generation does not yield empirical posterior probabilities; confidence is reported as `null` with explicit scientific warnings rather than fabricating numbers.
3. **No Multi-Temporal Registration**: Single-image analysis only. Multi-date change detection and image co-registration are scheduled for Phase 3.
4. **Single-Scene Processing**: Phase 1 processes one image per request; full STAC catalog streaming and large-scene tiling are scheduled for Phase 7.

---

## 13. Recommended Phase 2 Roadmap

- **Phase 2: Object Detection & Spatial Overlays**
  - Implement `object_detection` tool with `YOLOAdapter` (YOLOv8-OBB fine-tuned on DOTA) and `GroundingDINOAdapter`.
  - Extract horizontal and oriented bounding boxes with class confidence scores.
  - Render interactive GeoJSON vector overlays on the WebGIS map viewer.
- **Phase 3: Image Registration & Bi-Temporal Change Detection**
  - Implement `register_images()` spatial co-registration tool.
  - Implement `change_detection` tool with `DifferenceBasedAdapter` and `ChangeFormer` adapter.
  - Compute change polygons with geodesic area measurements in square meters and hectares.
- **Phase 4: Optical Preprocessing & Biophysical Indices (NDVI, NDWI, MNDWI, NDBI)**
- **Phase 5: SAR Preprocessing & Scene-Adaptive Water/Structural Analysis**
- **Phase 6: Cross-Modal (Optical + SAR) Feature Fusion**
- **Phase 7: Advanced Agentic Orchestration & Enterprise PostGIS Integration**
