# SatQuery AI - Frontend & WebGIS Cockpit
**Smart India Hackathon 2026 — ISRO — SIH26167**

Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis Through Text Queries.

---

## 🛰️ Architecture & Features

- **React 19 + Vite + Tailwind CSS v4** geospatial cockpit.
- **Dual Raster Uploader**:
  - Support for PNG, JPEG, JPG, TIFF, and GeoTIFF rasters.
  - Drag-and-drop, dimension auto-detection, and preview thumbnails.
  - Image 1 (Baseline) + optional Image 2 (Target / Multi-Temporal Change).
- **Interactive Multi-Layer Raster Viewer**:
  - Zoom, pan, fit-to-screen, and reset.
  - Overlays: Original, Detections (Horizontal BBoxes), Oriented BBoxes (OBB with rotation), Segmentation Masks, Change Mask, Building Footprints, Changed Buildings.
  - Opacity adjustment slider (0% to 100%).
  - Synchronized side-by-side Before/After comparison for bi-temporal change detection.
  - Interactive tooltips on hover: Class, Confidence, Oriented Angle, Derived Area, and Coordinates.
- **Four SIH Hero Demo Presets**:
  - **Demo 1**: Vehicle Detection (`Detect all vehicles`) -> Detections + Oriented BBoxes + Counts.
  - **Demo 2**: Building Footprint Segmentation (`Find and segment all buildings`) -> Segmentation masks + Building footprints + Hectares.
  - **Demo 3**: Bi-Temporal Change Detection (`What changed between these images?`) -> Before/After comparison + Change Map + Changed percentage (13.4%).
  - **Demo 4**: Changed Buildings Flagship (`Which buildings have changed?`) -> Highlighted altered buildings (7 altered zones) + Change explanation.
- **Explainable AI Result Panel**:
  - Large natural-language analytical report.
  - Execution Milestones & Trace.
  - Transparent Model Transparency section (YOLO26n-OBB, SAM 2.1, SpaceNet 7, ChangeFormer, ORB + RANSAC).
- **Backend Health Monitor**:
  - Live indicator for Backend Online/Offline, GPU status, and model readiness.

---

## 🚀 Quickstart & Commands

### 1. Start Backend API
```powershell
# From project root:
cd backend
python app/main.py
# Running on http://127.0.0.1:8000
```

### 2. Start Frontend Dev Server
```powershell
# In another terminal, from frontend directory:
cd frontend
npm.cmd run dev
# Running on http://localhost:5173
```

### 3. Production Build
```powershell
cd frontend
npm.cmd run build
```

---

## ⚙️ Environment Variables

Configure in `frontend/.env`:

```env
# By default, uses the Vite proxy to 127.0.0.1:8000:
VITE_API_URL=/api/v1
```

For direct backend URL without proxy:
```env
VITE_API_URL=http://127.0.0.1:8000/api/v1
```

---

## 🧪 Testing Backend & Endpoints

Run backend test suite:
```powershell
python -m pytest
```
All 25 unit and integration tests covering VQA, Object Detection, Segmentation, and Change Detection pass cleanly.
