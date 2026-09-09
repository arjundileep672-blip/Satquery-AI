# SatQuery AI — Model Cards & Provenance Registry

This document records the provenance, licensing, datasets, tasks, input formats, and technical limitations of all AI models integrated into SatQuery AI for SIH 2026 problem statement SIH26167.

All models operate locally and offline once checkpoints are downloaded via `python scripts/download_models.py`.

---

## 1. YOLO12n — Generic Object Detector

- **Model Identifier**: YOLO12n (`yolo12n.pt` / fallback: `yolo11n.pt`)
- **Source**: Ultralytics ([github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics))
- **Primary Checkpoint Path**: `models/yolo/yolo12n.pt`
- **Fallback Checkpoint Path**: `checkpoints/yolo11n.pt`
- **Pretrained Dataset**: COCO (Common Objects in Context, 80 classes)
- **Primary Task**: Generic object detection, object counting, initial bounding box localization
- **Software License**: AGPL-3.0 / Commercial License Available from Ultralytics
- **Input Format**: 3-channel RGB/BGR image, uint8, resized to 640x640 during inference
- **Expected Resolution**: Standard terrestrial photography resolution.
- **Recognized Classes**: 80 COCO classes (car, truck, bus, airplane, person, boat, train, etc.)
- **Known Limitations**:
  - **CRITICAL PROVENANCE NOTE**: Trained on terrestrial COCO photography, **NOT** satellite or aerial remote sensing imagery.
  - Sub-pixel or nadir satellite objects at low spatial resolution (GSD > 1m) will have degraded detection rates.
  - Do not present this model as a satellite-specialized detector. Use `YOLO26n-OBB` for aerial targets.

---

## 2. YOLO26n-OBB — Remote Sensing Oriented Object Detector

- **Model Identifier**: YOLO26n-OBB (`yolo26n-obb.pt` / fallback: `yolov8n-obb.pt`)
- **Source**: Ultralytics Assets Release v8.3.0 ([github.com/ultralytics/assets](https://github.com/ultralytics/assets))
- **Primary Checkpoint Path**: `models/yolo26_obb/yolo26n-obb.pt`
- **Fallback Checkpoint Path**: `checkpoints/yolov8n-obb.pt`
- **Pretrained Dataset**: DOTA-v1 (Dataset for Object deTection in Aerial images, v1.0)
- **Primary Task**: Oriented bounding box (OBB) detection for aerial and satellite remote sensing
- **Software License**: AGPL-3.0 (Ultralytics) / DOTA Academic License for weights
- **Input Format**: 3-channel RGB/BGR image, uint8, inference size 1024x1024
- **Expected Resolution**: Aerial and satellite imagery (GSD 0.1m - 1.0m)
- **DOTA-v1 Classes (15)**:
  1. `plane`
  2. `ship`
  3. `storage-tank`
  4. `baseball-diamond`
  5. `tennis-court`
  6. `basketball-court`
  7. `ground-track-field`
  8. `harbor`
  9. `bridge`
  10. `large-vehicle`
  11. `small-vehicle`
  12. `helicopter`
  13. `roundabout`
  14. `soccer-ball-field`
  15. `swimming-pool`
- **Known Limitations**:
  - **CRITICAL PROVENANCE NOTE**: DOTA-v1 does **NOT** contain generic building footprints or urban rooftops.
  - **DO NOT** claim or describe this model as a building detector. Use SpaceNet / `building_detector.py` for buildings.

---

## 3. SpaceNet Rio — Building Footprint Segmentation Model

- **Model Identifier**: SpaceNet Rio Building Model (`spacenet_rio.pt`)
- **Architecture**: U-Net with ResNet/VGG encoder (`models/spacenet/unet_arch/model.py`)
- **Source**: `harshinde/spacenet-models` / SpaceNet Challenge Rio Baseline
- **Primary Checkpoint Path**: `models/spacenet/spacenet_rio.pt`
- **Pretrained Dataset**: SpaceNet 1 & 7 Rio de Janeiro (high-resolution satellite imagery)
- **Primary Task**: Building footprint delineation, segmentation, polygon extraction
- **Software License**: Apache-2.0
- **Input Format**: 3-channel RGB image normalized to [0, 1], resized to 512x512
- **Expected Resolution**: 0.3m – 0.5m GSD satellite imagery (WorldView-2/3)
- **Classes**: Binary mask (`0`: background, `1`: building footprint)
- **Known Limitations**:
  - Semantic segmentation only; does not separate dense adjoining buildings natively.
  - Connected component analysis is used to extract individual instances.
  - If checkpoint is unavailable, gracefully falls back to YOLO building class detection + SAM 2.1 mask refinement.

---

## 4. SAM 2.1 Hiera Tiny — Prompted Instance Segmenter

- **Model Identifier**: SAM 2.1 Hiera Tiny (`sam2.1_hiera_tiny.pt`)
- **Source**: Meta AI Research ([github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2))
- **Primary Checkpoint Path**: `models/sam2/sam2.1_hiera_tiny.pt`
- **Pretrained Dataset**: SA-1B (Segment Anything 1-Billion masks)
- **Primary Task**: Zero-shot promptable segmentation (box, point, mask prompt refinement)
- **Software License**: Apache-2.0
- **Input Format**: 3-channel RGB image, uint8, arbitrary resolution (internal image encoder)
- **Expected Resolution**: Agnostic; handles multi-scale features via Hiera backbone
- **Classes**: Class-agnostic promptable segmentation
- **Known Limitations**:
  - High memory usage during batch inference. Mitigated by lazy loading (only instantiated when needed).
  - Requires bounding box or point prompt from upstream detector.
  - Falls back to OpenCV GrabCut if PyTorch or GPU memory is constrained.

---

## 5. ChangeFormer V6 — Bi-Temporal Change Detector

- **Model Identifier**: ChangeFormer V6 (`changeformer_levir.pt`)
- **Architecture**: Siamese Mix-Transformer Encoder + Difference Feature Decoder (`models/changeformer/changeformer_arch/ChangeFormer.py`)
- **Source**: wgcban/ChangeFormer ([github.com/wgcban/ChangeFormer](https://github.com/wgcban/ChangeFormer))
- **Primary Checkpoint Path**: `models/changeformer/changeformer_levir.pt`
- **Pretrained Dataset**: LEVIR-CD (Large-scale Building Change Detection Dataset)
- **Primary Task**: Bi-temporal urban and building change detection
- **Software License**: MIT License
- **Input Format**: Two co-registered 3-channel RGB images (T1, T2) of identical spatial dimensions
- **Expected Resolution**: 0.5m GSD aerial/satellite imagery
- **Output Classes**: Binary change mask (`0`: unchanged, `1`: changed)
- **Known Limitations**:
  - Output is a raw change score / segmentation logit, **NOT** a calibrated probability.
  - Highly sensitive to image misregistration. Must be preceded by ORB + RANSAC alignment.
  - Falls back to Otsu-thresholded pixel differencing with morphological cleanup if checkpoint is absent.

---

## 6. Local VLM / LLM (Ollama / Mistral)

- **Model Identifier**: Mistral-7B-Instruct / LLaVA via Ollama
- **Source**: Ollama ([ollama.com](https://ollama.com)) / Mistral AI
- **Primary Endpoint**: Local REST daemon (`http://localhost:11434`)
- **Fallback Adapter**: `MockVLMAdapter` (deterministic grounded synthetic responses)
- **Primary Task**: Natural language query intent extraction and explainable answer synthesis grounded exclusively in CV evidence
- **Software License**: Apache-2.0 (Mistral 7B) / MIT (Ollama)
- **Input Format**: Structured text prompts with JSON-constrained output and technical metadata
- **Known Limitations**:
  - Must NOT be used to invent visual measurements or fabricate detections.
  - All numbers, counts, and overlap percentages come directly from the CV and geospatial pipelines.
