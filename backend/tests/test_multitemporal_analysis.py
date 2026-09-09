"""
Unit & Integration Tests for Multitemporal Change Parameter Analysis
===================================================================
Tests parameter extraction across all 5 domains, object tracking/status,
NDVI band checks, zero-baseline percentage handling, and transition matrices.
"""

import numpy as np
import pytest

from app.core.multitemporal_analysis import MultitemporalAnalyzer
from app.core.query_router import TaskType, route_query, extract_target_parameter
from app.core.orchestrator import get_orchestrator


def _create_synthetic_scene(
    w: int = 256,
    h: int = 256,
    num_buildings: int = 5,
    veg_intensity: int = 120,
    water_present: bool = True,
) -> np.ndarray:
    """Create a synthetic satellite-like 3-channel BGR image."""
    img = np.full((h, w, 3), 100, dtype=np.uint8)  # Grayish bare ground

    # Vegetation zone (greenish)
    if veg_intensity > 0:
        img[: h // 2, : w // 2, 0] = 30   # Blue
        img[: h // 2, : w // 2, 1] = veg_intensity  # Green
        img[: h // 2, : w // 2, 2] = 40   # Red

    # Water zone (blueish)
    if water_present:
        img[h // 2 :, : w // 3, 0] = 180  # Blue
        img[h // 2 :, : w // 3, 1] = 120  # Green
        img[h // 2 :, : w // 3, 2] = 20   # Red

    # Buildings (white/high-contrast rectangular boxes)
    for i in range(num_buildings):
        bx = 50 + (i * 35) % (w - 60)
        by = h // 2 + 20 + (i * 25) % (h // 2 - 40)
        bw, bh = 20, 20
        img[by : by + bh, bx : bx + bw] = [230, 230, 230]

    return img


class TestMultitemporalAnalyzer:
    def setup_method(self):
        self.analyzer = MultitemporalAnalyzer()

    def test_date_resolution(self):
        # 1. From filenames
        meta = [
            {"filename": "sentinel2_2024-01-15_t1.tif"},
            {"filename": "sentinel2_2026-01-20_t2.tif"},
        ]
        dates = self.analyzer.resolve_dates(meta)
        assert dates[0] == "2024-01-15"
        assert dates[1] == "2026-01-20"

        # 2. From query
        meta_empty = [{}, {}]
        dates_q = self.analyzer.resolve_dates(
            meta_empty, query="What changed between 2023 and 2025?"
        )
        assert "2023" in dates_q[0]
        assert "2025" in dates_q[1]

        # 3. Fallback
        dates_fb = self.analyzer.resolve_dates([None, None])
        assert len(dates_fb) == 2
        assert "Date 1" in dates_fb[0]

    def test_parameter_extraction_rgb_disclaimer(self):
        img = _create_synthetic_scene(w=256, h=256, num_buildings=4)
        params = self.analyzer.extract_parameters(
            image=img,
            geo_meta=None,
            date_label="2024-01-15",
            is_multispectral=False,
        )

        assert params["date"] == "2024-01-15"
        assert params["num_buildings"] >= 1
        assert params["builtup_hectares"] > 0
        assert params["vegetation_pct"] > 0
        assert params["water_pct"] > 0
        assert params["road_length_km"] >= 0

        # Mandatory Constitution Rule: Never calculate NDVI from RGB without NIR
        assert params["has_true_nir"] is False
        assert params["mean_ndvi"] is None
        assert "True NDVI cannot be calculated because the required spectral bands are unavailable" in params["ndvi_note"]

    def test_parameter_extraction_multispectral_nir(self):
        img = _create_synthetic_scene(w=128, h=128)
        nir = np.full((128, 128), 200, dtype=np.uint8)

        params = self.analyzer.extract_parameters(
            image=img,
            geo_meta=None,
            date_label="2024-01-15",
            is_multispectral=True,
            nir_channel=nir,
        )

        assert params["has_true_nir"] is True
        assert params["mean_ndvi"] is not None
        assert "True NDVI calculated" in params["ndvi_note"]

    def test_object_tracking_and_zero_baseline(self):
        img1 = _create_synthetic_scene(num_buildings=2)
        img2 = _create_synthetic_scene(num_buildings=5)  # 3 new buildings

        p1 = self.analyzer.extract_parameters(img1, None, "2024-01-15")
        p2 = self.analyzer.extract_parameters(img2, None, "2026-01-20")

        objects = self.analyzer.track_objects(p1, p2, None)
        assert len(objects) >= 2

        new_objects = [o for o in objects if o.status == "NEW"]
        assert len(new_objects) > 0

        # Zero-baseline requirement
        for no in new_objects:
            assert no.area_date1 == 0.0
            assert no.percentage_change is None
            assert "baseline was zero" in (no.percentage_change_text or "")

    def test_transition_matrix(self):
        img1 = _create_synthetic_scene(w=128, h=128, num_buildings=1, veg_intensity=150)
        img2 = _create_synthetic_scene(w=128, h=128, num_buildings=6, veg_intensity=0)

        p1 = self.analyzer.extract_parameters(img1, None, "2024-01-15")
        p2 = self.analyzer.extract_parameters(img2, None, "2026-01-20")

        transitions = self.analyzer.compute_transition_matrix(p1, p2)
        assert isinstance(transitions, list)
        if transitions:
            first = transitions[0]
            assert first.previous_class != first.current_class
            assert first.area_changed_pixels > 0
            assert first.percentage_of_total_change > 0

    def test_comparisons_and_change_ranking(self):
        img1 = _create_synthetic_scene(num_buildings=2)
        img2 = _create_synthetic_scene(num_buildings=4)

        p1 = self.analyzer.extract_parameters(img1, None, "2024-01-15")
        p2 = self.analyzer.extract_parameters(img2, None, "2026-01-20")

        objects = self.analyzer.track_objects(p1, p2, None)
        comparisons, rankings = self.analyzer.compare_parameters(p1, p2, objects)

        assert len(comparisons) > 0
        bld_count_comp = next((c for c in comparisons if c.parameter == "Building Count"), None)
        assert bld_count_comp is not None
        assert bld_count_comp.status in ("INCREASED", "DECREASED", "UNCHANGED")

        # Ranking items exist
        assert len(rankings) <= 5
        if rankings:
            assert rankings[0].rank == 1
            assert rankings[0].raw_magnitude >= 0


class TestMultitemporalQueryRouting:
    def test_query_router_multitemporal(self):
        queries = [
            "What parameters changed between these images?",
            "How much did the built-up area change?",
            "How many new buildings appeared?",
            "How much vegetation was lost?",
            "Which roads were newly constructed?",
            "Which parameters changed the most?",
            "Compare building density between 2024 and 2026",
            "Show me the land-cover transition matrix",
        ]
        for q in queries:
            task, rationale = route_query(q)
            assert task == TaskType.MULTITEMPORAL_ANALYSIS, f"Failed for '{q}': got {task}"

    def test_extract_target_parameter(self):
        assert extract_target_parameter("How much did built-up area change?") == "built_up"
        assert extract_target_parameter("How many new buildings appeared?") == "buildings"
        assert extract_target_parameter("Which roads were newly constructed?") == "roads"
        assert extract_target_parameter("How much vegetation was lost?") == "vegetation"
        assert extract_target_parameter("How did the water area expand?") == "water"


class TestE2EMultitemporalOrchestrator:
    def test_orchestrator_multitemporal_run(self):
        img1 = _create_synthetic_scene(w=128, h=128, num_buildings=2)
        img2 = _create_synthetic_scene(w=128, h=128, num_buildings=5)

        import cv2
        _, b1 = cv2.imencode(".png", img1)
        _, b2 = cv2.imencode(".png", img2)

        orch = get_orchestrator()
        result = orch.analyze(
            image1_bytes=b1.tobytes(),
            query="What parameters changed between January 2024 and January 2026?",
            image2_bytes=b2.tobytes(),
            metadata={"filename": "site_2024-01-15.png"},
            request_id="test_req_multi_001",
        )

        assert result.task == TaskType.MULTITEMPORAL_ANALYSIS.value
        assert result.multitemporal_report is not None
        rep = result.multitemporal_report
        assert len(rep.dates) == 2
        assert len(rep.parameters) > 0
        assert len(rep.object_changes) > 0
        assert len(rep.limitations_and_disclaimers) > 0
        assert "2024" in rep.dates[0] or "Date 1" in rep.dates[0]
        assert result.statistics is not None
