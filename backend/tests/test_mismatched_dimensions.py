"""
Tests for handling multi-temporal imagery with mismatched raster dimensions.
Verifies that images of differing resolutions/dimensions (e.g. (976, 1805) and (996, 1732))
are safely aligned without numpy broadcasting ValueError exceptions.
"""

import numpy as np
from vision.registration import register_images
from models.changeformer import detect_changes
from pipelines.change_detection import run_change_detection


def test_registration_with_mismatched_shapes():
    """Verify registration failure fallback resizes to match image1 dimensions."""
    # Create two images with different shapes (matching user's dimensions)
    img1 = np.random.randint(0, 255, (976, 1805, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (996, 1732, 3), dtype=np.uint8)

    reg = register_images(img1, img2)
    assert reg["aligned_image"] is not None
    assert reg["aligned_image"].shape[:2] == (976, 1805)


def test_detect_changes_with_mismatched_shapes():
    """Verify detect_changes automatically handles mismatched input shapes."""
    img1 = np.ones((976, 1805, 3), dtype=np.uint8) * 100
    img2 = np.ones((996, 1732, 3), dtype=np.uint8) * 120

    res = detect_changes(img1, img2)
    assert "change_mask" in res
    assert res["change_mask"].shape == (976, 1805)


def test_pipeline_change_detection_with_mismatched_shapes():
    """Verify end-to-end change detection pipeline handles mismatched shapes."""
    img1 = np.zeros((976, 1805, 3), dtype=np.uint8)
    img2 = np.zeros((996, 1732, 3), dtype=np.uint8)
    # Add a synthetic change patch in img2
    img2[200:400, 300:500] = 255

    cd = run_change_detection(img1, img2)
    assert cd.total_pixels == 976 * 1805
    assert cd.change_percentage >= 0.0
