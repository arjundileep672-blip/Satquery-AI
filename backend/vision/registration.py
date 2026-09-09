"""
SatQuery AI — ORB + RANSAC Image Registration
Aligns two images for change detection using feature-based homography.

OpenCV is required (cv2) — already installed as opencv-python-headless.
No PyTorch dependency.
"""

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class RegistrationError(Exception):
    """Raised when registration produces an unusable result."""


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert image to uint8 grayscale for feature detection."""
    if image.dtype != np.uint8:
        # Normalize to 0-255
        img_min, img_max = float(image.min()), float(image.max())
        if img_max > img_min:
            image = ((image.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)

    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is BGR uint8 for warping."""
    if image.dtype != np.uint8:
        img_min, img_max = float(image.min()), float(image.max())
        if img_max > img_min:
            image = ((image.astype(np.float32) - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def register_images(
    image1: np.ndarray,
    image2: np.ndarray,
    max_features: int = 5000,
    good_match_ratio: float = 0.75,
    min_inliers: int = 10,
) -> Dict[str, Any]:
    """
    Register image2 to image1 using ORB feature detection + RANSAC homography.

    Args:
        image1: Reference image (numpy array, any dtype/channels).
        image2: Target image to align to image1.
        max_features: Max ORB keypoints per image.
        good_match_ratio: Lowe's ratio test threshold.
        min_inliers: Minimum RANSAC inliers required for a valid homography.

    Returns:
        {
          "aligned_image": np.ndarray,     # image2 warped to image1 frame
          "homography": np.ndarray | None, # 3x3 homography matrix
          "keypoints1": int,
          "keypoints2": int,
          "matches_total": int,
          "matches_good": int,
          "inliers": int,
          "success": bool,
          "error": str | None,
          "processing_ms": float,
        }
    """
    t0 = time.perf_counter()

    gray1 = _to_gray(image1)
    gray2 = _to_gray(image2)
    target_bgr = _to_bgr_uint8(image2)

    # ORB detector
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    n_kp1 = len(kp1) if kp1 else 0
    n_kp2 = len(kp2) if kp2 else 0
    h1, w1 = image1.shape[:2]

    def _failure(reason: str, aligned: Optional[np.ndarray] = None) -> Dict[str, Any]:
        fallback = aligned if aligned is not None else target_bgr
        if fallback.shape[:2] != (h1, w1):
            fallback = cv2.resize(fallback, (w1, h1), interpolation=cv2.INTER_LINEAR)

        return {
            "aligned_image": fallback,
            "homography": None,
            "keypoints1": n_kp1,
            "keypoints2": n_kp2,
            "matches_total": 0,
            "matches_good": 0,
            "inliers": 0,
            "success": False,
            "error": reason,
            "processing_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    if des1 is None or des2 is None or n_kp1 < 4 or n_kp2 < 4:
        return _failure(f"Insufficient keypoints: img1={n_kp1}, img2={n_kp2}")

    # Brute-force Hamming matcher (correct for ORB binary descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        raw_matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error as exc:
        return _failure(f"BFMatcher failed: {exc}")

    # Lowe's ratio test
    good: list = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < good_match_ratio * n.distance:
                good.append(m)

    n_total = len(raw_matches)
    n_good = len(good)

    if n_good < 4:
        return _failure(f"Too few good matches after ratio test: {n_good}")

    # Extract matched point coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # RANSAC homography
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
    inliers = int(mask.sum()) if mask is not None else 0

    if H is None or inliers < min_inliers:
        return _failure(
            f"RANSAC produced invalid homography or too few inliers ({inliers} < {min_inliers})",
            aligned=target_bgr,
        )

    # Warp image2 into image1's coordinate frame
    h, w = image1.shape[:2]
    aligned = cv2.warpPerspective(target_bgr, H, (w, h), flags=cv2.INTER_LINEAR)

    return {
        "aligned_image": aligned,
        "homography": H,
        "keypoints1": n_kp1,
        "keypoints2": n_kp2,
        "matches_total": n_total,
        "matches_good": n_good,
        "inliers": inliers,
        "success": True,
        "error": None,
        "processing_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
