"""
Geospatial Raster Operations & Multi-Sensor Signal Processing
"""

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import tifffile

from app.core.config import settings


def calculate_ndvi(red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)
    Values range from -1.0 to +1.0. High values (>0.4) indicate dense green canopy.
    """
    red = red_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    denominator = nir + red
    # Avoid zero division
    denominator[denominator == 0] = 1e-6
    ndvi = (nir - red) / denominator
    return np.clip(ndvi, -1.0, 1.0)


def calculate_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Water Index (McFeeters): (Green - NIR) / (Green + NIR)
    Values > 0 delineate surface water bodies.
    """
    green = green_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    denominator = green + nir
    denominator[denominator == 0] = 1e-6
    ndwi = (green - nir) / denominator
    return np.clip(ndwi, -1.0, 1.0)


def calibrate_sar_to_db(sar_dn: np.ndarray) -> np.ndarray:
    """
    Radiometric calibration of Sentinel-1 Digital Numbers (DN) to Sigma Nought (dB):
    sigma0_dB = 10 * log10(DN^2 + eps)
    """
    dn = sar_dn.astype(np.float32)
    # Clamp negative or zero values
    dn = np.maximum(dn, 1.0)
    sigma0_linear = dn ** 2
    sigma0_db = 10.0 * np.log10(sigma0_linear + 1e-7)
    return sigma0_db


def apply_lee_filter(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Refined Lee Speckle Filter for SAR Backscatter.
    Smooths homogeneous regions while preserving sharp edges and point targets (ships/buildings).
    """
    from scipy.ndimage import uniform_filter
    
    img = image.astype(np.float32)
    local_mean = uniform_filter(img, (window_size, window_size))
    local_sqr_mean = uniform_filter(img ** 2, (window_size, window_size))
    local_variance = np.maximum(local_sqr_mean - local_mean ** 2, 0)
    
    overall_variance = np.var(img)
    if overall_variance == 0:
        return img
        
    weight = local_variance / (local_variance + overall_variance + 1e-6)
    filtered = local_mean + weight * (img - local_mean)
    return filtered


def generate_demo_rasters_if_missing():
    """
    Generates realistic demo remote sensing multi-band assets if missing.
    Location: Mumbai Harbor / Jawaharlal Nehru Port (18.94N, 72.93E)
    - optical_t1.tif: 4-band Sentinel-2 (Blue, Green, Red, NIR) - Pre-construction / baseline
    - optical_t2.tif: 4-band Sentinel-2 - Post-construction (new port terminal, ships)
    - sar_vv_vh.tif: 2-band Sentinel-1 C-Band SAR (Band 1: VV, Band 2: VH)
    """
    demo_dir = settings.DEMO_IMAGERY_DIR
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    t1_path = demo_dir / "optical_t1.tif"
    t2_path = demo_dir / "optical_t2.tif"
    sar_path = demo_dir / "sar_vv_vh.tif"
    
    height, width = 512, 512
    
    # 1. Generate Optical T1 (Baseline)
    if not t1_path.exists():
        # Bands: 0: Blue, 1: Green, 2: Red, 3: NIR
        t1_data = np.zeros((4, height, width), dtype=np.uint16)
        
        # Water area (western half): High Blue, moderate Green, low Red, very low NIR
        t1_data[0, :, :250] = 1200 + np.random.randint(0, 100, (height, 250))
        t1_data[1, :, :250] = 1000 + np.random.randint(0, 80, (height, 250))
        t1_data[2, :, :250] = 600 + np.random.randint(0, 50, (height, 250))
        t1_data[3, :, :250] = 200 + np.random.randint(0, 30, (height, 250))
        
        # Land & Urban area (eastern half): Moderate RGB, High NIR
        t1_data[0, :, 250:] = 800 + np.random.randint(0, 150, (height, 262))
        t1_data[1, :, 250:] = 1100 + np.random.randint(0, 150, (height, 262))
        t1_data[2, :, 250:] = 1200 + np.random.randint(0, 150, (height, 262))
        t1_data[3, :, 250:] = 3200 + np.random.randint(0, 300, (height, 262))
        
        # 3 baseline ships in harbor
        ship_coords = [(120, 100), (280, 150), (400, 180)]
        for y, x in ship_coords:
            t1_data[:, y-8:y+8, x-4:x+4] = np.array([3500, 3500, 3500, 3200])[:, None, None]
            
        tifffile.imwrite(str(t1_path), t1_data)
        
    # 2. Generate Optical T2 (New port expansion + 3 additional vessels)
    if not t2_path.exists():
        t2_data = np.zeros((4, height, width), dtype=np.uint16)
        
        # Water area
        t2_data[0, :, :250] = 1180 + np.random.randint(0, 100, (height, 250))
        t2_data[1, :, :250] = 990 + np.random.randint(0, 80, (height, 250))
        t2_data[2, :, :250] = 610 + np.random.randint(0, 50, (height, 250))
        t2_data[3, :, :250] = 210 + np.random.randint(0, 30, (height, 250))
        
        # Land area
        t2_data[0, :, 250:] = 820 + np.random.randint(0, 150, (height, 262))
        t2_data[1, :, 250:] = 1120 + np.random.randint(0, 150, (height, 262))
        t2_data[2, :, 250:] = 1250 + np.random.randint(0, 150, (height, 262))
        t2_data[3, :, 250:] = 3150 + np.random.randint(0, 300, (height, 262))
        
        # Retain original ships
        for y, x in [(120, 100), (280, 150), (400, 180)]:
            t2_data[:, y-8:y+8, x-4:x+4] = np.array([3500, 3500, 3500, 3200])[:, None, None]
            
        # Add 3 NEW vessels in harbor
        new_ships = [(180, 120), (320, 80), (210, 190)]
        for y, x in new_ships:
            t2_data[:, y-10:y+10, x-5:x+5] = np.array([3800, 3800, 3900, 3400])[:, None, None]
            
        # Add NEW port terminal reclamation in water (Change Region)
        # Rectangular pier from x:230 to x:270, y:200 to y:310
        t2_data[0, 200:310, 230:270] = 1800  # High reflectance concrete
        t2_data[1, 200:310, 230:270] = 2000
        t2_data[2, 200:310, 230:270] = 2200
        t2_data[3, 200:310, 230:270] = 2500
        
        tifffile.imwrite(str(t2_path), t2_data)
        
    # 3. Generate Sentinel-1 SAR VV/VH co-registered
    if not sar_path.exists():
        # Band 0: VV (Digital Number), Band 1: VH
        sar_data = np.zeros((2, height, width), dtype=np.uint16)
        
        # Water area: specular reflection -> low backscatter (DN ~ 10-20, dB ~ -24 to -20)
        sar_data[0, :, :250] = 15 + np.random.poisson(3, (height, 250))
        sar_data[1, :, :250] = 8 + np.random.poisson(2, (height, 250))
        
        # Land area: rough surface -> moderate backscatter (DN ~ 60-120, dB ~ -14 to -8)
        sar_data[0, :, 250:] = 85 + np.random.poisson(15, (height, 262))
        sar_data[1, :, 250:] = 45 + np.random.poisson(10, (height, 262))
        
        # Ships and metallic cranes: Double bounce corner reflector -> Extreme backscatter (DN ~ 800-1200, dB ~ +2 to +8)
        all_ships = [(120, 100), (280, 150), (400, 180), (180, 120), (320, 80), (210, 190)]
        for y, x in all_ships:
            sar_data[0, y-6:y+6, x-3:x+3] = 950 + np.random.randint(0, 100, (12, 6))
            sar_data[1, y-6:y+6, x-3:x+3] = 550 + np.random.randint(0, 80, (12, 6))
            
        # Concrete pier
        sar_data[0, 200:310, 230:270] = 450 + np.random.randint(0, 50, (110, 40))
        sar_data[1, 200:310, 230:270] = 220 + np.random.randint(0, 30, (110, 40))
        
        tifffile.imwrite(str(sar_path), sar_data)


def export_preview_png(raster_path: str, output_png_path: str, bands: Tuple[int, int, int] = (2, 1, 0)):
    """Convert multi-band TIFF to normalized 8-bit RGB preview PNG."""
    try:
        data = tifffile.imread(raster_path)
        if data.ndim == 3:
            # (bands, H, W) or (H, W, bands)
            if data.shape[0] < data.shape[2]:
                r = data[bands[0]]
                g = data[bands[1]]
                b = data[bands[2]]
            else:
                r = data[:, :, bands[0]]
                g = data[:, :, bands[1]]
                b = data[:, :, bands[2]]
        elif data.ndim == 2:
            r = g = b = data
        else:
            return
            
        # 2% - 98% percentile linear stretch
        def stretch(arr):
            p2, p98 = np.percentile(arr, (2, 98))
            if p98 == p2:
                return np.zeros_like(arr, dtype=np.uint8)
            scaled = (arr - p2) / (p98 - p2) * 255.0
            return np.clip(scaled, 0, 255).astype(np.uint8)
            
        rgb = np.stack([stretch(r), stretch(g), stretch(b)], axis=-1)
        img = Image.fromarray(rgb)
        img.save(output_png_path, "PNG")
    except Exception as e:
        print(f"Error creating preview PNG: {e}")
