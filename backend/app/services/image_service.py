"""
Image Service for SatQuery AI
Handles file validation, sanitization, metadata extraction (Rasterio + Pillow),
and safe preview generation for web display.
"""

import io
import os
import re
import uuid
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import rasterio
from rasterio.errors import RasterioIOError

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.metadata import ImageMetadata

logger = get_logger("satquery.services.image")


class ImageProcessingError(Exception):
    """Custom domain exception for image processing errors."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to prevent directory traversal and special character issues."""
    name = Path(filename).name
    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    if not clean_name:
        clean_name = f"upload_{uuid.uuid4().hex[:8]}.png"
    return clean_name


def validate_file(filename: str, content_type: Optional[str], size_bytes: int) -> str:
    """
    Validate file extension, MIME type, and size limits.
    Returns the normalized extension.
    """
    if size_bytes <= 0:
        raise ImageProcessingError("Uploaded file is empty (0 bytes).", status_code=400)

    if size_bytes > settings.MAX_FILE_SIZE_BYTES:
        mb_limit = settings.MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise ImageProcessingError(
            f"File size ({size_bytes / (1024 * 1024):.1f} MB) exceeds maximum allowed limit ({mb_limit:.0f} MB).",
            status_code=413,
        )

    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise ImageProcessingError(
            f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(settings.ALLOWED_EXTENSIONS)}.",
            status_code=415,
        )

    return ext


def extract_geotiff_metadata(file_bytes: bytes, filename: str) -> Optional[ImageMetadata]:
    """
    Attempt to read geospatial metadata using Rasterio in-memory MemoryFile.
    Returns ImageMetadata if valid GeoTIFF with CRS/transform, else None.
    """
    try:
        with rasterio.io.MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                has_crs = src.crs is not None
                crs_str = str(src.crs) if has_crs else None
                bounds_list = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top] if has_crs else None
                transform_list = list(src.transform)[:6] if src.transform else None
                
                return ImageMetadata(
                    filename=filename,
                    format="GeoTIFF" if has_crs else "TIFF",
                    width=src.width,
                    height=src.height,
                    channels=src.count,
                    band_count=src.count,
                    size_bytes=len(file_bytes),
                    is_geotiff=has_crs,
                    crs=crs_str,
                    transform=transform_list,
                    bounds=bounds_list,
                    nodata=float(src.nodata) if src.nodata is not None else None,
                    dtype=str(src.dtypes[0]) if src.dtypes else None,
                    tags=dict(src.tags()),
                )
    except (RasterioIOError, Exception) as err:
        logger.debug(f"Rasterio could not open as GeoTIFF ({filename}): {err}")
        return None


def extract_standard_image_metadata(file_bytes: bytes, filename: str) -> ImageMetadata:
    """
    Extract standard raster metadata using Pillow.
    Validates that the file is indeed a decodable image and handles mobile EXIF rotation.
    """
    try:
        from PIL import ImageOps
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()  # Fast structural verification

        # Re-open for properties since verify() alters file pointer
        with Image.open(io.BytesIO(file_bytes)) as img:
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            width, height = img.size
            fmt = img.format or Path(filename).suffix.replace(".", "").upper()
            channels = len(img.getbands()) if hasattr(img, "getbands") else 3

            return ImageMetadata(
                filename=filename,
                format=fmt,
                width=width,
                height=height,
                channels=channels,
                band_count=channels,
                size_bytes=len(file_bytes),
                is_geotiff=False,
            )
    except Exception as err:
        raise ImageProcessingError(f"Corrupt or invalid image file: {str(err)}", status_code=400)


def extract_metadata(file_bytes: bytes, filename: str) -> ImageMetadata:
    """
    Unified metadata extraction: Tries GeoTIFF (Rasterio) first for .tif/.tiff,
    falls back to Pillow.
    """
    ext = Path(filename).suffix.lower()
    if ext in (".tif", ".tiff"):
        geo_meta = extract_geotiff_metadata(file_bytes, filename)
        if geo_meta:
            return geo_meta

    return extract_standard_image_metadata(file_bytes, filename)


def generate_web_preview(file_bytes: bytes, filename: str, max_dimension: int = 1200) -> Tuple[bytes, str]:
    """
    Produces a web-safe JPEG preview for uploaded images (essential for multi-band or 16-bit GeoTIFFs,
    as well as mobile formats).
    """
    ext = Path(filename).suffix.lower()

    # Try reading with rasterio if TIFF
    if ext in (".tif", ".tiff"):
        try:
            with rasterio.io.MemoryFile(file_bytes) as memfile:
                with memfile.open() as src:
                    # Select up to 3 bands for RGB
                    count = src.count
                    if count >= 3:
                        bands = [src.read(i) for i in (1, 2, 3)]
                    else:
                        b1 = src.read(1)
                        bands = [b1, b1, b1]

                    # Normalize to 8-bit using 2nd and 98th percentiles
                    def norm8(arr: np.ndarray) -> np.ndarray:
                        p2, p98 = np.percentile(arr, (2, 98))
                        if p98 <= p2:
                            return np.zeros_like(arr, dtype=np.uint8)
                        scaled = (arr - p2) / (p98 - p2) * 255.0
                        return np.clip(scaled, 0, 255).astype(np.uint8)

                    rgb = np.stack([norm8(b) for b in bands], axis=-1)
                    pil_img = Image.fromarray(rgb)

                    # Resize if needed
                    pil_img.thumbnail((max_dimension, max_dimension))
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=88)
                    return buf.getvalue(), "image/jpeg"
        except Exception:
            pass

    # Fallback to standard Pillow conversion with EXIF orientation support
    try:
        from PIL import ImageOps
        with Image.open(io.BytesIO(file_bytes)) as img:
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            rgb_img = img.convert("RGB")
            rgb_img.thumbnail((max_dimension, max_dimension))
            buf = io.BytesIO()
            rgb_img.save(buf, format="JPEG", quality=88)
            return buf.getvalue(), "image/jpeg"
    except Exception as e:
        logger.warning(f"Could not generate preview: {e}")
        return file_bytes, "image/png"


def prepare_image_for_vlm(file_bytes: bytes, filename: str, mime_type: str) -> Tuple[bytes, str]:
    """
    Produces VLM-safe image bytes from the uploaded raster file.

    For GeoTIFF/TIFF/WebP/BMP inputs: generates an 8-bit RGB JPEG visualization
    percentile-normalized or thumbnail-safe.
    For standard PNG/JPEG inputs: returns (file_bytes, mime_type) unchanged.

    Returns:
        Tuple[bytes, str]: (vlm_image_bytes, vlm_mime_type)
    """
    ext = Path(filename).suffix.lower()
    if ext in (".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"):
        jpeg_bytes, jpeg_mime = generate_web_preview(file_bytes, filename)
        logger.debug(
            f"Image '{filename}' converted to web-safe JPEG for VLM inference "
            f"({len(jpeg_bytes)} bytes)."
        )
        return jpeg_bytes, jpeg_mime

    # PNG and JPEG pass through
    return file_bytes, mime_type

