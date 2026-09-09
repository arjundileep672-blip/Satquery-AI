"""
SatQuery AI - Changed Objects Pipeline
Full pipeline: registration -> change detection -> building detection -> overlap.
"""
import numpy as np
from typing import Any, Dict, Optional

from app.core.logging import get_logger

logger = get_logger("satquery.pipelines.changed_objects")


def run_changed_objects(
    image1: np.ndarray,
    image2: np.ndarray,
    object_type: str = "building",
    sam_refine_top_n: int = 5,
    geo_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Semantic change intelligence pipeline.
    Delegates to app.core.semantic_change_intelligence.
    """
    from app.core.semantic_change_intelligence import run_semantic_change_pipeline
    return run_semantic_change_pipeline(
        image1=image1,
        image2=image2,
        object_type=object_type,
        sam_refine_top_n=sam_refine_top_n,
        geo_meta=geo_meta,
    )
