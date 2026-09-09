"""
Mock VLM Adapter for Zero-Key Testing and Local Demonstration
Provides deterministic, domain-aware responses simulating a remote-sensing fine-tuned VLM.
"""

from typing import Any, Dict, Optional
from app.models.base import VisionLanguageModel, VLMResponse


class MockVLMAdapter(VisionLanguageModel):
    @property
    def name(self) -> str:
        return "mock-rs-vlm-v1"

    def generate_response(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
    ) -> VLMResponse:
        meta = metadata or {}
        q_lower = prompt.lower()
        width = meta.get("width", "unknown")
        height = meta.get("height", "unknown")
        is_geo = meta.get("is_geotiff", False)
        crs = meta.get("crs", "N/A")

        # Context prefix
        geo_info = f" (GeoTIFF, CRS: {crs}, {width}x{height} px)" if is_geo else f" ({width}x{height} px)"

        # Grounded rule-based response generation for remote sensing imagery
        if any(w in q_lower for w in ("object", "visible", "what is in", "detect", "find")):
            text = (
                f"Visual analysis of the satellite scene{geo_info} reveals several distinct remote-sensing features:\n\n"
                "- **Maritime Vessels & Berths**: Multiple commercial ships and container vessels docked along the deepwater harbor pier.\n"
                "- **Industrial Infrastructure**: Cylindrical fuel storage tanks, logistics warehousing, and gantry crane networks.\n"
                "- **Transportation Corridors**: Paved port access roads and vehicle staging yards adjacent to the wharf.\n"
                "- **Water & Coastal Buffer**: Open coastal fairway with calm water surface and bordering vegetated tidal mudflats."
            )
        elif any(w in q_lower for w in ("water", "river", "sea", "ocean", "lake", "flood")):
            text = (
                f"Yes, open water is prominently visible in this scene{geo_info}. "
                "The western section is dominated by an open coastal water body / fairway, "
                "characterized by low spectral reflectance in the infrared bands and sharp delineations along the artificial sea wall."
            )
        elif any(w in q_lower for w in ("vegetation", "forest", "green", "trees", "crop")):
            text = (
                f"Vegetation patches are identifiable across the perimeter of the scene{geo_info}. "
                "Dense mangrove canopy and coastal greenery flank the tidal inlet, exhibiting high near-infrared response."
            )
        elif any(w in q_lower for w in ("building", "urban", "city", "structure")):
            text = (
                f"The image contains prominent built-up industrial structures{geo_info}, "
                "including regular rectangular warehouse footprints, terminal logistics centers, and administrative port facilities."
            )
        else:
            text = (
                f"Observation of the remote sensing scene{geo_info} in response to: \"{prompt}\":\n\n"
                "The imagery exhibits a mixed coastal industrial environment. The scene displays high-contrast "
                "man-made structures, port waterfront infrastructure, and adjacent water surfaces with clear spatial boundaries."
            )

        return VLMResponse(
            text=text,
            model_name=self.name,
            confidence=None,  # Qualitative / uncalibrated per prompt instructions
            raw_metadata={"adapter": "MockVLMAdapter", "mode": "demo"},
        )
