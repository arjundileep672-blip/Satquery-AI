"""
SatQuery AI — Query Router
Classifies natural-language queries into one of 9 task types.
Uses keyword/pattern matching only — no LLM dependency for routing.
This keeps routing fast and deterministic.
"""

import re
from enum import Enum
from typing import Optional, Tuple


class TaskType(str, Enum):
    DETECTION = "detection"
    REMOTE_DETECTION = "remote_detection"
    BUILDING_DETECTION = "building_detection"
    COUNTING = "counting"
    SEGMENTATION = "segmentation"
    CHANGE_DETECTION = "change_detection"
    CHANGED_OBJECTS = "changed_objects"
    MULTITEMPORAL_ANALYSIS = "multitemporal_analysis"
    SPATIAL_ANALYSIS = "spatial_analysis"
    IMAGE_UNDERSTANDING = "image_understanding"


# ── Pattern banks ─────────────────────────────────────────────────────────────
# Order matters: more specific patterns must come before general ones.

_MULTITEMPORAL_PATTERNS = [
    r"multitemporal",
    r"multi[- ]date",
    r"parameter(s)? changed",
    r"what parameters? (have )?changed",
    r"which parameters? (have )?changed",
    r"how much did (the )?(built[- ]up|vegetation|water|road|building)",
    r"how much (vegetation|water|built[- ]up|building|road).*(lost|gained|changed|increased|decreased|expanded|was lost)",
    r"how many (new|removed|demolished) buildings",
    r"which roads (were|are) (newly constructed|built|new)",
    r"which parameters? changed (the )?most",
    r"compare building density",
    r"vegetation.*(loss|gain|changed|lost|decreased|increased)",
    r"land[- ]cover transition",
    r"transition matrix",
    r"between (the )?(dates?|years?|20\d{2})",
    r"time series",
    r"show me (all )?areas where (vegetation|water|built[- ]up) changed",
]


_CHANGED_OBJECTS_PATTERNS = [
    # "which buildings have changed", "which buildings changed", "which buildings were modified"
    r"which buildings?.*(changed|demolished|built|modified|altered)",
    r"buildings?.*(have|has|have been|has been).*(changed|demolished|built|modified)",
    r"changed buildings?",
    r"buildings? that (have |has )?(changed|disappeared|appeared)",
    r"(identify|find|detect|show|list) (the )?changed (buildings?|structures?|objects?)",
    r"building changes?",
    r"structures? (that )?(changed|appeared|disappeared)",
    r"new or demolished buildings?",
    r"which (objects?|structures?) (have )?(changed|been built|been demolished)",
]

_CHANGE_DETECTION_PATTERNS = [
    r"what (changed|is (different|new))",
    r"what('s| has| have) changed",
    r"change(s)? between (the )?two images?",
    r"(detect|find|identify|show|highlight|map) (the )?(changes?|differences?)",
    r"(temporal|bi-temporal|bitemporal) change",
    r"difference(s)? between",
    r"(before|after) (image|comparison)",
    r"percentage (of (the )?(image|area) )?(that )?changed",
    r"how much (of )?(the image |area )?(has |have )?changed",
    r"change mask",
    r"change detection",
]


_BUILDING_DETECTION_PATTERNS = [
    r"(find|detect|identify|locate|map|segment|count|show|extract) (all )?(the )?buildings?",
    r"building (footprints?|detection|count|masks?|polygons?)",
    r"buildings? (in|on|across|within|visible|present)",
    r"structures? and buildings?",
    r"(residential|commercial|industrial) buildings?",
    r"rooftops?",
    r"built-up area",
]

_REMOTE_DETECTION_PATTERNS = [
    r"(find|detect|identify|locate|show) (all )?(the )?(aircraft|airplanes?|planes?|jets?|helicopters?|runways?)",
    r"(find|detect|identify|locate|show) (all )?(the )?(ships?|vessels?|boats?|tankers?)",
    r"(find|detect|identify|locate|show) (all )?(the )?(vehicles?|trucks?|cars?) (in|on|at) (the )?(road|highway|parking|airport|airfield)",
    r"dota",
    r"oriented bounding box",
    r"(aerial|satellite|remote sensing) (object )?detection",
    r"(harbor|port|airport|airfield|storage tank|ground vehicle|large vehicle|small vehicle|swimming pool|soccer-ball field|tennis court|basketball court|roundabout)",
    r"(find|detect|show) (all )?(the )?(sports? (fields?|courts?)|storage tanks?|containers?)",
]

_COUNTING_PATTERNS = [
    r"how many",
    r"count (the |all )?(the )?",
    r"number of (the )?",
    r"total (number|count) of",
    r"(quantity|tally|enumerate)",
]

_SEGMENTATION_PATTERNS = [
    r"segment (the|all|a)",
    r"(draw|extract|generate|produce|create) (the )?(mask|masks|polygon|outline|boundary|boundaries|contour)",
    r"pixel-?level",
    r"instance segmentation",
    r"semantic segmentation",
    r"delineate",
]

_SPATIAL_ANALYSIS_PATTERNS = [
    r"(area|size|extent|coverage) of",
    r"(calculate|compute|measure|estimate) (the )?(area|distance|perimeter|coverage)",
    r"(spatial|geographic|geospatial)",
    r"(coordinates?|latitude|longitude|crs|projection|epsg)",
    r"(georeferenced|georeferencing)",
]

_DETECTION_PATTERNS = [
    r"(detect|locate) (all )?(the )?",
    r"find all (objects?|vehicles?|targets?|items?)",
    r"\bobject detection\b",
    r"\bdetect\b",
]

_UNDERSTANDING_PATTERNS = [
    r"(describe|overview|summary|summarize|explain|analyze|analyse|interpret)",
    r"(what (is|kind|type)|what('s| is) (this|the) (image|scene|area|region))",
    r"(scene|terrain|landscape|land (use|cover|type))",
    r"general (analysis|understanding|context|description)",
    r"(tell|explain) (me )?(about|what('s| is) (in|visible))",
]


def _matches_any(text: str, patterns: list) -> bool:
    """Return True if text matches any compiled regex pattern."""
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def extract_target_parameter(query: str) -> Optional[str]:
    """
    Extract specific requested parameter domain from user query:
    e.g. 'built_up', 'buildings', 'roads', 'vegetation', 'water', 'land_cover'.
    """
    q = query.lower()
    if re.search(r"built[- ]up|urban", q):
        return "built_up"
    if re.search(r"buildings?|structures?", q):
        return "buildings"
    if re.search(r"roads?|highways?|streets?|infrastructure", q):
        return "roads"
    if re.search(r"vegetation|greenery|forest|trees?|canopy|ndvi", q):
        return "vegetation"
    if re.search(r"water|rivers?|lakes?|coastal|flooding|ndwi", q):
        return "water"
    if re.search(r"land[- ]cover|transitions?", q):
        return "land_cover"
    return None


def route_query(query: str) -> Tuple[TaskType, str]:
    """
    Classify a natural-language query into a TaskType.

    Returns:
        (TaskType, rationale_string)

    Precedence (most specific → most general):
    1. multitemporal_analysis — parameter changes, transition matrix, multi-date comparisons
    2. changed_objects — requires two images + change + objects
    3. change_detection — requires two images
    4. building_detection
    5. remote_detection  (DOTA classes)
    6. counting
    7. segmentation
    8. spatial_analysis
    9. detection
    10. image_understanding (fallback)
    """
    q = query.strip().lower()

    if _matches_any(q, _MULTITEMPORAL_PATTERNS):
        param = extract_target_parameter(q)
        rationale = (
            f"Query requests multitemporal parameter analysis (target: {param})."
            if param
            else "Query requests comprehensive multitemporal change parameter analysis."
        )
        return TaskType.MULTITEMPORAL_ANALYSIS, rationale

    if _matches_any(q, _CHANGED_OBJECTS_PATTERNS):
        return TaskType.CHANGED_OBJECTS, "Query asks which specific objects/buildings changed."

    if _matches_any(q, _CHANGE_DETECTION_PATTERNS):
        return TaskType.CHANGE_DETECTION, "Query asks about temporal changes between two images."


    if _matches_any(q, _BUILDING_DETECTION_PATTERNS):
        return TaskType.BUILDING_DETECTION, "Query targets building/structure detection."

    if _matches_any(q, _REMOTE_DETECTION_PATTERNS):
        return TaskType.REMOTE_DETECTION, "Query targets DOTA-class aerial objects (aircraft, vessels, vehicles)."

    if _matches_any(q, _COUNTING_PATTERNS):
        # Counting buildings → building_detection with count
        if _matches_any(q, _BUILDING_DETECTION_PATTERNS):
            return TaskType.BUILDING_DETECTION, "Query counts buildings."
        return TaskType.COUNTING, "Query requests a count of objects."

    if _matches_any(q, _SEGMENTATION_PATTERNS):
        return TaskType.SEGMENTATION, "Query requests pixel-level segmentation masks."

    if _matches_any(q, _SPATIAL_ANALYSIS_PATTERNS):
        return TaskType.SPATIAL_ANALYSIS, "Query requests spatial or geospatial metrics."

    if _matches_any(q, _DETECTION_PATTERNS):
        return TaskType.DETECTION, "Query requests generic object detection."

    return TaskType.IMAGE_UNDERSTANDING, "Query requests scene-level image understanding."
