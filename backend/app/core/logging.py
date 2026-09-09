"""
Structured Application Logging for SatQuery AI
Provides JSON-structured, security-conscious operational logs.
Never logs API keys, tokens, or raw binary payloads.
"""

import json
import logging
import sys
import time
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Attach structured contextual fields if present
        for field in ("request_id", "operation", "tool", "duration_ms", "status", "error_code"):
            if hasattr(record, field):
                val = getattr(record, field)
                # Ensure no secrets leak
                if field == "request_id" or field in ("operation", "tool", "status", "error_code"):
                    log_obj[field] = str(val)
                elif field == "duration_ms":
                    log_obj[field] = val
                    
        return json.dumps(log_obj)


def get_logger(name: str = "satquery") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = get_logger("satquery.core")
