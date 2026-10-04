#!/usr/bin/env python3
"""Start the IntelliScreen FastAPI AI Proctoring Server for WebRTC frame ingestion."""

import os
import sys
from pathlib import Path
import uvicorn

# Ensure services/src is on PYTHONPATH
SERVICES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICES_DIR / "src"))

if __name__ == "__main__":
    host = os.environ.get("INTELLISCREEN_HOST", "0.0.0.0")
    port = int(os.environ.get("INTELLISCREEN_PORT", "8000"))

    print(f"Starting IntelliScreen AI Proctoring Server on http://{host}:{port}")
    print("API Documentation: http://localhost:8000/docs")
    print("WebRTC Ingestion Endpoint: http://localhost:8000/api/v1/sessions/{session_id}/frame")

    uvicorn.run(
        "intelliscreen.api.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
