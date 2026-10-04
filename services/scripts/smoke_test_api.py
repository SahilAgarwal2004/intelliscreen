#!/usr/bin/env python3
"""Smoke test demonstrating WebRTC frame ingestion and proctoring session API."""

import json
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

SERVICES_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICES_DIR / "src"))

from intelliscreen.api.app import app

client = TestClient(app)
FIXTURE_IMAGE = SERVICES_DIR / "tests" / "fixtures" / "sample_face.jpg"


def run_smoke_test() -> None:
    print("=" * 78)
    print("      INTELLISCREEN WEBRTC INGESTION & EXAM PROCTORING SMOKE TEST")
    print("=" * 78)

    # 1. Start Proctoring Session
    start_resp = client.post(
        "/api/v1/sessions/start",
        json={
            "candidate_id": "cand_rohit_772",
            "test_id": "data_structures_mcq",
            "metadata": {"total_questions": 30, "duration_minutes": 45},
        },
    )
    assert start_resp.status_code == 201
    session_data = start_resp.json()
    session_id = session_data["session_id"]
    print(f"\n[1] Initialized Candidate Exam Session: {session_id}")
    print(f"    Candidate: {session_data['candidate_id']} | Status: {session_data['status']}")

    # 2. Simulate WebRTC Video Frame Upload (Multipart Form-Data)
    print("\n[2] Ingesting WebRTC Frame #1 (Normal Candidate Face)...")
    with open(FIXTURE_IMAGE, "rb") as f:
        img_bytes = f.read()

    frame_resp = client.post(
        f"/api/v1/sessions/{session_id}/frame",
        files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
        data={"frame_index": "1", "timestamp_ms": "100.0"},
    )
    assert frame_resp.status_code == 200
    frame_data = frame_resp.json()
    print("    Received Real-Time Frame Telemetry:")
    print(json.dumps(frame_data, indent=4))

    # 3. Simulate Client-Side Exam Incident (Candidate switched tab during MCQ test)
    print("\n[3] Logging Client-Side Exam Incident ('tab_switch')...")
    incident_resp = client.post(
        f"/api/v1/sessions/{session_id}/test-incident",
        json={"incident_type": "tab_switch", "timestamp_ms": 14500.0, "details": {"target_url": "unknown"}},
    )
    assert incident_resp.status_code == 200
    print(f"    Incident Status: {incident_resp.json()}")

    # 4. Fetch Complete Final Audit Report
    print("\n[4] Generating Final Exam Audit Summary Report...")
    summary_resp = client.get(f"/api/v1/sessions/{session_id}/summary")
    assert summary_resp.status_code == 200
    summary_data = summary_resp.json()
    print(json.dumps(summary_data, indent=4))

    print("\n" + "=" * 78)
    print("SMOKE TEST COMPLETE: WebRTC Endpoint and Exam Proctoring Fully Operational!")
    print("=" * 78)


if __name__ == "__main__":
    run_smoke_test()
