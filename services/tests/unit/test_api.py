"""Integration and unit tests for FastAPI Proctoring Service endpoints."""

import base64
import os
import cv2
import pytest
from fastapi.testclient import TestClient

from intelliscreen.api.app import app

client = TestClient(app)

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")
SAMPLE_FACE_PATH = os.path.join(FIXTURES_DIR, "sample_face.jpg")


def test_health_check():
    """Verify health endpoint returns status healthy."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "IntelliScreen" in data["service"]


def test_session_lifecycle_and_frame_ingestion():
    """Verify complete lifecycle: start session -> post frame -> post incident -> get summary."""
    # 1. Start Session
    start_resp = client.post(
        "/api/v1/sessions/start",
        json={"candidate_id": "cand_mcq_101", "test_id": "python_core_mcq"},
    )
    assert start_resp.status_code == 201
    start_data = start_resp.json()
    session_id = start_data["session_id"]
    assert session_id is not None
    assert start_data["status"] == "ACTIVE"

    # 2. Ingest Video Frame via Multipart Form-Data (WebRTC format)
    with open(SAMPLE_FACE_PATH, "rb") as f:
        img_bytes = f.read()

    frame_resp = client.post(
        f"/api/v1/sessions/{session_id}/frame",
        files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
        data={"frame_index": "1", "timestamp_ms": "100.0"},
    )
    assert frame_resp.status_code == 200
    frame_data = frame_resp.json()
    assert frame_data["session_id"] == session_id
    assert frame_data["face_detected"] is True
    assert frame_data["face_count"] >= 1
    assert "instantaneous_suspicion_score" in frame_data
    assert frame_data["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    # 3. Ingest Video Frame via Base64 JSON
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    b64_resp = client.post(
        f"/api/v1/sessions/{session_id}/frame-base64",
        json={
            "image_base64": f"data:image/jpeg;base64,{b64_str}",
            "frame_index": 2,
            "timestamp_ms": 200.0,
        },
    )
    assert b64_resp.status_code == 200
    assert b64_resp.json()["frame_index"] == 2

    # 4. Record Client-Side Test Incident (e.g. Tab Switch)
    incident_resp = client.post(
        f"/api/v1/sessions/{session_id}/test-incident",
        json={"incident_type": "tab_switch", "timestamp_ms": 350.0},
    )
    assert incident_resp.status_code == 200
    assert incident_resp.json()["status"] == "recorded"

    # 5. Fetch Final Session Summary
    summary_resp = client.get(f"/api/v1/sessions/{session_id}/summary")
    assert summary_resp.status_code == 200
    summary = summary_resp.json()
    assert summary["session_id"] == session_id
    assert summary["candidate_id"] == "cand_mcq_101"
    assert summary["total_frames_processed"] == 2
    assert "final_suspicion_score" in summary
    assert "audit_explanation" in summary


def test_invalid_session_returns_404():
    """Verify invalid session ID returns 404 Not Found."""
    with open(SAMPLE_FACE_PATH, "rb") as f:
        img_bytes = f.read()

    resp = client.post(
        "/api/v1/sessions/non-existent-session-id/frame",
        files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 404


def test_empty_frame_returns_400():
    """Verify empty frame upload returns 400 Bad Request."""
    start_resp = client.post("/api/v1/sessions/start", json={"candidate_id": "cand_empty_test"})
    session_id = start_resp.json()["session_id"]

    resp = client.post(
        f"/api/v1/sessions/{session_id}/frame",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400
