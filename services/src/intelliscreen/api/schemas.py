"""API Request and Response schemas for WebRTC ingestion and Proctoring service."""

from typing import Any
import uuid

from pydantic import BaseModel, Field


class SessionStartRequest(BaseModel):
    """Payload to initialize a candidate proctoring session."""

    candidate_id: str = Field(description="Unique candidate identifier or roll number.")
    test_id: str = Field(default="mcq_test", description="Identifier of the exam/test being administered.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Custom exam metadata.")


class SessionStartResponse(BaseModel):
    """Returned upon successful proctoring session initialization."""

    session_id: str = Field(description="Unique session token to pass in frame uploads.")
    candidate_id: str
    status: str = "ACTIVE"
    created_at: float


class FrameBase64Request(BaseModel):
    """JSON alternative payload for posting base64 encoded video frames."""

    image_base64: str = Field(description="Base64-encoded image frame (JPEG/PNG).")
    frame_index: int = Field(default=0, ge=0)
    timestamp_ms: float = Field(default=0.0, ge=0.0, description="Session elapsed milliseconds.")


class TestIncidentRequest(BaseModel):
    """Client-side exam event (e.g. tab-switch, fullscreen exit)."""

    incident_type: str = Field(description="Type of incident: 'tab_switch', 'fullscreen_exit', 'copy_paste'.")
    timestamp_ms: float = Field(ge=0.0)
    details: dict[str, Any] = Field(default_factory=dict)


class RealtimeFrameResponse(BaseModel):
    """Per-frame real-time AI perception and proctoring response."""

    session_id: str
    frame_index: int
    timestamp_ms: float
    face_detected: bool
    face_count: int
    is_looking_away: bool
    gaze_direction: str
    phone_detected: bool
    laptop_detected: bool
    secondary_person_detected: bool
    active_events: list[str]
    instantaneous_suspicion_score: float = Field(ge=0.0, le=100.0)
    severity: str
    is_malpractice_flagged: bool
    latency_ms: float


class SessionSummaryResponse(BaseModel):
    """Final comprehensive exam audit report returned after test submission."""

    session_id: str
    candidate_id: str
    total_frames_processed: int
    final_suspicion_score: float
    final_severity: str
    is_flagged_for_review: bool
    total_events_count: int
    events_breakdown: dict[str, int]
    detailed_events: list[dict[str, Any]]
    audit_explanation: str
