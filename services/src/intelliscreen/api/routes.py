"""API Routes for Candidate Proctoring, WebRTC Frame Ingestion, and Exam Audit Reports."""

import base64
import time
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from intelliscreen.api.schemas import (
    FrameBase64Request,
    RealtimeFrameResponse,
    SessionStartRequest,
    SessionStartResponse,
    SessionSummaryResponse,
    TestIncidentRequest,
)
from intelliscreen.api.session_manager import SessionManager
from intelliscreen.core.exceptions import FramePreprocessingError, SessionError

router = APIRouter(prefix="/api/v1", tags=["Proctoring"])
manager = SessionManager.get_instance()


@router.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Health check endpoint confirming AI proctoring service readiness."""
    return {
        "status": "healthy",
        "service": "IntelliScreen AI Proctoring",
        "version": "0.1.0",
    }


@router.post("/sessions/start", response_model=SessionStartResponse, status_code=status.HTTP_201_CREATED)
def start_session(req: SessionStartRequest) -> SessionStartResponse:
    """Initialize a new candidate exam session."""
    session = manager.create_session(candidate_id=req.candidate_id, test_id=req.test_id)
    return SessionStartResponse(
        session_id=session.session_id,
        candidate_id=session.candidate_id,
        status="ACTIVE",
        created_at=session.created_at,
    )


@router.post("/sessions/{session_id}/frame", response_model=RealtimeFrameResponse)
async def process_frame_multipart(
    session_id: str,
    file: UploadFile = File(..., description="JPEG/PNG encoded video frame from WebRTC canvas"),
    frame_index: int = Form(0),
    timestamp_ms: float = Form(0.0),
) -> RealtimeFrameResponse:
    """Primary WebRTC ingestion endpoint receiving raw image frame bytes via multipart/form-data."""
    try:
        session = manager.get_session(session_id)
    except SessionError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image byte buffer received."
        )

    return _execute_proctoring_pipeline(
        session=session,
        raw_bytes=raw_bytes,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
    )


@router.post("/sessions/{session_id}/frame-base64", response_model=RealtimeFrameResponse)
def process_frame_base64(
    session_id: str,
    req: FrameBase64Request,
) -> RealtimeFrameResponse:
    """Alternative frame ingestion endpoint receiving base64-encoded image payloads."""
    try:
        session = manager.get_session(session_id)
    except SessionError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    try:
        # Strip data URL prefix if present (e.g. data:image/jpeg;base64,...)
        payload = req.image_base64
        if "," in payload:
            payload = payload.split(",", 1)[1]
        raw_bytes = base64.b64decode(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid base64 payload: {exc}"
        ) from exc

    return _execute_proctoring_pipeline(
        session=session,
        raw_bytes=raw_bytes,
        frame_index=req.frame_index,
        timestamp_ms=req.timestamp_ms,
    )


@router.post("/sessions/{session_id}/test-incident")
def record_test_incident(
    session_id: str,
    req: TestIncidentRequest,
) -> dict[str, Any]:
    """Log client-side exam incidents (e.g. candidate switched browser tab, exited fullscreen)."""
    try:
        session = manager.get_session(session_id)
    except SessionError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    incident_str = f"{req.incident_type} at {req.timestamp_ms:.0f}ms"
    session.test_incidents.append(incident_str)
    return {"status": "recorded", "incident": incident_str}


@router.get("/sessions/{session_id}/summary", response_model=SessionSummaryResponse)
def get_session_summary(session_id: str) -> SessionSummaryResponse:
    """Generate final exam proctoring audit report with breakdown of all violations."""
    try:
        session = manager.get_session(session_id)
    except SessionError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    # Finalize any ongoing temporal events
    final_events = session.temporal_engine.flush()
    all_events = session.temporal_engine.get_all_events()

    window_summary = session.temporal_engine.get_window_summary()
    final_eval = session.scorer.evaluate(
        window_summary=window_summary,
        active_events=[],
        test_incidents=session.test_incidents,
    )

    # Event breakdown count
    breakdown: dict[str, int] = {}
    for ev in all_events:
        breakdown[ev.event_type.value] = breakdown.get(ev.event_type.value, 0) + 1

    detailed = [ev.model_dump() for ev in all_events]

    return SessionSummaryResponse(
        session_id=session.session_id,
        candidate_id=session.candidate_id,
        total_frames_processed=session.total_frames,
        final_suspicion_score=final_eval.score,
        final_severity=final_eval.severity.value,
        is_flagged_for_review=final_eval.is_malpractice_flagged,
        total_events_count=len(all_events),
        events_breakdown=breakdown,
        detailed_events=detailed,
        audit_explanation=final_eval.explanation,
    )


def _execute_proctoring_pipeline(
    session: Any,
    raw_bytes: bytes,
    frame_index: int,
    timestamp_ms: float,
) -> RealtimeFrameResponse:
    """Core executor: decodes frame, runs vision DAG, updates temporal engine, and scores."""
    t0 = time.perf_counter()

    try:
        # 1. Preprocess raw byte buffer
        preprocessed = manager.vision_pipeline.preprocessor.process_bytes(
            raw_bytes=raw_bytes,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )
    except FramePreprocessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Frame decode error: {exc}"
        ) from exc

    # 2. Run Unified Vision Pipeline
    pipeline_result = manager.vision_pipeline.process_frame(
        frame=preprocessed.bgr_image,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        is_rgb=False,
    )
    obs = pipeline_result.observation

    # 3. Update Temporal Event Engine & 30s Rolling Buffer
    new_events = session.temporal_engine.process_observation(obs)
    active_events = session.temporal_engine.get_active_events()
    window_summary = session.temporal_engine.get_window_summary()
    session.total_frames += 1

    # 4. Compute Suspicion Score
    evaluation = session.scorer.evaluate(
        window_summary=window_summary,
        active_events=active_events,
        test_incidents=session.test_incidents,
        timestamp=timestamp_ms / 1000.0,
    )
    session.latest_evaluation = evaluation

    total_latency_ms = (time.perf_counter() - t0) * 1000

    # Extract primary flags
    gaze_dir = obs.gaze.direction.value if obs.gaze else "CENTER"
    is_looking_away = obs.gaze.is_deviated() if obs.gaze else False
    if obs.head_pose and obs.head_pose.is_looking_away():
        is_looking_away = True

    phone_detected = pipeline_result.object_result.phone_detected
    laptop_detected = pipeline_result.object_result.laptop_detected
    secondary_person = pipeline_result.object_result.secondary_person_detected or obs.face_count > 1

    return RealtimeFrameResponse(
        session_id=session.session_id,
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        face_detected=obs.face_detected,
        face_count=obs.face_count,
        is_looking_away=is_looking_away,
        gaze_direction=gaze_dir,
        phone_detected=phone_detected,
        laptop_detected=laptop_detected,
        secondary_person_detected=secondary_person,
        active_events=[e.event_type.value for e in active_events],
        instantaneous_suspicion_score=evaluation.score,
        severity=evaluation.severity.value,
        is_malpractice_flagged=evaluation.is_malpractice_flagged,
        latency_ms=round(total_latency_ms, 2),
    )
