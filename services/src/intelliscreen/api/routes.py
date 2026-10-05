"""API Routes for Candidate Proctoring, WebRTC Frame Ingestion, and Exam Audit Reports."""

import base64
import time
from typing import Any

import numpy as np
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
    session = manager.create_session(
        candidate_id=req.candidate_id,
        test_id=req.test_id,
        session_id=req.session_id,
    )

    # Process baseline photo if provided for examinee identity and resting gaze calibration
    if req.baseline_photo:
        try:
            b_payload = req.baseline_photo
            if "," in b_payload:
                b_payload = b_payload.split(",", 1)[1]
            raw_b = base64.b64decode(b_payload)
            ref_prep = manager.vision_pipeline.preprocessor.process_bytes(raw_b, frame_index=0, timestamp_ms=0.0)
            ref_res = manager.vision_pipeline.process_frame(ref_prep.bgr_image, frame_index=0, timestamp_ms=0.0)
            if ref_res.face_result.face_detected and ref_res.face_result.primary_face:
                box = ref_res.face_result.primary_face.box
                session.baseline_photo = req.baseline_photo
                session.baseline_face_aspect_ratio = float(box.width / (box.height + 1e-6))
                # Store inter-eye and face area metrics for stronger identity verification
                lm = ref_res.landmark_result
                if lm.landmarks_detected and len(lm.landmark_points_2d) >= 474:
                    pts = lm.landmark_points_2d
                    # Left iris center [473], right iris center [468]
                    left_iris = pts[473]
                    right_iris = pts[468]
                    eye_dist = float(np.linalg.norm(left_iris - right_iris))
                    face_width = float(box.width + 1e-6)
                    session.baseline_eye_dist_ratio = eye_dist / face_width
                    session.baseline_face_area = float(box.width * box.height)
                    # Nose tip [1] to chin [152] vs face height
                    nose_tip = pts[1]
                    chin = pts[152]
                    nose_chin_dist = float(abs(chin[1] - nose_tip[1]))
                    session.baseline_nose_chin_ratio = nose_chin_dist / (box.height + 1e-6)
                if ref_res.gaze_result.gaze_detected:
                    # Store baseline gaze per-session (NOT in global estimator — that's a singleton bug)
                    session.baseline_gaze_resting = (ref_res.gaze_result.yaw, ref_res.gaze_result.pitch)
        except Exception:
            pass

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
    session = manager.get_or_create_session(session_id)

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
    session = manager.get_or_create_session(session_id)

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
    session = manager.get_or_create_session(session_id)

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
    gaze_res = pipeline_result.gaze_result

    # --- Per-session gaze baseline correction ---
    # The gaze estimator is a shared singleton so we must NOT mutate its baseline.
    # Instead, subtract each session's stored resting offset from the raw angles here.
    corrected_gaze_yaw: float = gaze_res.yaw
    corrected_gaze_pitch: float = gaze_res.pitch
    if session.baseline_gaze_resting is not None and gaze_res.gaze_detected:
        b_yaw, b_pitch = session.baseline_gaze_resting
        corrected_gaze_yaw = gaze_res.yaw - b_yaw
        corrected_gaze_pitch = gaze_res.pitch - b_pitch

    gaze_yaw_thresh = manager.vision_pipeline.gaze_estimator.yaw_threshold
    gaze_pitch_thresh = manager.vision_pipeline.gaze_estimator.pitch_threshold
    is_looking_away_gaze = (
        gaze_res.gaze_detected
        and (abs(corrected_gaze_yaw) > gaze_yaw_thresh or abs(corrected_gaze_pitch) > gaze_pitch_thresh)
    )

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
    is_looking_away = is_looking_away_gaze
    if obs.head_pose and obs.head_pose.is_looking_away():
        is_looking_away = True

    phone_detected = pipeline_result.object_result.phone_detected
    laptop_detected = pipeline_result.object_result.laptop_detected
    secondary_person = pipeline_result.object_result.secondary_person_detected or obs.face_count > 1

    # --- Multi-metric examinee identity verification ---
    # Uses 3 landmark-geometry signals; requires ≥2 to deviate to flag mismatch.
    examinee_verified = True
    has_any_baseline = (
        session.baseline_face_aspect_ratio is not None
        or session.baseline_eye_dist_ratio is not None
        or session.baseline_face_area is not None
    )
    if has_any_baseline and obs.face_detected and obs.primary_face_box:
        lm_now = pipeline_result.landmark_result
        deviation_count = 0

        # Metric 1: Bounding box aspect ratio (width/height) — tightened to 20%
        curr_ratio = obs.primary_face_box.width / (obs.primary_face_box.height + 1e-6)
        if session.baseline_face_aspect_ratio is not None:
            ratio_diff = abs(curr_ratio - session.baseline_face_aspect_ratio) / (session.baseline_face_aspect_ratio + 1e-6)
            if ratio_diff > 0.20:
                deviation_count += 1

        # Metric 2: Normalized inter-eye distance (eye_dist / face_width)
        if (
            session.baseline_eye_dist_ratio is not None
            and lm_now.landmarks_detected
            and len(lm_now.landmark_points_2d) >= 474
        ):
            pts = lm_now.landmark_points_2d
            left_iris = pts[473]
            right_iris = pts[468]
            curr_eye_dist = float(np.linalg.norm(left_iris - right_iris))
            curr_eye_ratio = curr_eye_dist / (obs.primary_face_box.width + 1e-6)
            eye_ratio_diff = abs(curr_eye_ratio - session.baseline_eye_dist_ratio) / (session.baseline_eye_dist_ratio + 1e-6)
            if eye_ratio_diff > 0.22:
                deviation_count += 1

        # Metric 3: Face area ratio — detects gross size changes (distance change = different person or covered face)
        if session.baseline_face_area is not None:
            curr_area = float(obs.primary_face_box.width * obs.primary_face_box.height)
            area_ratio = curr_area / (session.baseline_face_area + 1e-6)
            # Very different area: either tiny (covering face) or huge (different distance) — flag if < 0.35 or > 2.8
            if area_ratio < 0.35 or area_ratio > 2.8:
                deviation_count += 1

        # Require ≥2 metrics to deviate before flagging — avoids false positives from single noisy signal
        if deviation_count >= 2:
            examinee_verified = False

    # Determine decisive malpractice flag and severity
    has_active_violation = (
        len(active_events) > 0
        or phone_detected
        or secondary_person
        or (not examinee_verified)
    )

    is_malpractice = evaluation.is_malpractice_flagged or has_active_violation
    severity_val = evaluation.severity.value

    if phone_detected or not examinee_verified:
        severity_val = "CRITICAL"
        is_malpractice = True
    elif has_active_violation and severity_val == "LOW":
        severity_val = "HIGH"
        is_malpractice = True

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
        severity=severity_val,
        is_malpractice_flagged=is_malpractice,
        examinee_verified=examinee_verified,
        latency_ms=round(total_latency_ms, 2),
    )

