"""Centralized settings and configuration management for IntelliScreen."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration parameters with environment override support."""

    model_config = SettingsConfigDict(
        env_prefix="INTELLISCREEN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment & Logging
    env: Literal["development", "testing", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    debug: bool = False

    # Vision Ingestion & Processing
    vision_target_fps: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Target sampling FPS for vision pipeline inference.",
    )
    vision_frame_width: int = Field(
        default=640,
        gt=0,
        description="Standardized frame width for computer vision input.",
    )
    vision_frame_height: int = Field(
        default=480,
        gt=0,
        description="Standardized frame height for computer vision input.",
    )
    vision_object_detection_stride: int = Field(
        default=1,
        ge=1,
        description="Execute object detection every N processed frames to conserve CPU.",
    )
    face_confidence_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to accept a face detection.",
    )
    object_confidence_threshold: float = Field(
        default=0.22,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to accept an object detection (phone, book, person).",
    )

    # Angular Deviation Thresholds (Degrees)
    gaze_yaw_threshold: float = Field(
        default=16.0,
        gt=0.0,
        le=90.0,
        description="Yaw threshold (degrees) beyond which gaze is considered off-screen.",
    )
    gaze_pitch_threshold: float = Field(
        default=14.0,
        gt=0.0,
        le=90.0,
        description="Pitch threshold (degrees) beyond which gaze is considered off-screen.",
    )

    # Iris-to-Degree Scaling — controls how much iris deflection contributes vs. head pose.
    # Lower values = rely more on stable head pose, less on noisy iris position estimates.
    # Human ocular range is approx ±30° H / ±25° V but iris pixel noise warrants dampening.
    gaze_iris_yaw_scale: float = Field(
        default=45.0,
        gt=0.0,
        le=120.0,
        description="Angular multiplier mapping horizontal iris displacement to degrees.",
    )
    gaze_iris_pitch_scale: float = Field(
        default=35.0,
        gt=0.0,
        le=120.0,
        description="Angular multiplier mapping vertical iris displacement to degrees.",
    )
    head_yaw_threshold: float = Field(
        default=20.0,
        gt=0.0,
        le=90.0,
        description="Yaw threshold (degrees) for detecting head turn.",
    )
    head_pitch_threshold: float = Field(
        default=18.0,
        gt=0.0,
        le=90.0,
        description="Pitch threshold (degrees) for detecting head tilt (up/down).",
    )

    # Temporal Duration Thresholds (Seconds)
    temporal_normal_look_away_max_sec: float = Field(
        default=1.2,
        ge=0.5,
        le=10.0,
        description="Max gaze deviation duration considered natural thinking time.",
    )
    temporal_suspicious_look_away_min_sec: float = Field(
        default=1.5,
        ge=0.5,
        le=30.0,
        description="Minimum continuous gaze deviation to trigger a suspicious event.",
    )
    temporal_face_absence_alert_sec: float = Field(
        default=1.2,
        ge=0.5,
        le=30.0,
        description="Duration of missing face before triggering absence alert.",
    )
    temporal_multiple_faces_alert_sec: float = Field(
        default=0.8,
        ge=0.5,
        le=10.0,
        description="Duration of multiple face presence before triggering alert.",
    )
    temporal_phone_presence_alert_sec: float = Field(
        default=0.8,
        ge=0.5,
        le=10.0,
        description="Duration of phone visibility before triggering unauthorized device alert.",
    )
    temporal_sliding_window_sec: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Time window for aggregating recent behavioral frequencies.",
    )

    # Suspicion Scoring Thresholds (0.0 to 100.0)
    scoring_low_threshold: float = Field(
        default=30.0,
        ge=0.0,
        le=100.0,
        description="Scores above this boundary are classified as MEDIUM severity.",
    )
    scoring_medium_threshold: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Scores above this boundary are classified as HIGH severity.",
    )
    scoring_high_threshold: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Scores above this boundary are classified as CRITICAL severity.",
    )

    # NLP & Speech-to-Text
    nlp_embedding_model: str = "all-MiniLM-L6-v2"
    stt_model_size: str = "base.en"
    stt_device: Literal["cpu", "cuda", "mps"] = "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton instance of application settings."""
    return Settings()
