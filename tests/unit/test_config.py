"""Unit tests for configuration loading and validation."""

import pytest
from pydantic import ValidationError

from intelliscreen.config.settings import Settings, get_settings


def test_default_settings_values():
    """Verify default thresholds align with system design requirements."""
    settings = Settings()

    assert settings.env == "development"
    assert settings.log_level == "INFO"
    assert settings.vision_target_fps == 10.0
    assert settings.vision_frame_width == 640
    assert settings.vision_frame_height == 480
    assert settings.gaze_yaw_threshold == 25.0
    assert settings.temporal_normal_look_away_max_sec == 2.5
    assert settings.temporal_suspicious_look_away_min_sec == 4.0
    assert settings.temporal_phone_presence_alert_sec == 2.0


def test_settings_environment_override(monkeypatch):
    """Verify that INTELLISCREEN_ prefixed environment variables override defaults."""
    monkeypatch.setenv("INTELLISCREEN_ENV", "production")
    monkeypatch.setenv("INTELLISCREEN_VISION_TARGET_FPS", "15.0")
    monkeypatch.setenv("INTELLISCREEN_GAZE_YAW_THRESHOLD", "30.0")

    settings = Settings()
    assert settings.env == "production"
    assert settings.vision_target_fps == 15.0
    assert settings.gaze_yaw_threshold == 30.0


def test_settings_validation_errors():
    """Verify bounds validation on settings fields."""
    # Target FPS must be between 1.0 and 60.0
    with pytest.raises(ValidationError):
        Settings(vision_target_fps=0.0)

    # Confidence must be between 0.0 and 1.0
    with pytest.raises(ValidationError):
        Settings(face_confidence_threshold=1.5)

    # Frame dimensions must be positive integers
    with pytest.raises(ValidationError):
        Settings(vision_frame_width=-100)


def test_get_settings_lru_cache():
    """Verify get_settings returns consistent cached singleton instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
