#!/usr/bin/env python3
"""End-to-end smoke test for the Unified Vision Perception Pipeline Orchestrator.

Validates the complete perception stack:
  Preprocessing -> Face Detection -> 478 Landmark Mesh ->
  3D Head Pose (PnP) -> Gaze Ray Vectorization -> YOLOv8n Object Detection ->
  FrameObservation Domain Model Aggregation -> Annotated HUD Visualization.
"""

from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from intelliscreen.core.schemas import FrameObservation
from intelliscreen.vision.pipeline import (
    VisionPipeline,
    VisionPipelineConfig,
    VisionPipelineResult,
)


def run_smoke_test() -> None:
    print("=" * 78)
    print("      INTELLISCREEN UNIFIED VISION PIPELINE - END-TO-END SMOKE TEST")
    print("=" * 78)

    fixture_path = ROOT_DIR / "tests" / "fixtures" / "sample_face.jpg"
    if not fixture_path.exists():
        print(f"ERROR: Fixture image not found at {fixture_path}")
        sys.exit(1)

    image = cv2.imread(str(fixture_path))
    if image is None:
        print(f"ERROR: Failed to load fixture from {fixture_path}")
        sys.exit(1)

    print(f"\n1. Ingested Fixture Image: shape={image.shape}, dtype={image.dtype}")

    # Initialize VisionPipeline with default settings (objects enabled, stride=3)
    config = VisionPipelineConfig(
        target_width=640,
        target_height=480,
        target_fps=10.0,
        enable_objects=True,
        object_stride=3,
        face_confidence=0.5,
        object_confidence=0.4,
    )
    print(
        f"2. Initializing VisionPipeline (res={config.target_width}x{config.target_height}, "
        f"fps={config.target_fps}, stride={config.object_stride})..."
    )
    pipeline = VisionPipeline(config=config)
    print("   VisionPipeline initialized successfully.")

    # Execute a 5-frame sequence to demonstrate stride caching and domain observation
    print("\n3. Processing 5-Frame Sequence through Pipeline:")
    results: list[VisionPipelineResult] = []
    latencies: list[float] = []

    for idx in range(5):
        t0 = time.perf_counter()
        res = pipeline.process_frame(image)
        lat = (time.perf_counter() - t0) * 1000.0
        results.append(res)
        latencies.append(lat)

        obs = res.observation
        cached_str = "CACHED" if res.object_result.is_cached else "INFERENCE"
        pose_str = (
            f"Y={obs.head_pose.yaw:+.1f}°, P={obs.head_pose.pitch:+.1f}°, R={obs.head_pose.roll:+.1f}°"
            if obs.head_pose
            else "N/A"
        )
        gaze_str = (
            f"{obs.gaze.direction.value} (Y={obs.gaze.yaw:+.1f}°, P={obs.gaze.pitch:+.1f}°)"
            if obs.gaze
            else "N/A"
        )
        print(
            f"   Frame #{obs.frame_index:02d} | TS={obs.timestamp_ms:6.1f}ms | "
            f"Latency: {lat:5.1f}ms | YOLO: {cached_str:9s} | Head: {pose_str} | Gaze: {gaze_str}"
        )

    # Frame 0 Detailed Verification
    res0 = results[0]
    obs0 = res0.observation
    print("\n4. FrameObservation Schema Validation (Frame #0):")
    print(f"   - Observation Type: {type(obs0).__name__}")
    print(f"   - Face Detected: {obs0.face_detected} (count={obs0.face_count})")
    assert obs0.face_detected, "Assertion failed: Face was not detected in fixture."
    assert obs0.primary_face_box is not None, "Assertion failed: primary_face_box is None."
    print(
        f"   - Primary Face Box: [{obs0.primary_face_box.xmin:.1f}, {obs0.primary_face_box.ymin:.1f}, "
        f"{obs0.primary_face_box.xmax:.1f}, {obs0.primary_face_box.ymax:.1f}] "
        f"(confidence={obs0.primary_face_box.confidence:.2f})"
    )

    assert obs0.head_pose is not None, "Assertion failed: Head pose angles are None."
    print(
        f"   - Head Pose Angles: Yaw={obs0.head_pose.yaw:.2f}°, Pitch={obs0.head_pose.pitch:.2f}°, "
        f"Roll={obs0.head_pose.roll:.2f}° (looking_away={res0.head_pose_result.is_looking_away})"
    )

    assert obs0.gaze is not None, "Assertion failed: Gaze vector is None."
    print(
        f"   - Gaze Vector: Direction={obs0.gaze.direction.value}, Yaw={obs0.gaze.yaw:.2f}°, "
        f"Pitch={obs0.gaze.pitch:.2f}° (is_deviated={res0.gaze_result.is_deviated})"
    )

    print(f"   - Detected Objects: {len(obs0.detected_objects)} item(s)")
    for obj in obs0.detected_objects:
        print(f"       * {obj.label}: conf={obj.confidence:.2f}, box={obj.box.center}")

    # Latency Breakdown Table
    print("\n5. Component Latency Breakdown (Frame #0):")
    for comp, c_lat in res0.component_latencies.items():
        print(f"   - {comp:16s}: {c_lat:6.2f} ms")
    print(f"   - TOTAL LATENCY   : {res0.total_latency_ms:6.2f} ms")

    # Stride Caching Performance Check
    cached_lats = [latencies[1], latencies[2], latencies[4]]
    uncached_lats = [latencies[0], latencies[3]]
    avg_cached = sum(cached_lats) / len(cached_lats)
    avg_uncached = sum(uncached_lats) / len(uncached_lats)
    amortized = sum(latencies) / len(latencies)
    print("\n6. Performance Optimization & Stride Caching Summary:")
    print(f"   - Uncached (Full YOLO) Latency : {avg_uncached:.2f} ms")
    print(f"   - Stride Cached Frame Latency  : {avg_cached:.2f} ms")
    print(f"   - 5-Frame Amortized Latency    : {amortized:.2f} ms")

    # Degraded / Absence Frame Test
    print("\n7. Testing Absence / Dark Frame Resilience:")
    dark_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res_dark = pipeline.process_frame(dark_frame)
    print(
        f"   - Dark Frame: face_detected={res_dark.observation.face_detected}, "
        f"is_low_light={res_dark.preprocessed.quality.is_low_light}, "
        f"is_suspicious={res_dark.is_suspicious_instant}"
    )
    assert not res_dark.observation.face_detected, "Dark frame falsely detected a face."
    assert res_dark.is_suspicious_instant, "Dark frame must trigger instantaneous suspicion."

    # Render Visual Annotation Canvas
    print("\n8. Rendering Multi-Modal Annotated Perception Canvas:")
    annotated = pipeline.annotate_frame(
        result=res0,
        draw_face_box=True,
        draw_pose_axes=True,
        draw_gaze_rays=True,
        draw_objects=True,
        draw_hud=True,
    )
    artifact_path = ROOT_DIR / "tests" / "fixtures" / "annotated_pipeline_demo.jpg"
    cv2.imwrite(str(artifact_path), annotated)
    print(f"   - Saved annotated pipeline demo to: {artifact_path} ({artifact_path.stat().st_size} bytes)")

    # Diagnostics
    diag = pipeline.get_diagnostics()
    print("\n9. Pipeline Diagnostics:")
    for k, v in diag.items():
        print(f"   - {k}: {v}")

    print("\n" + "=" * 78)
    print("SMOKE TEST SUCCESSFUL: Unified Vision Pipeline functioning with full fidelity!")
    print("=" * 78)


if __name__ == "__main__":
    run_smoke_test()
