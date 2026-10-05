import React, { useRef, useEffect, useState } from "react";
import { Camera, Mic, AlertTriangle } from "lucide-react";

export const ProctorCamera = ({
  socket,
  attemptId,
  sessionId,
  strikes = 0,
  maxStrikes = 3,
  onWarningIssued,
  onForceTerminate,
}) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [streamActive, setStreamActive] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);

  useEffect(() => {
    let stream = null;
    let frameInterval = null;
    let audioContext = null;
    let analyser = null;
    let animFrame = null;

    const startMediaCapture = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { max: 15 } },
          audio: true,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }
        setStreamActive(true);

        // Setup Audio Analyser for VU Meter
        try {
          audioContext = new (window.AudioContext || window.webkitAudioContext)();
          analyser = audioContext.createAnalyser();
          const source = audioContext.createMediaStreamSource(stream);
          source.connect(analyser);
          analyser.fftSize = 64;
          const dataArray = new Uint8Array(analyser.frequencyBinCount);

          const updateVolume = () => {
            analyser.getByteFrequencyData(dataArray);
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
              sum += dataArray[i];
            }
            const average = sum / dataArray.length;
            setAudioLevel(Math.min(100, Math.round((average / 128) * 100)));
            animFrame = requestAnimationFrame(updateVolume);
          };
          updateVolume();
        } catch (audioErr) {
          console.warn("Audio meter setup skipped:", audioErr);
        }

        // Setup Audio Chunks via MediaRecorder (every 2.5s)
        try {
          const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
          mediaRecorderRef.current = recorder;

          recorder.ondataavailable = async (e) => {
            if (e.data && e.data.size > 0 && socket && socket.connected) {
              const reader = new FileReader();
              reader.onloadend = () => {
                const base64Audio = reader.result.split(",")[1];
                socket.emit("audio_chunk", {
                  attemptId,
                  sessionId,
                  audioBase64: base64Audio,
                  timestamp: Date.now(),
                });
              };
              reader.readAsDataURL(e.data);
            }
          };

          recorder.start(2500); // 2.5s slices
        } catch (recErr) {
          console.warn("MediaRecorder audio slice init error:", recErr);
        }

        // Setup Canvas Snapshot Interval (2.0 FPS = every 500ms at 640x480)
        let frameIndex = 0;
        frameInterval = setInterval(() => {
          if (!videoRef.current || !canvasRef.current || !socket || !socket.connected) return;

          const video = videoRef.current;
          const canvas = canvasRef.current;
          const ctx = canvas.getContext("2d");

          if (video.videoWidth > 0 && video.videoHeight > 0) {
            canvas.width = 640;
            canvas.height = 480;
            ctx.drawImage(video, 0, 0, 640, 480);

            // Compress to JPEG 0.65 quality for crisp eye landmarks and object edge detection
            const frameBase64 = canvas.toDataURL("image/jpeg", 0.65);

            socket.emit("media_frame", {
              attemptId,
              sessionId,
              frameBase64,
              frameIndex: frameIndex++,
              timestamp: Date.now(),
            });
          }
        }, 500);

      } catch (err) {
        console.error("Camera & Mic capture error:", err);
      }
    };

    startMediaCapture();

    return () => {
      if (frameInterval) clearInterval(frameInterval);
      if (animFrame) cancelAnimationFrame(animFrame);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      if (audioContext && audioContext.state !== "closed") {
        audioContext.close();
      }
    };
  }, [attemptId, sessionId, socket]);

  // Setup Socket Listeners for 3-Strike Events
  useEffect(() => {
    if (!socket) return;

    const handleWarning = (data) => {
      if (onWarningIssued) onWarningIssued(data);
    };

    const handleTerminate = (data) => {
      if (onForceTerminate) onForceTerminate(data);
    };

    socket.on("warning_issued", handleWarning);
    socket.on("force_terminate", handleTerminate);

    return () => {
      socket.off("warning_issued", handleWarning);
      socket.off("force_terminate", handleTerminate);
    };
  }, [socket, onWarningIssued, onForceTerminate]);

  return (
    <div style={{
      position: "fixed",
      bottom: "24px",
      right: "24px",
      width: "240px",
      zIndex: 50,
      background: "#161616",
      border: `2px solid ${strikes > 0 ? "var(--warning)" : "var(--border-strong)"}`,
      borderRadius: "var(--radius-lg)",
      overflow: "hidden",
      boxShadow: strikes > 0 ? "var(--shadow-danger)" : "var(--shadow-lg)",
      transition: "all var(--transition-normal)",
    }}>
      {/* Hidden offscreen canvas for snapshot generation */}
      <canvas ref={canvasRef} style={{ display: "none" }} />

      {/* Video Container */}
      <div style={{ position: "relative", width: "100%", height: "150px", background: "#0c0c0c" }}>
        <video
          ref={videoRef}
          muted
          playsInline
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            transform: "scaleX(-1)", // Mirror video
          }}
        />

        {/* Radar Scanning Beam Overlay */}
        <div className="radar-beam" />

        {/* Live Proctoring Badge */}
        <div style={{
          position: "absolute",
          top: "8px",
          left: "8px",
          background: "rgba(18, 18, 18, 0.8)",
          backdropFilter: "blur(8px)",
          padding: "3px 8px",
          borderRadius: "var(--radius-full)",
          display: "flex",
          alignItems: "center",
          gap: "5px",
          fontSize: "0.68rem",
          fontWeight: 600,
          color: "#34d399",
        }}>
          <span style={{
            width: "6px",
            height: "6px",
            borderRadius: "50%",
            background: "#34d399",
            boxShadow: "0 0 8px #34d399",
          }} />
          PROCTORED
        </div>

        {/* Strike Indicator */}
        <div style={{
          position: "absolute",
          top: "8px",
          right: "8px",
          background: strikes > 0 ? "rgba(239, 68, 68, 0.95)" : "rgba(18, 18, 18, 0.8)",
          color: "#fff",
          padding: "3px 8px",
          borderRadius: "var(--radius-full)",
          fontSize: "0.7rem",
          fontWeight: 700,
          display: "flex",
          alignItems: "center",
          gap: "4px",
        }}>
          <AlertTriangle size={11} />
          {strikes} / {maxStrikes} STRIKES
        </div>
      </div>

      {/* Bottom Sensor Panel */}
      <div style={{
        padding: "8px 12px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        fontSize: "0.75rem",
        color: "var(--text-secondary)",
        borderTop: "1px solid var(--border-subtle)",
        background: "#161616",
      }}>
        {/* Camera Sensor Status */}
        <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
          <Camera size={13} color="#c084fc" />
          <span>Vision (1.5 FPS)</span>
        </div>

        {/* Mic Sensor VU Meter */}
        <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
          <Mic size={13} color={audioLevel > 30 ? "var(--warning)" : "var(--success)"} />
          <div style={{
            width: "36px",
            height: "6px",
            background: "#222222",
            borderRadius: "3px",
            overflow: "hidden",
          }}>
            <div style={{
              width: `${audioLevel}%`,
              height: "100%",
              background: audioLevel > 50 ? "var(--danger)" : audioLevel > 25 ? "var(--warning)" : "var(--success)",
              transition: "width 0.1s ease-out",
            }} />
          </div>
        </div>
      </div>
    </div>
  );
};
