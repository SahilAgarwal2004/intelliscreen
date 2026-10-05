"""In-memory session manager maintaining stateful temporal engines per active exam session."""

from dataclasses import dataclass, field
import threading
import time
import uuid

from intelliscreen.core.exceptions import SessionError
from intelliscreen.core.logging import get_logger
from intelliscreen.scoring.proctoring_scorer import ProctoringEvaluation, ProctoringScorer
from intelliscreen.temporal.engine import TemporalEventEngine
from intelliscreen.vision.pipeline import VisionPipeline

logger = get_logger("session_manager")


@dataclass
class ActiveSession:
    """Stateful context for an active candidate taking an exam."""

    session_id: str
    candidate_id: str
    test_id: str
    created_at: float = field(default_factory=time.time)
    temporal_engine: TemporalEventEngine = field(default_factory=TemporalEventEngine)
    scorer: ProctoringScorer = field(default_factory=ProctoringScorer)
    total_frames: int = 0
    test_incidents: list[str] = field(default_factory=list)
    latest_evaluation: ProctoringEvaluation | None = None
    is_active: bool = True


class SessionManager:
    """Thread-safe proctoring session registry."""

    _instance: "SessionManager | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._sessions: dict[str, ActiveSession] = {}
        # Shared singleton vision pipeline for inference efficiency
        self.vision_pipeline = VisionPipeline()

    @classmethod
    def get_instance(cls) -> "SessionManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def create_session(
        self, candidate_id: str, test_id: str = "mcq_test", session_id: str | None = None
    ) -> ActiveSession:
        """Initialize and register a new exam session."""
        assigned_id = session_id.strip() if session_id and session_id.strip() else str(uuid.uuid4())
        session = ActiveSession(
            session_id=assigned_id,
            candidate_id=candidate_id,
            test_id=test_id,
        )
        with self._lock:
            self._sessions[assigned_id] = session
        logger.info(f"Initialized exam session {assigned_id} for candidate {candidate_id}")
        return session

    def get_or_create_session(
        self, session_id: str, candidate_id: str = "candidate", test_id: str = "mcq_test"
    ) -> ActiveSession:
        """Retrieve existing active session or provision a new one if not found."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None and session.is_active:
                return session

        logger.info(f"Session '{session_id}' not found in registry; auto-provisioning.")
        return self.create_session(candidate_id=candidate_id, test_id=test_id, session_id=session_id)

    def get_session(self, session_id: str) -> ActiveSession:
        """Retrieve an existing session or raise SessionError."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None or not session.is_active:
            raise SessionError(f"Session '{session_id}' not found or has been completed.")
        return session

    def end_session(self, session_id: str) -> ActiveSession:
        """Mark session complete."""
        session = self.get_session(session_id)
        session.is_active = False
        logger.info(f"Closed exam session {session_id}")
        return session
