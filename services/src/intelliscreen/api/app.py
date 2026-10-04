"""FastAPI application factory for IntelliScreen AI Proctoring Service."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from intelliscreen.api.routes import router


def create_app() -> FastAPI:
    """Instantiate and configure the IntelliScreen FastAPI application."""
    app = FastAPI(
        title="IntelliScreen — AI Proctoring Service",
        description=(
            "Real-time, context-aware AI proctoring backend for online MCQ & text-based assessments. "
            "Ingests WebRTC video frames and outputs instant behavioral alerts and explainable suspicion scores."
        ),
        version="0.1.0",
    )

    # Enable CORS for WebRTC frontend clients (React, Vue, vanilla JS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
