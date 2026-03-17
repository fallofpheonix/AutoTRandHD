from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile

from autotrandhd.api.schemas import InferenceResponse
from autotrandhd.config import load_settings
from autotrandhd.services.model_registry import ModelDescription
from autotrandhd.services.ocr_pipeline import InferenceResult
from autotrandhd.services import ModelRegistry, OCRPipelineService


def create_app() -> FastAPI:
    settings = load_settings()
    registry = ModelRegistry(settings)
    service = OCRPipelineService(settings, registry)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        registry.load_if_present()
        yield

    app = FastAPI(title=settings.api_title, lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, bool | str]:
        return {"status": "healthy", "model_loaded": registry.is_loaded()}

    @app.get("/model_info")
    async def model_info() -> ModelDescription:
        return registry.describe()

    @app.post("/infer", response_model=InferenceResponse)
    async def infer(file: UploadFile = File(...)) -> InferenceResult:
        try:
            payload = await file.read()
            return service.transcribe_bytes(payload, file.filename or "upload")
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/batch", response_model=List[InferenceResponse])
    async def batch_infer(files: List[UploadFile] = File(...)) -> list[InferenceResult]:
        responses: list[InferenceResult] = []
        for file in files:
            payload = await file.read()
            try:
                responses.append(service.transcribe_bytes(payload, file.filename or "upload"))
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except ValueError as exc:
                responses.append(
                    {
                        "image": file.filename or "upload",
                        "text": "",
                        "confidence": 0.0,
                        "latency_ms": 0,
                    }
                )
                # TODO: wire this into structured request logging once API usage settles.
                _ = exc
        return responses

    return app


app = create_app()
