from pydantic import BaseModel


class InferenceResponse(BaseModel):
    image: str
    text: str
    confidence: float
    latency_ms: int
