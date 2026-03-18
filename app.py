from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from RAG import run_full_pipeline


class QueryRequest(BaseModel):
    question: str
    iterations: int = 3
    stop_when_satisfied: bool = True


app = FastAPI(title="RAG Multi-hop Service")


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    """Simple health check endpoint for Cloud Run / load balancers."""
    return {"status": "ok"}


@app.post("/query")
def query_rag(req: QueryRequest) -> Dict[str, Any]:
    """Run the full RAG + multihop pipeline for a user question.

    This wraps RAG.run_full_pipeline in a HTTP API suitable for Cloud Run.
    """
    result = run_full_pipeline(
        user_question=req.question,
        iterations=req.iterations,
        stop_when_satisfied=req.stop_when_satisfied,
    )
    return result
