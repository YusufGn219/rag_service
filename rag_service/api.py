"""HTTP API over SearchService (FastAPI). Start it with `python -m rag_service.server`."""
import hmac

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from rag_service.config import Config
from rag_service.service import MAX_RESULTS, Busy, NoIndex, SearchService, ServiceError


class SearchRequest(BaseModel):
    query: str
    k: int = Field(5, ge=1, le=MAX_RESULTS)


def _supplied_key(request: Request) -> str:
    key = request.headers.get("x-api-key")
    if key:
        return key
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def create_app(service: SearchService, cfg: Config) -> FastAPI:
    def require_key(request: Request) -> None:
        if cfg.api_key and not hmac.compare_digest(_supplied_key(request), cfg.api_key):
            raise HTTPException(status_code=401, detail="missing or wrong API key")

    app = FastAPI(title="rag_service", dependencies=[Depends(require_key)])

    @app.exception_handler(NoIndex)
    async def _no_index(_, exc: NoIndex):
        return _error(503, exc)

    @app.exception_handler(Busy)
    async def _busy(_, exc: Busy):
        return _error(409, exc)

    @app.exception_handler(ServiceError)
    async def _service_error(_, exc: ServiceError):
        return _error(404, exc)

    # Plain `def` endpoints: FastAPI runs them in a worker thread, so a slow search or
    # reindex does not block the others.
    @app.post("/search")
    def search(req: SearchRequest):
        return {"results": service.search(req.query, req.k)}

    @app.get("/note")
    def note(path: str = Query(..., min_length=1)):
        return {"path": path, "text": service.read_note(path)}

    @app.get("/health")
    def health():
        return service.status()

    @app.post("/reindex")
    def reindex():
        return service.reindex()

    return app


def _error(status: int, exc: Exception):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=status, content={"detail": str(exc)})
