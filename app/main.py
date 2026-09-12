from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="ClaimGraph – Claim Verifier API",
    description="API for scientific claim extraction, verification, and graph-based analysis.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
from app.api.routes import verify as verify_router  # noqa: E402

app.include_router(verify_router.router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness probe."""
    return {"status": "ok"}
