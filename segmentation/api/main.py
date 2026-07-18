from dotenv import load_dotenv

load_dotenv()


import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from segmentation.api.routes.datasets import router as datasets_router
from segmentation.api.routes.evaluation import router as evaluation_router
from segmentation.api.routes.health import router as health_router
from segmentation.api.routes.segmentation import router as segmentation_router
from segmentation.api.routes.songs import router as songs_router
from segmentation.infrastructure.storage.postgres import close_db_pool, init_db_pool
from segmentation.application.orchestration.result_listener import ResultListener
from segmentation.infrastructure.logging import get_logger

logger = get_logger()

result_listener = ResultListener()


from contextlib import asynccontextmanager


def _mark_stale_jobs_failed():
    """Mark any BatchEvalJob still 'running' at startup as failed (server was restarted mid-run)."""
    from segmentation.infrastructure.storage.db_models import BatchEvalJob
    from segmentation.infrastructure.storage.postgres import SessionLocal
    from datetime import datetime, timezone
    db = SessionLocal()
    try:
        stale = db.query(BatchEvalJob).filter(BatchEvalJob.status == "running").all()
        for job in stale:
            job.status = "failed"
            job.error = "server_restart"
            job.completed_at = datetime.now(timezone.utc)
        if stale:
            db.commit()
            logger.info("Marked %d stale batch job(s) as failed (server restart).", len(stale))
    except Exception as exc:
        db.rollback()
        logger.warning("Could not mark stale jobs: %s", exc)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application...")
    await init_db_pool(app)

    from segmentation.infrastructure.storage.db_models import Base
    from segmentation.infrastructure.storage.postgres import engine

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created.")

    _mark_stale_jobs_failed()

    result_listener.start()
    yield

    logger.info("Shutting down...")
    await close_db_pool(app)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Request Processed: {request.method} {request.url} -> {response.status_code}")
    return response


app.include_router(health_router)
app.include_router(segmentation_router)
app.include_router(songs_router)
app.include_router(datasets_router)
app.include_router(evaluation_router)


if __name__ == "__main__":
    uvicorn.run("segmentation.api.main:app", host="0.0.0.0", port=8000, reload=True)
