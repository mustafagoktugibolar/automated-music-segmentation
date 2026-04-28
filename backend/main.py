from dotenv import load_dotenv

load_dotenv()


import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.api.algorithms import router as algorithms_router
from backend.api.datasets import router as datasets_router
from backend.api.evaluation import router as evaluation_router
from backend.api.health import router as health_router
from backend.api.segmentation import router as segmentation_router
from backend.api.songs import router as songs_router
from backend.db.postgreSQL import close_db_pool, init_db_pool
from backend.services.result_listener import ResultListener
from shared.logger import get_logger

logger = get_logger()

result_listener = ResultListener()


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application...")
    await init_db_pool(app)

    from backend.db.models import Base
    from backend.db.postgreSQL import engine

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created.")

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
app.include_router(algorithms_router)
app.include_router(datasets_router)
app.include_router(evaluation_router)


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
