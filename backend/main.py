from dotenv import load_dotenv

load_dotenv()

import argparse
import asyncio

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Segmentation Backend")
    parser.add_argument("--sync-data", action="store_true", help="Run dataset worker pipeline to sync audio files")
    args, _ = parser.parse_known_args()

    if args.sync_data:

        async def run_dataset_worker():
            logger.info("Running dataset worker pipeline...")
            from backend.services.dataset_worker import (
                create_song_list,
                normalize_archive_paths,
                process_all_songs,
            )

            song_list = create_song_list()
            normalize_archive_paths(song_list)
            await process_all_songs(song_list, concurrency=4)
            logger.info("Dataset worker pipeline finished.")

        asyncio.run(run_dataset_worker())
    else:
        uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
