import uvicorn

from fastapi import FastAPI, Request
from shared.logger import get_logger
from backend.api.health import router as health_router
from backend.api.segmentation import router as segmentation_router

from contextlib import asynccontextmanager

logger = get_logger()

# Initialize Result Listener
from backend.services.result_listener import ResultListener
result_listener = ResultListener()

from backend.db.postgreSQL import init_db_pool, close_db_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up application...")
    
    # Initialize DB Pool
    await init_db_pool(app)
    
    # Create Tables
    from backend.db.postgreSQL import engine
    from backend.db.models import Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created.")

    # Start Result Listener
    result_listener.start()
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await close_db_pool(app)

app = FastAPI(lifespan=lifespan)

# Add CORS Middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Request Processed: {request.method} {request.url} -> {response.status_code}")
    return response

app.include_router(health_router)
app.include_router(segmentation_router)

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)