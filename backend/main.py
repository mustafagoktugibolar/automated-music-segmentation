import uvicorn

from fastapi import FastAPI, Request
from backend.core import logger
from backend.core.db import register_db, ping_db
from backend.api.health import router as health_router
from backend.api.segmentation import router as segmentation_router
app = FastAPI()

register_db(app)

app.include_router(health_router)
app.include_router(segmentation_router)

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)