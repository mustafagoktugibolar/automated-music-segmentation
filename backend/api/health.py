# backend/api/health.py
from fastapi import APIRouter, Request
from backend.core.db import ping_db
from backend.core import logger
import os
import platform
import time

router = APIRouter(prefix="/health", tags=["health"])
START_TIME = time.time()

@router.get("/app")
def health_app():
    uptime = round(time.time() - START_TIME, 1)
    return {
        "app_ok": True,
        "message": "Application is running",
        "uptime_seconds": uptime,
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "python_version": platform.python_version(),
        "environment": os.getenv("APP_ENV", "development"),
    }


@router.get("/db")
def health_db(request: Request):
    try:
        db_ok = ping_db(request)
    except Exception as e:
        db_ok = False
        logger.error("Database health check failed.", exception=e)

    return {"db_ok": db_ok}


@router.get("")
def health_root(request: Request):
    uptime = round(time.time() - START_TIME, 1)

    try:
        db_ok = ping_db(request)
    except Exception as e:
        db_ok = False
        logger.error("Database check failed in /health.", exception=e)

    return {
        "app_ok": True,
        "db_ok": db_ok,
        "uptime_seconds": uptime,
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "python_version": platform.python_version(),
        "environment": os.getenv("APP_ENV", "development"),
    }
