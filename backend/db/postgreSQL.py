from fastapi import FastAPI, Request, Depends
from shared.config import DBSettings
from contextlib import asynccontextmanager
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor
from typing import Generator, Optional
from shared.logger import get_logger

import psycopg2
import asyncio

logger = get_logger()

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

MIN_CONN = DBSettings.PG_MIN_CONN
MAX_CONN = DBSettings.PG_MAX_CONN
RETRY_COUNT = DBSettings.PG_RETRY_COUNT
RETRY_WAIT = DBSettings.PG_RETRY_WAIT
STMT_TIMEOUT_MS = DBSettings.PG_STATEMENT_TIMEOUT_MS

# SQLAlchemy Engine and Session for ORM usage
engine = create_engine(DBSettings.DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def prepare_db_connection(conn) -> None:
    conn.autocommit = False
    with conn.cursor() as cursor:
        cursor.execute("SET application_name = %s;", ("fastapi-app",))
        cursor.execute("SET statement_timeout = %s;", (str(STMT_TIMEOUT_MS),))
        logger.info(f"Prepared database connection with {STMT_TIMEOUT_MS} statement_timeout.")


async def init_db_pool(app: FastAPI):
    if not getattr(DBSettings, "DB_URL", None):
        logger.error("DB_URL is not set in settings.")
        raise RuntimeError("DB_URL is required in settings.")
    
    pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
    last_exc: Optional[Exception] = None
    
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            logger.info(f"[DB] Creating pool attempt {attempt}/{RETRY_COUNT} ...")
            pool = ThreadedConnectionPool(
                minconn=MIN_CONN,
                maxconn=MAX_CONN,
                dsn=DBSettings.DB_URL
            )
            conn = pool.getconn()
            try:
                prepare_db_connection(conn)
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1;")
                    cursor.fetchone()
                    logger.info("[DB] Connection pool created and tested successfully.")
            finally:
                pool.putconn(conn)
                
            logger.info("[DB] Connection pool ready.")
            break
        except Exception as e:
            last_exc = e
            logger.error(f"[DB] Pool creation failed on attempt {attempt}/{RETRY_COUNT}. Retrying in {RETRY_WAIT} seconds...", exc_info=True)
            await asyncio.sleep(RETRY_WAIT)
            
    if pool is None:
        raise RuntimeError(f"[DB] Could not initialize pool: {last_exc}")
    
    app.state.db_pool = pool

async def close_db_pool(app: FastAPI):
    try:
        pool = getattr(app.state, "db_pool", None)
        if pool:
            pool.closeall()
            logger.info("[DB] Connection pool closed.")
    except Exception as e:
        logger.error("[DB] Error closing connection pool.", exc_info=True)


def get_db_conn(request: Request) -> Generator:
    pool: ThreadedConnectionPool = request.app.state.db_pool
    conn = pool.getconn()
    try:
        prepare_db_connection(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
        
def get_cursor(conn = Depends(get_db_conn)):
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        yield cursor

def ping_db(request: Request) -> bool:
    pool: ThreadedConnectionPool = request.app.state.db_pool
    conn = pool.getconn()
    try:
        prepare_db_connection(conn)
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1;")
            cursor.fetchone()
        return True
    finally:
        pool.putconn(conn)
