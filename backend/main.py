from fastapi import FastAPI
import uvicorn

from core import logger

app = FastAPI()
  
@app.get("/probe")
def probe():
    logger.info("Probe called!")
    return {"status": "app running!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)