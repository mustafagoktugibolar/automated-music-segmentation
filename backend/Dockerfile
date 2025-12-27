FROM continuumio/miniconda3:latest

ENV CONDA_ENV_PATH /opt/conda/envs/music-segmentation-env

ENV PATH $CONDA_ENV_PATH/bin:$PATH

WORKDIR /app

# API/Backend environment setup
COPY backend/environment.yml .
RUN conda env create -f environment.yml

# Copy source code
COPY ./backend /app/backend
COPY ./shared /app/shared

# Add /app to python path for shared modules
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Command to run FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]