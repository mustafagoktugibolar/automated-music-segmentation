FROM continuumio/miniconda3:latest

ENV CONDA_ENV_PATH /opt/conda/envs/music-segmentation-env

ENV PATH $CONDA_ENV_PATH/bin:$PATH

WORKDIR /app

# API/Backend environment setup
COPY segmentation/api/environment.yml .
RUN conda env create -f environment.yml

# Copy source code — the whole segmentation service tree, so imports like
# 'from segmentation.core...'/'...infrastructure...'/'...application...' resolve.
COPY ./segmentation /app/segmentation

# Add /app to python path
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Command to run FastAPI
CMD ["uvicorn", "segmentation.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]