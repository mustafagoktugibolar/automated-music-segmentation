FROM continuumio/miniconda3:latest

ENV CONDA_ENV_PATH /opt/conda/envs/music-segmentation-worker-env

ENV PATH $CONDA_ENV_PATH/bin:$PATH

WORKDIR /app

COPY workers/environment.yml .
RUN conda env create -f environment.yml

COPY ./workers /app/workers
COPY ./shared /app/shared
COPY ./backend /app/backend

# Python path needs to include /app so imports like 'from shared...' work
ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python", "workers/main.py"]