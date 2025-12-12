FROM continuumio/miniconda3:latest

ENV CONDA_ENV_PATH /opt/conda/envs/music-segmentation-env

ENV PATH $CONDA_ENV_PATH/bin:$PATH

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

COPY ./workers /app/workers
COPY ./shared /app/shared

# Python path needs to include /app so imports like 'from shared...' work
ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python", "workers/main.py"]