FROM continuumio/miniconda3:latest

ENV CONDA_ENV_PATH /opt/conda/envs/music-segmentation-env

ENV PATH $CONDA_ENV_PATH/bin:$PATH

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml

COPY ./backend /app/backend

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]