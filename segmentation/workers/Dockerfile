FROM continuumio/miniconda3:latest

ENV CONDA_ENV_PATH /opt/conda/envs/music-segmentation-worker-env

ENV PATH $CONDA_ENV_PATH/bin:$PATH

WORKDIR /app

COPY segmentation/workers/environment.yml .
RUN conda env create -f environment.yml

# Copy the whole segmentation service tree, so imports like
# 'from segmentation.core...'/'...infrastructure...'/'...application...' resolve.
COPY ./segmentation /app/segmentation

# Python path needs to include /app so imports resolve
ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["python", "segmentation/workers/main.py"]