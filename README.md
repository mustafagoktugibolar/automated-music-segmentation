# Music Segmentation Project

This project is an automated music segmentation tool.

## Getting Started

This project uses Docker and Docker Compose to manage all services, including backend API, workers, PostgreSQL, and RabbitMQ.

### Prerequisites

-   [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running on your system.

### Installation & Setup

1.  **Clone the Repository**

    Clone the project to your local machine.
    ```bash
    git clone <https://github.com/mustafagoktugibolar/automated-music-segmentation.git>
    cd music-segmentation
    ```

2.  **Create Environment File**

    Copy the template file `.env.template` to a new file named `.env`.
    ```bash
    cp .env.template .env
    ```
    Open the `.env` file and change `DB_PASSWORD` to a password of your choice. All other default values are configured to work with Docker Compose out-of-the-box.

## Running the Application with Docker

The entire application stack (backend + database) is managed by Docker Compose.

1.  **Build and Run the Services**

    From the project's root directory, run the following command. This will build the backend Docker image and start all services in the background.
    ```bash
    docker-compose up -d --build
    ```

2.  **Check the Status**

    To see if the containers are running correctly, you can use:
    ```bash
    docker-compose ps
    ```
    You should see `music_segmentation_db`, `music_segmentation_backend`, and worker/rabbitmq services with a status of "running" or "Up".

3.  **Access the API**

    Once the services are running, the API will be available at `http://localhost:8000`. You can test it by navigating to:
    `http://localhost:8000/health`

### Full Dataset Batch Evaluation and Custom Worker Scaling

The frontend `Batch Eval` page includes a `Run all dataset` option. It sends `max_tracks=0` to `POST /evaluation/batch`, so the backend evaluates every available SALAMI track found in MinIO with local annotations.

For a 32 GB RAM / 8 core / 16 thread machine, start with four custom worker containers and one active task per container:

```bash
CUSTOM_WORKER_REPLICAS=4 WORKER_CONCURRENCY=1 docker compose up -d --build
```

This starts four `worker-custom` service instances, named by Docker Compose as `...worker-custom-1` through `...worker-custom-4`. In the frontend, set batch concurrency to `4` or use the `4 workers` preset.

If CPU stays below roughly 80% and RAM stays below roughly 24 GB during a full run, you can try two tasks per custom worker:

```bash
CUSTOM_WORKER_REPLICAS=4 WORKER_CONCURRENCY=2 docker compose up -d --build
```

Then set frontend batch concurrency to `8`. Keep the LLM worker at low concurrency because API latency, rate limits, and cost become the bottleneck.

You can also change the custom worker count without editing `.env`:

```bash
CUSTOM_WORKER_REPLICAS=2 WORKER_CONCURRENCY=1 docker compose up -d --build
```

## API Endpoints

- `POST /segmentation/upload`  
  Uploads an audio file and dispatches segmentation jobs.  
  Multipart fields:
  - `file`: audio file
  - `algorithms`: JSON list string (default: `["custom","foote","cnmf","scluster"]`)
  - `params`: optional JSON string with typed worker parameters

- `POST /segmentation/from-storage`  
  Runs segmentation for an existing storage song by `song_id`.

- `GET /songs`  
  Lists available songs from Azure Blob storage as `{ song_id, blob_name }`.

- `GET /segmentation/status/{task_id}`  
  Returns task status and collected per-algorithm results.

### Viewing Logs

To view the real-time logs from the backend service (useful for debugging):
```bash
docker-compose logs -f backend
```

### Stopping the Application

To stop all running services:
```bash
docker-compose down
```
To stop the services and remove the database volume (deleting all data):
```bash
docker-compose down -v
```

## Automated Music Segmentation Pipeline

This project implements a classic pipeline for music segmentation. The goal is to identify the structural boundaries within a piece of music (e.g., verse, chorus, bridge).

The process is broken down into the following key steps:

1.  **Feature Extraction**
    *   **What:** The raw audio signal is converted into a more meaningful representation. We extract features that capture the harmonic and melodic content of the music over time.
    *   **How:** We are using **Chroma Features**, which represent the intensity of each of the 12 pitch classes (C, C#, D, etc.) in the audio. This gives us a compact "fingerprint" of the harmony at each moment.

2.  **Self-Similarity Matrix (SSM) Creation**
    *   **What:** A square matrix that compares every part of the song to every other part. If two moments in the song have similar features, the corresponding cell in the matrix will have a high value.
    *   **How:** We calculate the cosine similarity between the chroma feature vectors of every pair of time frames. The resulting matrix visually reveals the song's structure, showing repetitions, verses, and choruses as patterns (lines, squares).

3.  **Novelty Curve Calculation**
    *   **What:** A one-dimensional curve that represents the likelihood of a structural boundary occurring at each point in time. Peaks in this curve signify moments of significant change in the music.
    *   **How:** We slide a "checkerboard" kernel along the diagonal of the Self-Similarity Matrix. The correlation between the kernel and the matrix at each position gives us the novelty score. High scores occur when the music transitions from one section to a dissimilar one.

4.  **Boundary Detection**
    *   **What:** The process of identifying the exact timestamps of the segment boundaries from the novelty curve.
    *   **How:** We find the peaks in the novelty curve. These peaks correspond to the most significant changes in the song's structure and are selected as our segment boundaries.

5.  **Segment Clustering & Labeling (Optional)**
    *   **What:** After identifying the segments, we can group similar-sounding segments together.
    *   **How:** By analyzing the features within each segment, we can cluster them. For example, all segments corresponding to the chorus should have similar features and will be grouped into the same cluster, which can then be labeled "Chorus".

## Evaluation Metrics

Segmentation quality is measured by comparing predicted boundaries against human-annotated ground truth boundaries (SALAMI dataset) using `mir_eval`.

### Tolerance
The time window (in seconds) within which a predicted boundary must fall to be counted as a correct detection. Two standard tolerances are used:
- **±0.5s** — strict: rewards precise boundary placement
- **±3s** — lenient: rewards finding the right region even if timing is slightly off

### Precision
Of all boundaries **predicted** by the algorithm, the fraction that are within tolerance of a true boundary.

```
Precision = True Positives / (True Positives + False Positives)
```

A low precision means the algorithm is over-segmenting — predicting boundaries that don't correspond to real structural changes.

### Recall
Of all **annotated** ground-truth boundaries, the fraction that were successfully detected by the algorithm.

```
Recall = True Positives / (True Positives + False Negatives)
```

A low recall means the algorithm is under-segmenting — missing real structural boundaries.

### F1
The harmonic mean of Precision and Recall. This is the primary summary metric.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 balances both concerns: an algorithm that predicts a boundary every 0.1 seconds achieves perfect recall but near-zero precision, and F1 penalizes this. A good segmentation algorithm needs both.
