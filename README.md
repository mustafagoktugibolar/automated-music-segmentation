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
    git clone https://github.com/mustafagoktugibolar/automated-music-segmentation.git
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

### Full Dataset Batch Evaluation

The frontend `Batch Eval` page includes a `Run all dataset` option. It sends `max_tracks=0` to `POST /evaluation/batch`, so the backend evaluates every available SALAMI track found in MinIO with local annotations.

Set the frontend's batch concurrency to match the number of ready `worker-custom` pod replicas in the cluster (see [Deployment on Kubernetes](#deployment-on-kubernetes)). Keep the LLM worker at low concurrency regardless of replica count, because API latency, rate limits, and cost become the bottleneck.

## API Endpoints

- `POST /segmentation/upload`  
  Uploads an audio file and dispatches segmentation jobs.  
  Multipart fields:
  - `file`: audio file
  - `algorithms`: JSON list string (default: `["custom_librosa","foote","cnmf","scluster"]`)
  - `params`: optional JSON string with typed worker parameters

- `POST /segmentation/from-storage`  
  Runs segmentation for an existing storage song by `song_id`.

- `GET /songs`  
  Lists available songs from Azure Blob storage as `{ song_id, blob_name }`.

- `GET /segmentation/status/{task_id}`  
  Returns task status and collected per-algorithm results.

### Fusion Request Example

`custom` is still accepted as a backward-compatible alias, but the canonical deterministic algorithm is `custom_librosa`.

```bash
curl -X POST http://localhost:8000/segmentation/from-storage \
  -H "Content-Type: application/json" \
  -d '{
    "song_id": "1013",
    "algorithms": ["custom_librosa", "foote", "cnmf", "scluster", "fusion"],
    "params": {
      "fusion": {
        "merge_window_seconds": 2.5,
        "threshold": 0.45,
        "required_vote_count": 2
      }
    }
  }'
```

When `fusion` is requested, the backend first dispatches `custom_librosa`, `foote`, `cnmf`, and `scluster`. The result listener dispatches `segmentation.fusion` only after the base outputs are available.

### Batch Evaluation Example

```bash
curl -X POST http://localhost:8000/evaluation/batch \
  -H "Content-Type: application/json" \
  -d '{
    "max_tracks": 20,
    "algorithms": ["custom_librosa", "foote", "cnmf", "scluster", "fusion"],
    "tolerances": [0.5, 3.0],
    "concurrency": 3
  }'
```

The batch output includes per-track, per-algorithm rows and aggregate comparison lines for `f1_0_5`, `f1_3_0`, precision, recall, and estimated/reference boundary ratios.

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

## Deployment on Kubernetes

For production and full-dataset batch runs, the stack is deployed to a Kubernetes cluster instead of `docker-compose`. The backend API, frontend, PostgreSQL, RabbitMQ, MinIO, and each segmentation worker (`worker-custom`, `worker-msaf-foote`, `worker-msaf-cnmf`, `worker-msaf-scluster`, `worker-fusion`, `worker-llm`) run as their own `Deployment`/`StatefulSet` with a matching `Service`, defined under [k8s/](k8s/). See [k8s/README.md](k8s/README.md) for image build/push steps and secret setup before applying.

1.  **Apply the manifests**

    ```bash
    kubectl apply -f k8s/
    ```

2.  **Check rollout status**

    ```bash
    kubectl get pods -n music-segmentation
    ```

    You should see `backend`, `frontend`, `worker-custom`, `worker-msaf-foote`, `worker-msaf-cnmf`, `worker-msaf-scluster`, `worker-fusion`, `worker-llm`, `music-segmentation-db`, `rabbitmq`, and `minio` pods reach `Running`.

3.  **Access the API**

    Expose the backend `Service` (via `Ingress`, `LoadBalancer`, or `kubectl port-forward`) and check `/health`:

    ```bash
    kubectl port-forward -n music-segmentation svc/backend 8000:8000
    curl http://localhost:8000/health
    ```

### Scaling Workers

The `worker-custom` Deployment is scaled independently of the rest of the stack. To run four worker pods with one active task per pod:

```bash
kubectl scale deployment worker-custom --replicas=4 -n music-segmentation
```

Then set the frontend `Batch Eval` concurrency to `4` (or use the `4 workers` preset).

If CPU stays below roughly 80% and memory stays comfortable across nodes during a full run, increase per-pod concurrency via the `WORKER_CONCURRENCY` environment variable in the Deployment spec and set frontend concurrency to match (e.g. `8` for four pods at concurrency `2`).

For automatic scaling based on load, use a `HorizontalPodAutoscaler` targeting the `worker-custom` Deployment:

```bash
kubectl autoscale deployment worker-custom --cpu-percent=70 --min=2 --max=8 -n music-segmentation
```

Keep `worker-llm` at a low, fixed replica count — API latency, rate limits, and cost are the bottleneck there, not compute.

### Logs and Teardown

```bash
kubectl logs -f -n music-segmentation deployment/backend
kubectl delete -f k8s/
```

## Automated Music Segmentation Pipeline

This project implements deterministic music structure segmentation. Boundary detection is the primary output. Structural labels such as `A`, `B`, and `C` identify repeated section groups; semantic names such as `Intro`, `Verse`, and `Chorus` are optional heuristic annotations with confidence and reasons.

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

5.  **Segment Clustering & Labeling**
    *   **What:** After identifying segments, similar segments are grouped into structural labels (`A`, `B`, `C`, ...).
    *   **How:** The custom Librosa segmenter prefers SSM-based segment similarity; MSAF and fusion outputs use lightweight deterministic segment descriptors when audio is available and stable fallbacks otherwise. These labels mean "similar/repeated section," not human semantic names.

6.  **Conservative Semantic Labels**
    *   **What:** Optional labels such as `Intro`, `Verse`, `Chorus`, `Bridge`, and `Outro`.
    *   **How:** Semantic labels are assigned only when simple measurable evidence supports them, such as position, repetition, and relative energy. The system does not label the longest section as Chorus by default.

## Fusion Concepts

There are two separate fusion layers:

- **Feature-level fusion inside `custom_librosa`:** combines SSM novelty, RMS changes, onset flux, chord proxy, beat/phrase candidates, and optional lyrics candidates into one deterministic boundary set.
- **Algorithm-level fusion (`fusion`):** combines completed outputs from `custom_librosa`, MSAF Foote, MSAF CNMF, and MSAF SCluster using weighted boundary voting. Default weights are `custom_librosa=0.35`, `scluster=0.30`, `cnmf=0.20`, `foote=0.15`.

MSAF algorithms are treated as baseline boundary detectors. Their raw labels are preserved as raw algorithm output when present, but they are not treated as Verse/Chorus semantic labelers by default.

## Evaluation Metrics

Segmentation quality is measured by comparing predicted segment intervals against human-annotated ground truth segment intervals (SALAMI dataset) using `mir_eval.segment.detection(..., trim=True)`.

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

Evaluation output reports strict and lenient boundary metrics separately:

- `precision_0_5`, `recall_0_5`, `f1_0_5`
- `precision_3_0`, `recall_3_0`, `f1_3_0`
- reference/estimated segment and internal-boundary counts
- over/under-segmentation ratio

Structural labeling metrics are secondary. Semantic section names are tertiary and should only be evaluated against comparable semantic annotations.
