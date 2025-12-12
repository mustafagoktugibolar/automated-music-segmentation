# Music Segmentation Project

This project is an automated music segmentation tool.

## Getting Started

This project uses Docker and Docker Compose to manage all its services, including the backend API and the PostgreSQL database.

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
    You should see both `music_segmentation_db` and `music_segmentation_backend` with a status of "running" or "Up".

3.  **Access the API**

    Once the services are running, the API will be available at `http://localhost:8000`. You can test it by navigating to the health check endpoint:
    `http://localhost:8000/probe`

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