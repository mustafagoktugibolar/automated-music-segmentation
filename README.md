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