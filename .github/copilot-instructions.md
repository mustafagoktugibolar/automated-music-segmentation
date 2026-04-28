# Project Guidelines

## Code Style
- **Python**: Follows PEP8 with type hints where possible. See [backend/api/segmentation.py](../backend/api/segmentation.py) and [workers/segmenters/segmentation_service.py](../workers/segmenters/segmentation_service.py) for typical structure and formatting.
- **Frontend (Svelte/JS)**: Use idiomatic Svelte and modern JS. See [frontend/music-segmentation-ui/src/App.svelte](../frontend/music-segmentation-ui/src/App.svelte) for component patterns.
- **YAML/Config**: Indent with 2 spaces. See [docker-compose.yml](../docker-compose.yml) and [backend/environment.yml](../backend/environment.yml).

## Architecture
- **Monorepo**: Contains backend (FastAPI), workers (segmentation algorithms), frontend (Svelte), and shared utilities.
- **Backend**: FastAPI app in [backend/main.py](../backend/main.py), exposes REST endpoints for segmentation and song management.
- **Workers**: Python processes (see [workers/](../workers/)) for running segmentation algorithms, orchestrated via RabbitMQ.
- **Database**: PostgreSQL, models in [backend/db/models.py](../backend/db/models.py).
- **Messaging**: RabbitMQ for job dispatch and result collection ([shared/rabbitmq.py](../shared/rabbitmq.py)).
- **Data**: Salami dataset in [data/salami/](../data/salami/), used for training/evaluation.

## Build and Test
- **Full stack (Docker Compose):**
  ```bash
  docker-compose up -d --build
  ```
- **Stop all services:**
  ```bash
  docker-compose down
  ```
- **Backend only (local):**
  ```bash
  cd backend && uvicorn main:app --reload
  ```
- **Frontend dev server:**
  ```bash
  cd frontend/music-segmentation-ui && npm install && npm run dev
  ```
- **View logs:**
  ```bash
  docker-compose logs -f backend
  ```

## Project Conventions
- **API**: All endpoints documented in [README.md](../README.md). Use JSON for requests/responses.
- **Env config**: Copy `.env.template` to `.env` and set `DB_PASSWORD` before running.
- **Data**: Use [data/salami/](../data/salami/) for annotation files. See [get_sections_sept_2012.rb](../data/salami/get_sections_sept_2012.rb) for parsing logic.
- **Segmentation algorithms**: Add new algorithms as worker classes in [workers/segmenters/](../workers/segmenters/).

## Integration Points
- **PostgreSQL**: Connection via [backend/db/postgreSQL.py](../backend/db/postgreSQL.py).
- **RabbitMQ**: Used for backend-worker communication ([shared/rabbitmq.py](../shared/rabbitmq.py)).
- **Azure Blob Storage**: Song storage, see [backend/api/songs.py](../backend/api/songs.py).
- **Salami Dataset**: Used for evaluation/training ([data/salami/](../data/salami/)).

## Security
- **Secrets**: Never commit real credentials. Use `.env` for secrets.
- **Database**: Exposed only to Docker network.
- **Uploads**: User uploads are stored in [media/uploads/](../media/uploads/), validate file types in API.

---
Update this file if you add new services, conventions, or integration points.