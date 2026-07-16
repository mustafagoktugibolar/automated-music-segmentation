# Music Segmentation UI

Svelte 5 + Vite frontend for the [music-segmentation](../../README.md) project. It talks to the
FastAPI backend to run segmentation algorithms, browse the SALAMI-backed dataset, and inspect
evaluation results.

## Pages

- **Dataset Manager** ([src/components/DatasetManager.svelte](src/components/DatasetManager.svelte)) — browse songs in storage and dispatch segmentation runs.
- **Evaluation Dashboard** ([src/components/EvaluationDashboard.svelte](src/components/EvaluationDashboard.svelte)) — inspect per-track, per-algorithm boundary metrics.
- **Batch Eval Dashboard** ([src/components/BatchEvalDashboard.svelte](src/components/BatchEvalDashboard.svelte)) — run and compare full-dataset batch evaluations across algorithms.

## Getting Started

```bash
npm install
npm run dev
```

The app expects the backend at `http://localhost:8000` by default. To point at a different
backend, set `VITE_BACKEND_URL` (see [src/lib/api.js](src/lib/api.js)), e.g.:

```bash
VITE_BACKEND_URL=http://localhost:8000 npm run dev
```

## Build

```bash
npm run build
npm run preview
```

See the root [docker-compose.yml](../../docker-compose.yml) for how this is built and served
(`Dockerfile` + nginx) alongside the rest of the stack.
