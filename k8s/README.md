# Kubernetes manifests

Mirrors the services defined in [docker-compose.yml](../docker-compose.yml) for a production /
full-dataset-batch deployment: PostgreSQL, RabbitMQ, MinIO, the FastAPI backend, the Svelte
frontend, and every segmentation worker (`custom`, `msaf-foote/cnmf/scluster`, `fusion`, `llm`).

## 1. Build and push images

Unlike Compose, Kubernetes pulls pre-built images rather than building from source. Build and
push the backend/worker/frontend images to a registry your cluster can reach, then update the
`image:` field in each manifest (or re-tag to match):

```bash
docker build -t <registry>/music-segmentation-backend:latest -f backend/Dockerfile .
docker build -t <registry>/music-segmentation-worker:latest -f workers/Dockerfile .
docker build -t <registry>/music-segmentation-frontend:latest frontend/music-segmentation-ui

docker push <registry>/music-segmentation-backend:latest
docker push <registry>/music-segmentation-worker:latest
docker push <registry>/music-segmentation-frontend:latest
```

## 2. Fill in secrets

`secret.yaml` mirrors [.env.template](../.env.template) with the same placeholder values.
Replace every value with your real credentials before applying — **do not commit the filled-in
file**.

## 3. Apply

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/media-pvc.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/rabbitmq.yaml
kubectl apply -f k8s/minio.yaml
kubectl apply -f k8s/backend.yaml
kubectl apply -f k8s/worker-custom.yaml
kubectl apply -f k8s/worker-msaf.yaml
kubectl apply -f k8s/worker-fusion.yaml
kubectl apply -f k8s/worker-llm.yaml
kubectl apply -f k8s/frontend.yaml
```

Or simply `kubectl apply -f k8s/` once the secret and image references are filled in.

## Notes

- **`media-pvc.yaml` requires `ReadWriteMany`.** Backend and every worker share one media volume,
  same as Compose's `media_data`. If your cluster's default StorageClass is RWO-only (most cloud
  block storage), the PVC stays `Pending` — either install an RWX-capable StorageClass (NFS, EFS,
  Azure Files, Longhorn) or move shared media to MinIO/S3 instead.
- **GPU workers:** `worker-custom.yaml` has a commented-out `nvidia.com/gpu: 1` resource limit for
  nodes with the NVIDIA device plugin installed (e.g. a dev box with an RTX 3060).
- **Scaling:** see the root [README](../README.md#scaling-workers) for `kubectl scale` and HPA
  examples against `worker-custom`.
