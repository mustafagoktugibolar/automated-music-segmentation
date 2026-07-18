# Kubernetes manifests

Mirrors the services defined in [docker-compose.yml](../docker-compose.yml) for a production /
full-dataset-batch deployment: PostgreSQL, RabbitMQ, MinIO, the FastAPI backend, the Svelte
frontend, every segmentation worker (`custom`, `msaf-foote/cnmf/scluster`, `fusion`), and the
nginx gateway that fronts all of it.

## 1. Build and push images

Unlike Compose, Kubernetes pulls pre-built images rather than building from source. Build and
push the backend/worker/frontend images to a registry your cluster can reach, then update the
`image:` field in each manifest (or re-tag to match):

```bash
docker build -t <registry>/music-segmentation-backend:latest -f segmentation/api/Dockerfile .
docker build -t <registry>/music-segmentation-worker:latest -f segmentation/workers/Dockerfile .
docker build -t <registry>/music-segmentation-frontend:latest frontend/music-segmentation-ui
docker build -t <registry>/music-segmentation-gateway:latest gateway

docker push <registry>/music-segmentation-backend:latest
docker push <registry>/music-segmentation-worker:latest
docker push <registry>/music-segmentation-frontend:latest
docker push <registry>/music-segmentation-gateway:latest
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
kubectl apply -f k8s/frontend.yaml
kubectl apply -f k8s/gateway.yaml
```

Or simply `kubectl apply -f k8s/` once the secret and image references are filled in.

## Notes

- **`media-pvc.yaml` requires `ReadWriteMany`.** Backend and every worker share one media volume,
  same as Compose's `media_data`. If your cluster's default StorageClass is RWO-only (most cloud
  block storage), the PVC stays `Pending` — either install an RWX-capable StorageClass (NFS, EFS,
  Azure Files, Longhorn) or move shared media to MinIO/S3 instead.
- **GPU workers:** `worker-custom.yaml` has a commented-out `nvidia.com/gpu: 1` resource limit for
  nodes with the NVIDIA device plugin installed (e.g. a dev box with an RTX 3060).
- **`LABELING_METHOD=ml`/`ml_sequence` needs `models/` in-cluster.** Unlike Compose (which
  bind-mounts `./models:/app/models` on `worker-custom`), these manifests don't mount a `models/`
  volume and the worker image doesn't bake it in either — so `segmentation/core/labeling/ml.py` can't find
  `segment_label_clf.joblib` and every request silently falls back to the `heuristic` labeling
  method. To use ML labeling in-cluster, either add a PVC/ConfigMap volume for `models/` on
  `worker-custom.yaml` or `COPY ./models /app/models` in `segmentation/workers/Dockerfile` and rebuild the image.
- **Scaling:** see the root [README](../README.md#scaling-workers) for `kubectl scale` and HPA
  examples against `worker-custom`.
- **`gateway` is the single entry point.** `backend` and `frontend` Services are plain
  `ClusterIP` — not reachable from outside the cluster on their own. `gateway`'s Service is
  `type: LoadBalancer` and proxies `/api/*` to `backend` and everything else to `frontend`,
  same routing as `gateway/nginx.conf` (used by Compose). Get the external address with
  `kubectl get svc gateway -n music-segmentation`. Its `nginx.conf` lives in the
  `gateway-nginx-conf` ConfigMap in `gateway.yaml` — keep it in sync with `gateway/nginx.conf`
  when routes change, and add one more `location /api/<service>/` block here (and there) when a
  new service (e.g. `dataset`) joins.
