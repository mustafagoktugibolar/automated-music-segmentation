# Yeni PC'ye Geçiş Rehberi

## Gereksinimler

- Docker + Docker Compose
- Git
- SALAMI audio dosyaları (aşağıya bak)

---

## 1. Kodu Al

```bash
git clone <repo-url> music-segmentation
cd music-segmentation
```

---

## 2. `.env` Dosyasını Kopyala

Git'te yok, eski makineden al:

```bash
# Eski makinede:
cat /Users/goktugibolar/dev/music-segmentation/.env
```

Yeni makinede aynı içerikle `.env` oluştur.

---

## 3. MinIO Verilerini Taşı

Audio dosyaları MinIO'da tutuluyor (`data/audio/` klasörü boş, sadece referans). MinIO volume'unu eski makineden dışa aktar:

```bash
# Eski makinede — MinIO volume'unu yedekle:
docker run --rm \
    -v music-segmentation_minio_data:/data \
    -v $(pwd):/backup \
    alpine tar czf /backup/minio_backup.tar.gz /data

# Yeni makinede — geri yükle:
docker volume create music-segmentation_minio_data
docker run --rm \
    -v music-segmentation_minio_data:/data \
    -v $(pwd):/backup \
    alpine tar xzf /backup/minio_backup.tar.gz -C /
```

---

## 4. Harmonix Dataset'ini Hazırla

`data/harmonix/` git'te yok, script ile yeniden indir:

```bash
docker compose up -d
docker exec music-segmentation-worker-custom-5 \
    python /app/scripts/label_training/prepare_harmonix_dataset.py
```

---

## 5. Label Training Verisini Hazırla

`data/label_training/*.parquet` git'te yok, script ile yeniden üret:

```bash
docker exec music-segmentation-worker-custom-5 \
    python /app/scripts/label_training/prepare_label_dataset.py

docker exec music-segmentation-worker-custom-5 \
    python /app/scripts/label_training/prepare_harmonix_dataset.py
```

---

## 6. Modeli Eğit

```bash
docker exec music-segmentation-worker-custom-5 \
    python /app/scripts/label_training/train_label_classifier.py \
    --merge-mode none \
    --extra-parquet /app/data/label_training/harmonix_segments.parquet \
    --no-multi-seed
```

Trained `.joblib` modeller git'te **yok** — eğitim sonrası `models/` altında oluşur.

---

## 7. Servisleri Başlat

```bash
docker compose up -d
```

---

## Özet: Ne Git'te Var, Ne Yok?

| Dosya/Klasör | Git'te Var mı? | Aksiyon |
|---|---|---|
| Kod, Dockerfile, scripts | Var | Clone yeter |
| `.env` | **Yok** | Eski makineden kopyala |
| `data/salami/annotations` | Kısmen var | Audio'ları kopyala |
| `data/audio/` | Boş (MinIO'da) | MinIO volume'unu taşı |
| `data/harmonix/` | **Yok** | `prepare_harmonix_dataset.py` çalıştır |
| `data/label_training/*.parquet` | **Yok** | `prepare_label_dataset.py` çalıştır |
| `models/*.joblib` | **Yok** | `train_label_classifier.py` çalıştır |
| `models/evaluation/*.json/csv` | Var | Clone yeter |
