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
docker exec music-segmentation-worker-custom-1 python /app/scripts/label_training/train_label_classifier.py --merge-mode none --extra-parquet /app/data/label_training/harmonix_segments.parquet
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

PS C:\Development\automated-music-segmentation> docker exec automated-music-segmentation-worker-custom-1 python /app/scripts/label_training/train_label_classifier.py --merge-mode none --extra-parquet /app/data/label_training/harmonix_segments.parquet --no-multi-seed
>> 
Loaded 6357 segments from 436 songs  [/app/data/label_training/segments.parquet]
  + 8732 segments from 912 songs  [/app/data/label_training/harmonix_segments.parquet]
Combined: 15089 segments from 1348 songs.

Label distribution:
label
Chorus          4153
Verse           3246
Instrumental    1692
Other           1351
Intro           1242
Silence          945
Bridge           930
Pre-Chorus       834
Outro            696

Grouping split by raw_track_id  (1348 unique tracks, 1348 unique song_ids)

Feature matrix: (15089, 87)  |  Classes: ['Bridge', 'Chorus', 'Instrumental', 'Intro', 'Other', 'Outro', 'Pre-Chorus', 'Silence', 'Verse']
Feature set: full  |  Selected features: 87

############################################################
  PRIMARY RUN  (seed=42) — with full diagnostics
############################################################

Split integrity check:
  Split integrity (raw_track_id): OK
  Split integrity (song_id): OK

Split diagnostics:

  ── TRAIN split ──
     Songs: 808  |  Segments: 9158  |  Avg segs/song: 11.3
     Label distribution:
       Chorus            2486  ( 27.1%)
       Verse             1998  ( 21.8%)
       Instrumental      1085  ( 11.8%)
       Other              822  (  9.0%)
       Intro              740  (  8.1%)
       Silence            584  (  6.4%)
       Bridge             562  (  6.1%)
       Pre-Chorus         471  (  5.1%)
       Outro              410  (  4.5%)

  ── VAL split ──
     Songs: 270  |  Segments: 2944  |  Avg segs/song: 10.9
     Label distribution:
       Chorus             834  ( 28.3%)
       Verse              622  ( 21.1%)
       Instrumental       295  ( 10.0%)
       Intro              256  (  8.7%)
       Other              251  (  8.5%)
       Bridge             194  (  6.6%)
       Silence            180  (  6.1%)
       Pre-Chorus         163  (  5.5%)
       Outro              149  (  5.1%)

  ── TEST split ──
     Songs: 270  |  Segments: 2987  |  Avg segs/song: 11.1
     Label distribution:
       Chorus             833  ( 27.9%)
       Verse              626  ( 21.0%)
       Instrumental       312  ( 10.4%)
       Other              278  (  9.3%)
       Intro              246  (  8.2%)
       Pre-Chorus         200  (  6.7%)
       Silence            181  (  6.1%)
       Bridge             174  (  5.8%)
       Outro              137  (  4.6%)

Training lightgbm (early stopping on validation set) …
[warn] lightgbm not installed — falling back to HistGradientBoosting
  External validation set is not used for fitting.
  Early stopping uses an internal 10% split of the training set.
  Early stopping at iteration 123.
Done in 9.5s.

── Train (808 songs) ──
  Accuracy   : 0.9617
  Macro-F1   : 0.9650
  Weighted-F1: 0.9617
              precision    recall  f1-score   support

      Bridge       0.93      0.96      0.94       562
      Chorus       0.97      0.94      0.96      2486
Instrumental       0.93      0.97      0.95      1085
       Intro       0.97      0.99      0.98       740
       Other       0.98      0.96      0.97       822
       Outro       0.96      0.99      0.97       410
  Pre-Chorus       0.96      0.96      0.96       471
     Silence       1.00      1.00      1.00       584
       Verse       0.96      0.96      0.96      1998

    accuracy                           0.96      9158
   macro avg       0.96      0.97      0.97      9158
weighted avg       0.96      0.96      0.96      9158


── Validation (270 songs) ──
  Accuracy   : 0.6834
  Macro-F1   : 0.6759
  Weighted-F1: 0.6822
              precision    recall  f1-score   support

      Bridge       0.49      0.56      0.52       194
      Chorus       0.73      0.66      0.69       834
Instrumental       0.53      0.64      0.58       295
       Intro       0.84      0.86      0.85       256
       Other       0.65      0.46      0.54       251
       Outro       0.66      0.79      0.72       149
  Pre-Chorus       0.56      0.45      0.50       163
     Silence       0.98      0.96      0.97       180
       Verse       0.68      0.75      0.71       622

    accuracy                           0.68      2944
   macro avg       0.68      0.68      0.68      2944
weighted avg       0.69      0.68      0.68      2944


── Test (270 songs) ──
  Accuracy   : 0.6686
  Macro-F1   : 0.6603
  Weighted-F1: 0.6648
              precision    recall  f1-score   support

      Bridge       0.42      0.48      0.45       174
      Chorus       0.70      0.64      0.67       833
Instrumental       0.56      0.67      0.61       312
       Intro       0.79      0.91      0.85       246
       Other       0.72      0.44      0.54       278
       Outro       0.60      0.77      0.67       137
  Pre-Chorus       0.57      0.42      0.48       200
     Silence       0.98      0.97      0.98       181
       Verse       0.66      0.73      0.70       626

    accuracy                           0.67      2987
   macro avg       0.67      0.67      0.66      2987
weighted avg       0.67      0.67      0.66      2987


Model saved -> /app/models/segment_label_clf.joblib
Mode-specific model saved -> /app/models/segment_label_clf_none.joblib

Top misclassifications (test):
  Chorus           → Verse            : 129
  Verse            → Chorus           : 71
  Other            → Chorus           : 51
  Chorus           → Instrumental     : 47
  Pre-Chorus       → Chorus           : 43
  Verse            → Instrumental     : 38
  Other            → Instrumental     : 35
  Chorus           → Bridge           : 33
  Chorus           → Outro            : 33
  Pre-Chorus       → Verse            : 32

Artifacts saved -> /app/models/evaluation/

============================================================
  SUMMARY
============================================================
  Merge mode  : none
  Classes     : ['Bridge', 'Chorus', 'Instrumental', 'Intro', 'Other', 'Outro', 'Pre-Chorus', 'Silence', 'Verse']
  Feature set : full
  Reg preset  : default
  Features    : 87
  Primary seed: 42

  Split        Accuracy   Macro-F1  Weighted-F1
  --------------------------------------------
  Train          0.9617     0.9650       0.9617
  Val            0.6834     0.6759       0.6822
  Test           0.6686     0.6603       0.6648
  Train-Test Macro-F1 gap: 0.3047
  Groups train/val/test : 808/270/270
PS C:\Development\automated-music-segmentation> 