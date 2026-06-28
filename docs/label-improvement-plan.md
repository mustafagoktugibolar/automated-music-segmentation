# Label Classifier İyileştirme Planı

Stage 1 uygulandı. Kalan aşamalar.

---

## Stage 1 — Validation leakage fix ✅ (tamamlandı)

`train_label_classifier.py::train_model` sklearn dalında `X_val`'ı `clf.fit`'e dahil
etmiyordu sorunu giderildi. Artık early stopping sadece `X_train`'in %10'luk iç
dilimine bakıyor; `X_val` yalnızca dışarıdan raporlanan metrik olarak kalıyor.

Beklenen sonuç: val macro-F1 ≈ test macro-F1 (~0.66-0.72), eskiden val ~0.95 sahte
görünüyordu.

---

## Stage 2 — Target-derived feature leakage fix

**Problem:** `workers/core/labeling/features.py` içinde `_CONTEXT_NAMES` listesinde
yer alan `repetition_count` ve `is_repeated` feature'ları `structural_label`/`label`
alanlarından türetiliyor. Eğitim sırasında `label` ground-truth target'tır — bu bir
target leakage'dır.

**Düzeltme (`workers/core/labeling/features.py`):**
- `_CONTEXT_NAMES`'den `"repetition_count"` ve `"is_repeated"` satırlarını sil.
- `Counter` import'unu ve `label_counts`/`struct_labels` kullanımını kaldır.
- `ctx` matrisini ve döngüdeki `rep` hesabını güncelle (11 ve 12. indeks yok artık).
- `_CONTEXT_DIM` → 11, `_TOTAL_DIM` → 71.
- Docstring'deki "73 total" → 71'e güncelle; `feature_names()` docstring'indeki
  "67" yorumunu da düzelt (hep 60+13=73 olarak yanlıştı, şimdi 60+11=71).
- `feature_names()` dönüş listesindeki son yorum satırını da güncelle.

**Dikkat:** `workers/infrastructure/audio/features.py` docstring'lerindeki `(N, 54)`
ifadeleri de aslında yanlış (gerçek değer 60). Bu Stage 2'de de düzeltilebilir.

---

## Stage 3 — Raw track grouping

**Problem:** `prepare_label_dataset.py` içindeki `_load_salami_entries()` fonksiyonu
`raw_id` alanını oluşturuyor ama parquet'e yazmıyor. Aynı ham ses dosyasının farklı
annotator'larla farklı `song_id`'lere düşmesi ve farklı split'lere girmesi riski var.

**Düzeltme (`prepare_label_dataset.py`):**

`_load_salami_entries()` döndürdüğü her dict'e şunları ekle:
```python
{
    "song_id":      f"salami_{song_id}_ann{ann}",
    "raw_id":       song_id,          # ham SALAMI integer ID
    "raw_track_id": f"salami_{song_id}",   # annotator'dan bağımsız track kimliği
    "annotator_id": ann,
    "dataset":      "salami",
    "segments":     segs,
}
```

`_extract_rows()` içinde her satıra `raw_track_id` ve `annotator_id` sütunlarını da ekle:
```python
row = {
    "song_id":      song_id,
    "raw_track_id": entry["raw_track_id"],
    "annotator_id": entry["annotator_id"],
    "dataset":      dataset,
    ...
}
```

**Düzeltme (`train_label_classifier.py`):**

`META_COLS` setine `"raw_track_id"` ve `"annotator_id"` ekle:
```python
META_COLS = {"song_id", "raw_track_id", "annotator_id", "dataset",
             "segment_idx", "start", "end", "label"}
```

`make_grouped_split` çağrısı öncesinde group sütununu seç:
```python
if "raw_track_id" in df.columns:
    group_col = "raw_track_id"
    print("Grouping split by raw_track_id")
else:
    group_col = "song_id"
    print("Grouping split by song_id")
groups = df[group_col].values
```

`check_split_integrity` ve `print_split_diagnostics`'i bu group_col ile çağır.

`split_diagnostics.json`'a ekle:
```json
{
  "group_col": "raw_track_id",
  "unique_song_ids": 439,
  "unique_raw_track_ids": 439,
  "unique_annotators": [1, 2]
}
```

---

## Stage 4 — Dataset rebuild + clean baseline

Stage 2 ve 3 bittikten sonra:

```bash
C=music-segmentation-worker-custom-5
docker exec $C python /app/scripts/label_training/prepare_label_dataset.py
docker exec $C python /app/scripts/label_training/train_label_classifier.py --merge-mode none
docker exec $C python /app/scripts/label_training/train_label_classifier.py --merge-mode transition
docker exec $C python /app/scripts/label_training/train_label_classifier.py --merge-mode other
```

Her merge mode için raporla:
- class count, feature count
- Train / Val / Test Macro-F1
- Val-Test gap
- multi-seed mean ± std
- per-class Test F1
- top misclassifications

Kaydet: `models/evaluation/clean_baseline_comparison.csv`

---

## Stage 5 — Acoustic repetition features (non-leaky)

Stage 4 clean baseline hazır olduktan sonra, `label`'a **bakmadan** acoustic tekrar
feature'ları ekle.

**`workers/core/labeling/features.py::build_segment_label_vectors` içine:**

Segment'lerin `acoustic` matrisinden (`chroma_mean_0..11` = sütun 0-11,
`mfcc_mean_0..12` = sütun 24-36) intra-song kosinüs benzerlik matrisi hesapla.
Her segment i için:

| Feature | Hesaplama |
|---------|-----------|
| `max_chroma_similarity` | i'nin diğer segmentlere maks kosinüs benzerliği (chroma) |
| `mean_top3_chroma_similarity` | en benzer 3 segmente ortalama |
| `max_mfcc_similarity` | aynısı MFCC ile |
| `mean_top3_mfcc_similarity` | aynısı MFCC ile |
| `similar_count_chroma_080` | chroma benzerliği > 0.80 olan segment sayısı |
| `similar_count_mfcc_080` | MFCC benzerliği > 0.80 olan segment sayısı |
| `nearest_similar_dist_norm` | en benzer segmentin normalize pozisyon farkı |
| `is_repeated_acoustic` | `similar_count_chroma_080 >= 2` |

Kurallar:
- `label`, `structural_label` veya tahmin edilen label kullanılmaz.
- Sadece `acoustic` matrisinden hesaplanır (parquet'te mevcut sütunlar).
- `_CONTEXT_NAMES`'e ekle, `_CONTEXT_DIM`/`_TOTAL_DIM` güncelle.
- `feature_names()` otomatik güncellenecek.
- Inference'da `build_segment_label_vectors` aynı şekilde çalıştığı için
  training/inference paritesi otomatik korunur.

---

## Stage 6 — Local contrast features

Verse/Chorus/Instrumental ayrımı için göreli enerji ve geçiş sinyalleri:

| Feature | Hesaplama |
|---------|-----------|
| `rms_vs_song_mean` | `rms_mean[i] - mean(rms_mean over all segs)` |
| `rms_vs_prev` | `rms_mean[i] - rms_mean[i-1]` (ilk seg için 0) |
| `rms_vs_next` | `rms_mean[i] - rms_mean[i+1]` (son seg için 0) |
| `onset_vs_song_mean` | aynısı onset_density için |
| `onset_vs_prev` | aynısı |
| `onset_vs_next` | aynısı |
| `centroid_vs_song_mean` | aynısı spectral_centroid_mean_norm için |
| `duration_vs_song_mean` | `duration_s[i] - mean(duration_s)` |

Yine `label` kullanılmaz, sadece acoustic matris sütunları.
`_CONTEXT_NAMES`, `_CONTEXT_DIM`, `_TOTAL_DIM`, `feature_names()` güncelle.

---

## Stage 7 — Sequence smoothing (opsiyonel)

Stage 5-6 sonrası hâlâ Chorus/Verse karışması varsa:

`scripts/label_training/sequence_smooth.py` (yeni dosya).

Eğitim seti label dizilerinden geçiş matrisi öğren.
Inference'da GBDT `predict_proba`'sını emission olarak kullan, Viterbi decode et.

Ek kural tabanlı smoothing:
- Izole tek-segment label flip'lerini düzelt (i-1 == i+1 ≠ i ise merge).
- `Intro` yalnızca şarkının ilk %20'sinde çıkabilir.
- `Outro` yalnızca son %20'sinde.
- `transition` modunda `Transition` segment Chorus'tan önce ya da sonra beklenebilir,
  rastgele yerde değil.

Entegrasyon: `workers/core/labeling/ml.py::predict_semantic_labels`'e opsiyonel
post-processing adımı olarak ekle (bundle'a `transition_matrix` kaydet).

CLI flag: `--postprocess sequence_smoothing`

Karşılaştır:
```bash
docker exec $C python /app/scripts/label_training/train_label_classifier.py --merge-mode other
docker exec $C python /app/scripts/label_training/sequence_smooth.py --merge-mode other
```

---

## Dosya özetleri

| Dosya | Stage | Değişiklik |
|-------|-------|-----------|
| `scripts/label_training/train_label_classifier.py` | 1 ✅ | leakage fix |
| `workers/core/labeling/features.py` | 2 | repetition_count/is_repeated kaldır |
| `prepare_label_dataset.py` | 3 | raw_track_id + annotator_id ekle |
| `train_label_classifier.py` | 3 | raw_track_id ile grouping |
| `workers/core/labeling/features.py` | 5,6 | acoustic repetition + contrast features |
| `scripts/label_training/sequence_smooth.py` | 7 | yeni dosya, HMM Viterbi |
| `workers/core/labeling/ml.py` | 7 | smoothing entegrasyonu |
