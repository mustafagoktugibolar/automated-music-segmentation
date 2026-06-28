# ML Segment Labeling Pipeline

Music segmentation iki aşamadan oluşur: **boundary detection** (segmentlerin nerede başlayıp bittiği) ve **semantic labeling** (her segmentin ne olduğu: Verse, Chorus, Bridge…). Bu döküman ikinci aşamayı — ML tabanlı labeling pipeline'ını — kapsar.

---

## Genel Mimari

```
Audio (MinIO)  ──►  Feature Extraction  ──►  GBDT Classifier  ──►  Label
SALAMI GT      ──►  (train time only)
```

- **Training**: SALAMI ground-truth segment sınırları + label'ları kullanılır.
- **Inference**: Segmentasyon pipeline'ının belirlediği sınırlar kullanılır; bu nedenle boundary hataları label accuracy'yi de etkiler.
- **Fallback**: Model yüklenemezse veya `labeling_method=heuristic` seçilmişse kural tabanlı heuristic devreye girer.

---

## Adım 1 — Veri Hazırlama (`prepare_label_dataset.py`)

```bash
python scripts/label_training/prepare_label_dataset.py [--max-songs N] [--workers 8]
```

**Ne yapar:**

1. `data/salami/annotations/` altındaki annotation dizinlerini tarar → integer ID'leri listeler
2. MinIO'daki mevcut şarkı ID'leriyle kesişim alır (her iki kaynakta da olmayanlar atlanır)
3. Her şarkı için:
   - MinIO'dan audio bytes indirir (`salami/songs/{id}.mp3`)
   - ffmpeg ile decode eder → numpy array (22050 Hz, mono)
   - SALAMI annotation'ını parse eder → `[{start, end, label}, …]`
   - Her segment için 60-boyutlu feature vektörü çıkarır
4. `data/label_training/segments.parquet` dosyasına yazar

**Çıktı formatı (parquet):**

| Sütun | Tip | Açıklama |
|-------|-----|----------|
| `song_id` | str | `salami_{id}_ann{n}` |
| `dataset` | str | `"salami"` |
| `segment_idx` | int | Şarkı içindeki sıra |
| `start`, `end` | float | Saniye cinsinden zaman |
| `label` | str | Canonical label (Verse, Chorus, …) |
| `chroma_mean_0..11` | float | Feature vektörü (60 + 13 = 73 sütun) |
| … | … | … |

**Label normalizasyonu:** SALAMI annotation'larında `"verse"`, `"Verse"`, `"V"` gibi varyasyonlar `normalize_label()` ile canonical forma çevrilir. Vocabulary dışı label'lar `"Other"` olur.

---

## Adım 2 — Feature Extraction (73 feature)

Her segment için feature vektörü `workers/infrastructure/audio/features.py::build_segment_descriptors_from_audio` + `workers/core/labeling/features.py::build_segment_label_vectors` tarafından üretilir.

### Grup A — Akustik Features (54 dim)

Tüm features segment içindeki frame'ler üzerinden hesaplanır (`hop_length=512`, ~11.6ms/frame).

| İndeks | Feature | Hesaplama | Neden |
|--------|---------|-----------|-------|
| 0–11 | `chroma_mean_0..11` | `mean(chroma_cens, axis=time)` | Harmonik içerik — Chorus/Verse ayrımı |
| 12–23 | `chroma_std_0..11` | `std(chroma_cens, axis=time)` | Harmonik stabilite |
| 24–36 | `mfcc_mean_0..12` | `mean(MFCC(n=13), axis=time)` | Timbral karakteristik (vokal vs enstrümantal) |
| 37–49 | `mfcc_std_0..12` | `std(MFCC(n=13), axis=time)` | Timbral değişkenlik |
| 50 | `rms_mean` | `mean(RMS)` | Ortalama enerji — Intro/Silence tespiti |
| 51 | `rms_std` | `std(RMS)` | Enerji varyasyonu |
| 52 | `onset_density` | `mean(onset > 75. percentile)` | Ritmik yoğunluk |
| 53 | `norm_duration` | `segment_dur / song_dur` | Göreli süre |

> **Chroma CENS** (Chroma Energy Normalized Statistics): raw chroma'dan daha robust, tempo ve dinamik varyasyonlara karşı daha dayanıklı (FMP Section 7.2.3).

### Grup B — Ritim + Spektral Features (6 dim)

| İndeks | Feature | Hesaplama | Neden |
|--------|---------|-----------|-------|
| 54 | `tempo_norm` | `librosa.feature.tempo(onset_env) / 200` | Global tempo — tüm şarkı onset envelope'undan |
| 55 | `beat_density_norm` | `beat_count_in_segment / duration / 5` | Segment ritmik yoğunluğu |
| 56 | `beat_regularity` | `1 - CV(inter_beat_intervals)` | 1=metronom, 0=kaotik |
| 57 | `spectral_centroid_mean_norm` | `mean(centroid) / (sr/2)` | "Parlaklık" — vokal yüksek, bas düşük |
| 58 | `spectral_centroid_std_norm` | `std(centroid) / (sr/2)` | Parlaklık değişimi |
| 59 | `zcr_mean` | `mean(ZCR)` | Sıfır geçiş oranı — vokalli vs sessiz ayrımı |

> **Beat detection**: `librosa.beat.plp` (Predominant Local Pulse) kullanılır. `beat_track` (Viterbi DP) uzun şarkılarda (>3 dk) segfault yaptığı için tercih edilmez; PLP tempo-aware ve kararlıdır.

### Grup C — Bağlamsal Features (13 dim)

Saf akustik bilgi context-free olduğu için segment'in şarkı içindeki konumunu da encode ederiz.

| İndeks | Feature | Açıklama |
|--------|---------|----------|
| 60 | `normalized_start` | `start / song_dur` |
| 61 | `normalized_end` | `end / song_dur` |
| 62 | `position_center` | `(start + end) / 2 / song_dur` |
| 63 | `index_norm` | `segment_index / (n_segments - 1)` |
| 64 | `is_first` | İlk segment mi? (Intro kuvvetli sinyal) |
| 65 | `is_last` | Son segment mi? (Outro kuvvetli sinyal) |
| 66 | `n_segments` | Şarkıdaki toplam segment sayısı |
| 67 | `duration_s` | Segment süresi (saniye) |
| 68 | `log_duration` | `log1p(duration_s)` |
| 69 | `rms_energy_rank` | Segment'in RMS sıralaması (0=en sessiz, 1=en yüksek) |
| 70 | `is_max_energy` | Şarkının en yüksek enerjili segmenti mi? |
| 71 | `repetition_count` | Aynı yapısal label kaç kez tekrar ediyor |
| 72 | `is_repeated` | `repetition_count >= 2` |

> Bağlamsal features olmadan model, şarkının başında mı sonunda mı olduğunu bilemez. Intro genellikle `is_first=1` ve `rms_energy_rank` düşük olur; Outro `is_last=1`.

---

## Adım 3 — Model Eğitimi (`train_label_classifier.py`)

```bash
python scripts/label_training/train_label_classifier.py [--backend sklearn|xgboost]
```

### Split Stratejisi

**Song-level 60/20/20 split** — `GroupShuffleSplit` ile:

- Aynı şarkının segmentleri asla train ve test'e bölünmez (data leakage önlenir)
- %60 Train (~263 şarkı) → model öğrenir
- %20 Validation (~88 şarkı) → early stopping için
- %20 Test (~88 şarkı) → final rapor için, eğitim boyunca hiç bakılmaz

```
Tüm şarkılar (439)
│
├─ %80 Train+Val  ──► GroupShuffleSplit(val_size=0.25) ──► 60% Train + 20% Val
│
└─ %20 Test  (sadece sonunda bir kez değerlendirilir)
```

### Model

`sklearn.ensemble.HistGradientBoostingClassifier` (GBDT — Gradient Boosted Decision Trees):

| Parametre | Değer | Gerekçe |
|-----------|-------|---------|
| `max_iter` | 1000 | Early stopping keser |
| `learning_rate` | 0.05 | Düşük LR + early stopping = daha iyi genelleme |
| `min_samples_leaf` | 20 | Küçük yaprak = overfitting; 20 minimum |
| `l2_regularization` | 0.1 | Ağırlıkları küçük tutar |
| `early_stopping` | True | Validation loss artmaya başlarsa dur |
| `n_iter_no_change` | 30 | 30 iterasyon iyileşme olmasa dur |

**Sınıf dengesizliği:** `compute_sample_weight("balanced")` ile nadir sınıflar (Pre-Chorus, Bridge) daha yüksek ağırlık alır.

**Nadir label birleştirme:** `--min-class-count 50` (default) — train setinde 50'den az segmenti olan label'lar `"Other"`'a merge edilir. Şu an Pre-Chorus bu kapsamda.

### Çıktı

| Dosya | İçerik |
|-------|--------|
| `models/segment_label_clf.joblib` | Model bundle: `{clf, label_encoder, feature_names, classes, …}` |
| `models/segment_label_clf.meta.json` | Metrik raporu: accuracy, macro-F1, per-class F1, confusion matrix |

---

## Adım 4 — Değerlendirme (`eval_label_classifier.py`)

```bash
python scripts/label_training/eval_label_classifier.py [--mode clean pipeline] [--max-songs 50]
```

### İki Değerlendirme Modu

**Clean mode** (default) — GT sınırlarla:
- SALAMI ground-truth segment sınırları kullanılır
- Sadece labeling kalitesi ölçülür, boundary hatası dahil değil
- "Classifier teorik olarak ne kadar iyi?" sorusunu cevaplar

**Pipeline mode** — gerçek segmentasyon çıktısıyla:
- Önce `custom` segmenter çalıştırılır → tahmin edilen sınırlar
- Her tahmin segmenti için en çok örtüşen GT segmentin label'ı alınır
- Hem ML hem heuristic sonuçları yan yana raporlanır
- "Uçtan uca sistem ne kadar iyi?" sorusunu cevaplar

### Metrikler

| Metrik | Açıklama |
|--------|----------|
| Accuracy | Doğru tahmin oranı |
| Macro-F1 | Her sınıfın F1'inin ortalaması (imbalance'a karşı adil) |
| Per-class F1 | Verse, Chorus, … için ayrı ayrı precision/recall/F1 |
| Confusion matrix | Hangi label'lar birbiriyle karışıyor |

---

## Label Vocabulary

| Label | Tipik akustik özellikler |
|-------|-------------------------|
| Intro | Düşük/orta enerji, `is_first=1`, kısa |
| Verse | Vokal, orta enerji, yüksek tekrar sayısı |
| Pre-Chorus | Verse'e benzer ama daha yüksek enerji, kısa |
| Chorus | En yüksek enerji, `is_max_energy` yüksek |
| Post-Chorus | Chorus'a benzer, chorus'tan sonra |
| Bridge | Düşük tekrar, genellikle harmonik farklılık |
| Instrumental | Düşük ZCR (vokal yok), orta-yüksek enerji |
| Outro | Düşen enerji, `is_last=1` |
| Silence | Çok düşük RMS |
| Other | Yukarıdakilere uymayan her şey |

---

## Inference (Canlı Kullanım)

Segmentasyon tamamlanınca `workers/core/labeling/ml.py::predict_labels` çağrılır:

```
segments  ──►  build_segment_label_vectors()  ──►  clf.predict()  ──►  labels
               (73-dim feature)                     (joblib bundle)
```

`labeling_method` seçimi frontend'den veya API'den gelir:
- `"heuristic"` → kural tabanlı, model gerektirmez, her zaman çalışır
- `"ml"` → model bundle yüklenir; yüklenemezse heuristic'e düşer

Model bundle `feature_names` içerir — eğitim ve inference'taki feature sırası aynı olmak zorundadır. Acoustic dim değişirse model yeniden eğitilmelidir.

---

## Bilinen Sınırlamalar

1. **Val/Test gap** — v1'de val %95, test %68. Muhtemel nedenler: SALAMI annotation tutarsızlığı, az training data.
2. **Bridge ve Pre-Chorus** — akustik olarak komşu sınıflarla çok benzer; daha fazla data ve/veya daha ayırt edici features gerekiyor.
3. **`beat_track` kullanılamıyor** — `numpy<1.24` + librosa 0.11.0 kombinasyonunda uzun şarkılarda segfault. PLP kullanılıyor.
4. **SALAMI-only** — Harmonix Set audio içermediği için kullanılmıyor; eklenince model yeniden eğitilmeli.
