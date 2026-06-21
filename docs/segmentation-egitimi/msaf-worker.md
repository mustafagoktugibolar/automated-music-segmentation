# MSAF Adapter: msaf_worker.py

MSAF farklı müzik yapı algoritmalarını ortak arayüzle çalıştırır. Worker foote, cnmf ve scluster kabul eder. Algoritma matematiğini yeniden yazmaz; MSAF sonucunu proje şemasına uyarlar. Bu nedenle adapter'dır.

## Input–output sequence

~~~mermaid
sequenceDiagram
    participant Q as RabbitMQ / BaseWorker
    participant W as MSAFWorker
    participant M as msaf.process
    participant U as segmentation_utils
    participant L as labeling
    Q->>W: INPUT task {task_id, file_path, params}
    W->>W: Algoritma ve dosya doğrulama
    W->>M: boundaries_id + optional params
    M-->>W: est_times + est_labels
    W->>U: normalize_boundaries
    U-->>W: [0, internal boundaries, duration]
    W->>U: boundaries_to_segments
    U-->>W: segment interval'ları
    W->>L: apply_two_layer_labels
    L-->>W: structural + semantic labels
    W-->>Q: OUTPUT normalized result
~~~

MSAF output'u doğrudan yayınlanmaz. Önce boundary, label ve segment sözleşmelerine uyarlanır; adapter olmasının anlamı budur.

## MSAFWorker.__init__()

1. MESSAGE_CODE ile routing key'i alır.
2. MSAF_ALGORITHM ile algoritmayı seçer.
3. Features/estimations klasörlerini oluşturur.
4. BaseWorker'a service, queue, routing bilgisi verir.

Aynı image farklı environment değerleriyle üç worker olabilir.

## process_task() adım adım

### Task ve doğrulama

task_id ve params.msaf okunur. _resolve_file_path() üst sınıftan gelir. Algoritma whitelist dışında ise ValueError, dosya yoksa FileNotFoundError oluşur. Hatalı input başarılı gibi gösterilmez.

### Süre ve diagnostics

get_audio_duration() süreyi ölçer. Diagnostics; algoritma, süre, warning, parametre ve aşama sürelerini saklar.

### MSAF çağrısı

Temel argüman boundaries_id'dir. Request'te labeling_id ve hier varsa eklenir. msaf.process() est_times ve est_labels döndürür; ham sayılar diagnostics'e yazılır.

### Boundary normalization

normalize_boundaries():

- Geçersiz zamanları temizler.
- Duplicate/yakın zamanları düzenler.
- Track dışını engeller.
- include_edges=True ile 0 ve duration'ı sağlar.

Örnek:

    boundaries = [0, 18, 45, 70]
    segments   = [0–18], [18–45], [45–70]

0 ve 70 zorunlu kenardır. Algorithm fusion için keşfedilmiş internal oylar 18 ve 45'tir.

### Label normalization

segment_count = len(boundaries) - 1.

_normalize_msaf_labels():

- Fazla label'ı keser.
- Eksikleri A/B/C... ile doldurup warning yazar.
- None, boş, nan, unknown değerlerini deterministik harfle değiştirir.

Boundary-label uzunluk uyuşmazlığı böylece segment oluşturmayı bozmaz.

### Segment ve ortak label

boundaries_to_segments() ardışık zamanlardan interval üretir. Adapter boundary confidence'ını 1.0 yazar; bu bilimsel yüzde yüz doğruluk değil, MSAF'ın calibrated confidence sağlamadığı durumda normalize edilen değerdir.

Ham MSAF label raw_msaf_label alanında korunur. apply_two_layer_labels() ortak structural/semantic label üretir. Ham cluster kimliği doğrudan Chorus yapılmaz; structural benzerlik ve semantic rol farklı iddialardır.

normalize_algorithm_result() task/status/worker/algorithm, duration, boundaries, segments ve diagnostics'i ortak şemaya taşır. Frontend, evaluation ve fusion algoritmaya özel branching yapmaz.

## Hata politikası

process_task() exception'ı loglayıp yeniden fırlatır. Failed result üretip publish etme BaseWorker lifecycle'ının görevidir. Adapter domain dönüşümüne odaklanır.

## Metot özeti

| Method | Girdi | Çıktı | Neden |
|---|---|---|---|
| __init__ | Environment | Worker ayarı | Tek image, farklı algoritma |
| process_task | Task dict | Normalized result | MSAF-proje adapter'ı |
| _normalize_msaf_labels | Labels, count | Temiz labels | Uzunluk/boş değer güvenliği |

Savunma cümlesi:

> MSAF worker algoritmanın matematiğini sahiplenmez; dış kütüphane sonucunun süre, boundary, segment, label ve diagnostics sözleşmelerimize güvenli biçimde uymasını sağlar.
