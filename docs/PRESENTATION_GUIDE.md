# Music Segmentation Project - Presentation Guide

Bu dokuman, projenin sunumunu bastan sona yonetmek icin hazirlanmis ayrintili bir konusma ve teknik referans rehberidir. Amaci yalnizca sistemin ne yaptigini soylemek degil; neden bu mimarinin secildigini, bir istegin sistemde nasil ilerledigini, her segmentasyon yonteminin hangi probleme odaklandigini, iki farkli fusion katmaninin nasil calistigini ve sonuclarin nasil degerlendirildigini savunulabilir bicimde anlatmaktir.

Dokumanin en kritik mesaji sudur:

> Bu proje yalnizca bir audio file'i segment'lere ayiran tek bir algorithm degildir. Birden fazla segmentation approach'unu distributed worker architecture ile calistiran, result'lari common schema'ya donusturen, boundary prediction'larini iki farkli seviyede birlestiren ve output'lari human annotation'larla multi-tolerance evaluation'a sokan end-to-end bir music structure analysis sistemidir.

---

## 1. Sunumun Ana Hikayesi

Sunum boyunca teknik ayrintilara gecmeden once dinleyicinin zihninde su problem-hikaye-cozum zincirini kur:

1. Bir music track zaman icinde intro, verse, chorus, bridge ve outro gibi section'lara ayrilir.
2. Bilgisayar acisindan asil problem, bu section'larin degistigi time point'leri yani `boundary` degerlerini bulmaktir.
3. Tek bir acoustic feature her music genre'da reliable degildir. Harmonic repetition, energy change, onset density, rhythm ve timbre farkli track'lerde farkli derecede bilgi tasir.
4. Tek bir segmentasyon algoritmasi da her parca turunde ayni guvenilirlikte degildir. Bu nedenle custom Librosa pipeline'i ve uc MSAF baseline'i birlikte kullanilir.
5. Sistem iki ayri seviyede fusion uygular:
   - `feature-level fusion`: tek bir custom algorithm icinde farkli audio feature'lardan gelen boundary candidate'larini birlestirir.
   - `algorithm-level fusion`: tamamlanmis `custom_librosa`, `foote`, `cnmf` ve `scluster` algorithm result'larini weighted voting ile birlestirir.
6. Sonuclar ortak bir semaya normalize edilir ve SALAMI insan anotasyonlariyla `0.5 saniye` ve `3.0 saniye` toleranslarda degerlendirilir.

Sunumda bu zinciri kaybetme. Her teknik ayrintiyi su sorulardan birine bagla:

- Daha guvenilir boundary nasil uretiliyor?
- Farkli worker sonuclari nasil ayni dilde konusuyor?
- Fusion hangi false positive'leri veya missed boundary'leri azaltmaya calisiyor?
- Sonucun iyi oldugunu nasil olcuyoruz?

---

## 2. Onerilen Sunum Akisi

Yaklasik 25-35 dakikalik teknik sunum icin onerilen akistir. Daha kisa bir sunumda method ayrintilarini azaltabilirsin; fusion ve evaluation bolumlerini azaltma.

### Slayt 1 - Proje ve problem tanimi

Anlat:

- Projenin adi ve amaci.
- Giris olarak bir audio dosyasi veya storage'daki bir parca alindigi.
- Output olarak segment'ler, boundary timestamp'leri, structural label'lar, optional semantic label'lar ve diagnostics uretildigi.
- Temel hedefin muzikteki yapisal gecis noktalarini otomatik bulmak oldugu.

Ornek konusma:

> Projemiz, bir music track'in timeline uzerindeki structural section'larini otomatik olarak cikaran distributed music segmentation sistemidir. Sistemin primary prediction'i section name'den once boundary, yani bir section'dan digerine transition'in gerceklestigi timestamp'tir. Bu boundary'lerden segment interval'lari uretiyor, recurring section'lari A, B, C gibi structural label'larla grupluyor ve yeterli evidence varsa Intro, Verse veya Chorus gibi semantic annotation'lar ekliyoruz.

### Slayt 2 - Neden zor bir problem?

Anlat:

- Music section'larinin tek bir signal ile kesin belirlenemedigini.
- Bazi gecislerde armoninin, bazilarinda enerjinin, bazilarinda enstrumantasyonun degistigini.
- Insan anotasyonlarinda bile tam timestamp farklari olabildigini.
- Bu nedenle multi-feature, multi-algorithm ve tolerance-based evaluation gerektigini.

Vurgula:

> Boundary detection bir classification problemi gibi tek bir label secmek degildir. Timeline uzerinde hem dogru sayida hem de dogru konumda boundary bulmak gerekir. Fazla boundary over-segmentation, az boundary under-segmentation yaratir.

### Slayt 3 - Ust seviye mimari

Asagidaki akisi goster:

```text
Svelte Frontend
      |
      v
FastAPI Backend
      |
      v
SegmentationOrchestrator
      |
      v
RabbitMQ Topic Exchange
      |
      +--> custom_librosa worker
      +--> MSAF Foote worker
      +--> MSAF CNMF worker
      +--> MSAF SCluster worker
      +--> LLM worker (opsiyonel, fusion baseline'i degil)
      |
      v
ResultListener --> PostgreSQL --> status/SSE/frontend
      |
      +--> Fusion worker, base sonuclar tamamlaninca
```

Ana mesaj:

- Backend audio analizini kendisi yapmaz; isi worker'lara dagitir.
- RabbitMQ servisleri birbirinden ayirir.
- PostgreSQL task durumunu ve sonuclari kalici tutar.
- Result listener sadece kayit yapan pasif bir servis degildir; fusion lifecycle'ini yonetir.

### Slayt 4 - Bir request sistemde nasil ilerler?

Upload ve storage akisini birlikte anlat:

1. Frontend `POST /segmentation/upload` veya `POST /segmentation/from-storage` cagrisi yapar.
2. API algorithms ve params girdisini Pydantic semalariyla dogrular.
3. `SegmentationOrchestrator` algoritma isimlerini canonical hale getirir.
4. Fusion istenmisse base algoritmalar otomatik olarak expected ve dispatch listesine eklenir.
5. Task PostgreSQL'e `processing` olarak yazilir.
6. Her base algoritma icin RabbitMQ'ya ayri routing key ile mesaj publish edilir.
7. Worker sonucu `segmentation.result` olarak yayinlar.
8. `ResultListener` sonucu normalize eder, DB'ye kaydeder ve fusion kosullarini kontrol eder.
9. Fusion tamamlandiginda task'in tum expected algoritmalari gelmisse task `completed` olur.
10. Frontend polling, SSE veya status endpoint'i ile sonucu alir.

### Slayt 5 - Common result schema

Bu slayt fusion'dan once zorunludur. Fusion'in calisabilmesi icin algoritmalarin ortak bir dilde sonuc vermesi gerekir.

Temel nesneler:

- `Boundary`: zaman, confidence, source, sources ve metadata.
- `Segment`: start, end, structural label, semantic label ve confidence bilgileri.
- `AlgorithmResult`: task, status, worker, algorithm, duration, boundaries, segments ve diagnostics.

Ornek:

```json
{
  "task_id": "...",
  "status": "completed",
  "worker_type": "msaf",
  "algorithm": "foote",
  "duration_seconds": 180.2,
  "boundaries": [
    {"time": 31.4, "confidence": 1.0, "source": "foote"}
  ],
  "segments": [
    {
      "start": 0.0,
      "end": 31.4,
      "label": "A",
      "structural_label": "A",
      "semantic_label": "Intro",
      "section_type": "Intro"
    }
  ],
  "diagnostics": {}
}
```

### Slayt 6-9 - Segmentasyon yontemleri

Sirasiyla anlat:

- `custom_librosa`
- MSAF Foote
- MSAF CNMF
- MSAF SCluster

Bu bolumde amac algoritmalari sadece isim olarak saymak degil; birbirinden farkli hata profilleri urettiklerini gostermektir. Bu farklilik algorithm-level fusion'in temel gerekcesidir.

### Slayt 10-12 - Feature-level fusion

Uc slayta bol:

1. Candidate kaynaklari.
2. Gruplama, agirlik, confidence ve threshold.
3. Boundary snapping, minimum duration ve fallback.

### Slayt 13-16 - Algorithm-level fusion

Sunumun en onemli bolumu. Dort slayta bol:

1. Fusion neden ayri bir worker ve neden audio segmenter degil?
2. ResultListener base result lifecycle'i.
3. Weighted boundary voting formulu ve ornek hesap.
4. Post-processing, diagnostics, failure ve timeout davranisi.

### Slayt 17 - Structural ve semantic labeling ayrimi

Vurgula:

- `A/B/C`, birbirine benzeyen segment siniflarini temsil eder.
- `Verse/Chorus`, muziksel anlam iddiasidir ve daha zayif/heuristic bir katmandir.
- Semantik isimler yapisal label'i overwrite etmez.
- Bu ayrim bilimsel olarak daha durust ve evaluation acisindan daha temizdir.

### Slayt 18 - Evaluation

Anlat:

- SALAMI reference segment interval'lari.
- `mir_eval.segment.detection(..., trim=True)`.
- `0.5s` strict ve `3.0s` lenient tolerans.
- Precision, recall ve F1.
- Over-segmentation ve under-segmentation.

### Slayt 19 - Demo

Canli demoda su sirayi izle:

1. Bir song sec veya upload et.
2. Dort base algoritma ile fusion'i sec.
3. Task'in processing oldugunu goster.
4. Base sonuclarin farkli boundary tahminlerini goster.
5. Fusion diagnostics'te boundary groups, sources ve scores alanlarini goster.
6. Final segment timeline'i goster.
7. Vakit varsa ayni track icin evaluation metriklerini goster.

### Slayt 20 - Sonuc ve sinirlamalar

Bitis mesaji:

> Sistemin katkisi tek bir algorithm'in varligi degil; farkli segmentation signal'larini ve algorithm'leri common schema, distributed orchestration, two-level fusion ve reproducible evaluation ile tek sistemde birlestirmesidir.

### 2.1 Slayt Bazli Ayrintili Konusma Rehberi

Bu bolum, her slaytta ekranda ne bulunmasi gerektigini ve senin hangi sirayla ne anlatacagini ayrintilandirir. Slayta butun paragraflari koyma. Slaytta ana kavramlar, diagram, formula ve kisa maddeler bulunsun; asagidaki aciklamalari speaker notes veya konusma metni olarak kullan.

#### Slayt 1 - Project Overview and Problem Definition

Slaytta bulunacaklar:

- Project title: `Automated Music Structure Segmentation`
- Input: audio file veya storage track
- Primary output: boundary timestamps ve segment intervals
- Secondary output: structural labels ve optional semantic labels
- Tek cumlelik hedef: "Detect where the musical structure changes."

Bu slaytta anlat:

> Bir music track tek parca halinde duyulsa da zaman icinde farkli structural section'lardan olusur. Intro, Verse, Chorus, Bridge ve Outro bunun insanlar tarafindan verilen semantic isimleridir. Bilgisayar icin ilk problem bu isimleri vermek degil, section'lar arasindaki transition timestamp'lerini bulmaktir. Biz bu timestamp'lere boundary diyoruz. Iki boundary arasindaki zaman araligi ise segment'tir. Sistem audio file'i aliyor, boundary prediction'lari uretiyor, bunlardan segment interval'lari olusturuyor ve sonra benzer segment'leri structural label'larla grupluyor.

Mutlaka acikla:

- `Boundary`: Bir section'in bittigi ve digerinin basladigi timestamp.
- `Segment`: Iki boundary arasindaki time interval.
- Boundary detection ana task'tir; labeling bundan sonra gelir.

Ornek ver:

```text
0s          20s              50s           80s
| Intro      | Verse          | Chorus       |
              ^                ^
           boundary         boundary
```

Gecis cumlesi:

> Problem basit gorunuyor, fakat ayni structural change her track'te ayni acoustic signal ile ortaya cikmiyor.

#### Slayt 2 - Why Music Segmentation Is Difficult

Slaytta bulunacaklar:

- Harmonic change
- Energy change
- Timbre/instrumentation change
- Rhythm/onset change
- Repetition structure
- Annotation uncertainty

Bu slaytta anlat:

> Bir section transition her zaman sessizlik veya sert bir ses degisimiyle gelmez. Bazen chord progression degisir, bazen drums eklenir, bazen vocalist girer, bazen energy ayni kalirken daha once duyulan Chorus tekrar eder. Bu nedenle tek bir feature her track icin yeterli degildir. Ayrica human annotator'lar ayni transition'i tam olarak ayni millisecond'a koymayabilir. Bu nedenle evaluation'da tolerance window kullanilir.

Somut ornekler:

- Verse'ten Chorus'a geciste RMS artabilir, fakat her zaman artmaz.
- Acoustic guitar ile devam eden iki section arasinda energy ayni kalabilir ama chord pattern degisebilir.
- Drum fill yuksek onset uretir fakat gercek bir structural boundary olmayabilir.
- Uzun bir fade-in, tek bir keskin boundary yerine belirsiz bir transition yaratabilir.

Ana mesaj:

> Farkli feature'lar farkli turde evidence uretir. Farkli algorithm'ler de bu evidence'i farkli mathematical assumptions ile yorumlar.

#### Slayt 3 - High-Level Architecture

Slaytta bulunacaklar:

- Frontend
- FastAPI backend
- SegmentationOrchestrator
- RabbitMQ
- Base workers
- ResultListener
- PostgreSQL
- Fusion worker

Bu slaytta diagram uzerinden soldan saga veya yukaridan asagi ilerle. Her component'i bir cumleyle acikla:

- `Frontend`: User input, algorithm selection ve result visualization.
- `FastAPI`: HTTP contract ve validation.
- `SegmentationOrchestrator`: Task creation ve worker dispatch.
- `RabbitMQ`: Asynchronous message transport.
- `Workers`: CPU-intensive audio analysis.
- `ResultListener`: Result aggregation, normalization ve fusion coordination.
- `PostgreSQL`: Task status ve persistent result storage.
- `Fusion worker`: Base result'lar hazir olduktan sonra algorithm-level fusion.

Anlatim metni:

> Backend audio analysis'i request thread'i icinde yapmiyor. Bunun yerine bir task olusturup RabbitMQ uzerinden secilen algorithm worker'larina dagitiyor. Boylece HTTP request uzun DSP islemlerini beklemiyor, algorithm'ler parallel calisabiliyor ve her worker bagimsiz scale edilebiliyor. ResultListener tum result'lari tek noktada topluyor. Fusion istenmisse base result'lar hazir olduktan sonra fusion worker'i yine RabbitMQ uzerinden baslatiliyor.

Neden bu architecture secildi?

- Long-running processing API'yi block etmez.
- Worker bazinda horizontal scaling yapilabilir.
- Bir algorithm fail olsa digerleri result uretebilir.
- Yeni bir algorithm yeni worker olarak eklenebilir.

#### Slayt 4 - End-to-End Request Lifecycle

Slaytta sequence diagram veya numarali flow kullan:

```text
Request -> Validate -> Create task -> Dispatch base workers
        -> Collect results -> Dispatch fusion -> Complete task
```

Bu slaytta anlat:

1. User audio upload eder veya storage'dan `song_id` secer.
2. API `algorithms` ve `params` alanlarini validate eder.
3. Orchestrator UUID tabanli `task_id` olusturur.
4. Task PostgreSQL'e `processing` status ile yazilir.
5. Base worker message'lari RabbitMQ'ya publish edilir.
6. Worker'lar result'larini `segmentation.result` routing key'ine publish eder.
7. ResultListener her result'i common schema'ya normalize eder.
8. Fusion request edildiyse listener readiness condition'i kontrol eder.
9. Fusion tamamlandiginda expected result set tamamlanir.
10. Task `completed` olur ve frontend result'i gosterir.

`expected` ve `dispatch` farkini acikla:

- `dispatch`: Simdi calistirilacak worker'lar.
- `expected`: Task tamamlanmadan once beklenen result key'leri.
- Fusion expected'tir ama ilk dispatch listesinde degildir.

#### Slayt 5 - Common Result Schema

Slaytta compact JSON ornegi ve uc ana model olsun:

- `Boundary`
- `Segment`
- `AlgorithmResult`

Bu slaytta anlat:

> Custom pipeline ve MSAF algorithm'leri raw olarak farkli formatlar uretebilir. Fusion ve evaluation'in algorithm-specific parser'lara donusmemesi icin tum output'lari common schema'ya normalize ediyoruz. Bu schema yalnizca data format degil, system components arasindaki contract'tir.

Alanlari tek tek acikla:

- `task_id`: Result'in hangi request'e ait oldugu.
- `status`: `completed` veya `failed`.
- `worker_type`: Result'i hangi worker'in uretdigi.
- `algorithm`: Canonical algorithm name.
- `duration_seconds`: Track duration.
- `boundaries`: Timestamp ve confidence iceren point listesi.
- `segments`: Start/end interval listesi.
- `diagnostics`: Algorithm'in neden bu karari verdigini anlamaya yarayan metadata.

`structural_label` ve `semantic_label` ayrimini ilk kez burada tanit:

> `structural_label=A` iki segment'in birbirine benzedigini soyler. `semantic_label=Chorus` ise bu cluster'a muziksel anlam atar. Ikinci iddia daha guclu oldugu icin ayri tutulur.

#### Slayt 6 - Custom Librosa Pipeline Overview

Slaytta pipeline flow goster:

```text
Audio
 -> active region
 -> feature extraction
 -> SSM and novelty
 -> candidate generation
 -> feature-level fusion
 -> snapping
 -> segments and labels
```

Bu slaytta anlat:

> `custom_librosa`, projenin deterministic multi-feature segmentation pipeline'idir. Tek bir feature'a dayanmaz. Harmonic structure icin Chroma-CENS, timbre icin MFCC, energy icin RMS, transient activity icin onset strength ve repetition icin Self-Similarity Matrix kullanir. Her source boundary candidate uretir. Feature-level fusion bu candidate'lari tek boundary set'ine indirir.

Deterministic ne demek?

> Ayni audio ve ayni params verildiginde random bir generative decision yerine ayni computational pipeline calisir ve reproducible output hedeflenir.

Bu slaytta henuz formula verme. Dinleyiciye pipeline'in genel haritasini kur.

#### Slayt 7 - Audio Loading and Active Region Detection

Slaytta bulunacaklar:

- ffmpeg decode
- Mono, 22050 Hz, float32
- RMS-based active region
- Crop analysis, restore original timestamps

Bu slaytta anlat:

> Audio farkli codec, sample rate veya channel count ile gelebilir. ffmpeg audio'yu decode edip mono ve 22050 Hz common representation'a cevirir. Mono kullanmak left/right channel farklarini tek signal'da toplar. 22050 Hz music structure analysis icin yeterli frequency range saglarken computational cost'u dusurur.

Active region neden var?

> Track basinda 8 saniye silence varsa bu kisim SSM ve segment duration hesaplarini etkileyebilir. `_detect_active_region()` RMS energy uzerinden musically active start ve end'i tahmin eder. Analysis crop edilen region'da yapilir, fakat final timestamp'lere `active_start` tekrar eklenir. Boylece output her zaman original full-track timeline ile uyumludur.

Burada RMS'i kisaca tanimla, ayrintisini temel terimler bolumunde verecegini soyle:

> RMS, short-time signal energy veya loudness proxy'sidir. Burada section label vermek icin degil, leading/trailing low-energy region'i bulmak icin kullanilir.

#### Slayt 8 - Chroma, MFCC and Self-Similarity Matrix

Slaytta uc gorsel kullanmak idealdir:

- Chroma feature map
- MFCC feature map
- SSM heatmap

Bu slaytta anlat:

> Chroma-CENS audio energy'yi 12 pitch class'a indirger. Farkli octave'lardaki ayni note ayni pitch class'ta toplanir. Bu nedenle harmonic progression ve repetition icin kullanislidir. MFCC ise spectral envelope'in compact representation'idir; instrumentation ve timbre change'lerini yakalamaya yardim eder.

SSM'i acikla:

> Self-Similarity Matrix, track'teki her time frame'i diger tum frame'lerle karsilastirir. Iki time region benzerse matrix'te bright block veya diagonal pattern goruruz. Repeated Verse veya Chorus section'lari bu matrix'te tekrar eden geometric pattern'ler olusturur.

Transposition-invariant SSM:

> Ayni motif farkli key'e transpose edilmisse exact chroma vector ayni olmayabilir. Sistem 12 pitch shift'i deneyip maximum similarity'yi alarak transposition'a daha tolerant bir representation uretir.

#### Slayt 9 - MSAF Baselines: Foote, CNMF and SCluster

Slaytta uc kolon kullan:

| Foote | CNMF | SCluster |
|---|---|---|
| Local novelty | Matrix factorization | Spectral clustering |
| Transition-focused | Latent pattern-focused | Global structure-focused |

Bu slaytta anlat:

> Uc MSAF method'u ayni problemi farkli mathematical perspective ile cozer. Foote local novelty ve checkerboard response'a odaklanir. CNMF feature matrix'i recurring latent component'lere factorize eder. SCluster similarity structure'i graph olarak yorumlayip spectral clustering uygular. Bu diversity algorithm-level fusion icin gereklidir. Ayni method'un dort parameter variation'ini birlestirmek yerine farkli error profile'lara sahip method'lari birlestiriyoruz.

Her method icin tek cumlelik risk:

- Foote: Local change'i yakalar fakat non-structural transient'e duyarli olabilir.
- CNMF: Repetition pattern'lerini yakalar fakat factorization choice result'i etkileyebilir.
- SCluster: Global structure'i yakalar fakat cluster granularity boundary sayisini etkileyebilir.

#### Slayt 10 - Feature-Level Candidate Sources

Slaytta source listesi ve "evidence" kelimesi bulunsun:

- SSM novelty: structural change evidence
- Chord proxy: harmonic change evidence
- RMS: energy change evidence
- Onset flux: transient/rhythmic change evidence
- Beat: rhythmic alignment evidence
- Lyrics: optional textual timing evidence

Bu slaytta her source'u acikla:

- `SSM novelty`: Before/after similarity structure degisiyor mu?
- `Chord proxy`: Chroma similarity ani dusuyor mu?
- `RMS`: Energy level'da ani change var mi?
- `Onset flux`: Yeni note veya drum attack yogunlugu degisiyor mu?
- `Beat`: Boundary nearby beat veya phrase grid'e hizalanabilir mi?
- `Lyrics`: Timed lyric line yeni section icin secondary evidence olabilir mi?

Ana mesaj:

> Bu source'larin hicbiri tek basina ground truth degildir. Her biri candidate ve confidence uretir. Fusion karari bunlarin agreement'ina dayanir.

#### Slayt 11 - Feature-Level Fusion Formula

Slaytta formula ve bir numeric example goster:

```text
weighted_sum = sum(source_weight * candidate_confidence)
agreement_bonus = min(0.15, 0.035 * (source_count - 1))
score = min(1.0, weighted_sum + agreement_bonus)
```

Numeric example:

```text
SSM:   weight 0.42 * confidence 0.80 = 0.336
Chord: weight 0.18 * confidence 0.70 = 0.126
RMS:   weight 0.06 * confidence 0.60 = 0.036
Agreement bonus for 3 sources          = 0.070
Final score                            = 0.568
```

Bu slaytta anlat:

> Once birbirine yakin candidate timestamp'leri temporal group haline getiriyoruz. Ayni source ayni group icinde birden fazla peak uretmisse yalnizca en yuksek confidence'li candidate'i kullaniyoruz. Sonra source weight ile candidate confidence'i carpiyoruz. Birden fazla independent source ayni region'i destekliyorsa agreement bonus ekliyoruz.

Acceptance kuralini acikla:

- Score threshold'u gecerse accepted.
- Strong SSM candidate varsa tek basina da korunabilir.
- Accepted candidate'lar arasinda minimum segment duration uygulanir.

#### Slayt 12 - Boundary Snapping and Feature Fusion Output

Slaytta before/after timestamp ornegi kullan:

```text
SSM candidate:       30.82s
Strong onset nearby: 30.67s
Snapped boundary:    30.67s
```

Bu slaytta anlat:

> SSM structural transition'in region'ini iyi bulabilir, ancak frame resolution ve smoothing nedeniyle exact timestamp bir miktar kayabilir. `snap_fused_boundaries()` accepted boundary'yi limited window icindeki strong onset veya beat'e hizalar. Structural detector "nerede" sorusunun region cevabini, onset ise daha precise timing cevabini saglar.

Minimum duration'i acikla:

> Iki accepted boundary birbirine cok yakinsa arada musically meaningful olmayan micro-segment olusabilir. Bu durumda confidence'i daha yuksek boundary tutulur.

Fallback'i acikla:

> Diger source'lar zayif oldugu icin fusion hic boundary kabul etmezse, strong SSM peaks fallback olarak kullanilabilir. Bu sistemin tamamen empty result vermesini azaltir.

#### Slayt 13 - Why Algorithm-Level Fusion?

Slaytta dort algorithm'in ayri timeline'larini goster:

```text
custom_librosa: |------|---------|------|
foote:          |-------|--------|------|
cnmf:           |------|----------|-----|
scluster:       |-------|--------|------|
fusion:         |------|---------|------|
```

Bu slaytta anlat:

> Feature-level fusion yalnizca custom pipeline icindedir. Fakat custom pipeline da her track'te perfect degildir. Bu nedenle ikinci fusion level'inda farkli segmentation algorithm'lerinin final boundary prediction'larini birlestiriyoruz. Buradaki input raw audio feature degil, tamamlanmis `AlgorithmResult` object'leridir.

Neden simple average degil?

- Her algorithm ayni sayida boundary uretmez.
- Hangi boundary'lerin ayni transition'a ait oldugu once belirlenmelidir.
- Algorithm reliability ayni kabul edilmez.
- Confidence degeri de vote'a dahil edilmelidir.

#### Slayt 14 - Fusion Orchestration and Readiness

Slaytta state flow goster:

```text
requested
 -> base workers dispatched
 -> base results collected
 -> fusion ready
 -> fusion dispatched
 -> fusion completed
```

Bu slaytta `_maybe_dispatch_fusion()` methodunu anlat:

> Orchestrator fusion request'ini gordugunde dort baseline'i expected ve dispatch listesine ekler, fakat fusion worker'ini hemen baslatmaz. ResultListener base result'lari toplar. Fusion yalnizca `custom_librosa`, `foote`, `cnmf` ve `scluster` result'larinin tamami resolved olduktan sonra dispatch edilir. Dort baseline'dan en az ikisi successful result uretmisse fusion calisir; aksi halde failed fusion result uretilir.

`resolved` ve `successful` farki:

- `resolved`: Worker cevap verdi; status completed veya failed olabilir.
- `successful`: Completed status ve kullanilabilir segment result'i var.

Failure davranisi:

> Ikiden az successful base result varsa fusion yapmak anlamsiz oldugu icin normalized failed fusion result uretilir.

#### Slayt 15 - Algorithm-Level Weighted Voting

Slaytta default weight'ler:

```text
custom_librosa = 0.35
scluster       = 0.30
cnmf           = 0.20
foote          = 0.15
```

Formula:

```text
group_score = sum(algorithm_weight * boundary_confidence)
accepted = group_score >= threshold
           OR unique_algorithm_count >= required_vote_count
```

Bu slaytta adim adim anlat:

1. Tum internal boundary vote'lari toplanir.
2. Start ve end edge'leri vote olmaktan cikarilir.
3. `merge_window_seconds` icindeki timestamp'ler ayni boundary group'a alinir.
4. Her algorithm group basina en fazla bir vote verir.
5. Weighted score hesaplanir.
6. Score threshold veya vote count condition ile acceptance karari verilir.

Numeric example'i mutlaka anlat:

```text
custom_librosa: 60.2s, confidence 0.90 -> 0.35 * 0.90 = 0.315
scluster:       61.0s, confidence 0.80 -> 0.30 * 0.80 = 0.240
group score = 0.555
threshold = 0.45
result = accepted
```

#### Slayt 16 - Fused Timestamp, Diagnostics and Failure Handling

Slaytta iki timestamp strategy goster:

- `weighted_mean`
- `custom_snap`

`weighted_mean`:

```text
sum(weight * confidence * time) / sum(weight * confidence)
```

`custom_snap`:

> Group icinde custom_librosa vote'u varsa onun snapped timestamp'ini anchor alir; yoksa weighted mean kullanir.

Diagnostics'i acikla:

- Hangi algorithm'ler vote verdi?
- Raw timestamp'ler neydi?
- Confidence'lar neydi?
- Group score neydi?
- Accepted mi rejected mi?
- Hangi algorithm failed veya pending kaldi?

Bu slaytin ana mesaji:

> Fusion black box degildir. Her final boundary'nin decision trace'i diagnostics icinde saklanir.

Known limitation'i durustce soyle:

> Fusion tum baseline result'larini bekler. Bir worker hic completed veya failed result publish etmeden kaybolursa task beklemeye devam eder. Bu davranis tum algorithm'leri fusion'a dahil etme kararinin sonucudur; production ortaminda watchdog, retry veya hard timeout ile desteklenmelidir.

#### Slayt 17 - Structural Labels vs Semantic Labels

Slaytta iki katmanli ornek kullan:

```text
Segment 1 -> structural A -> semantic Intro
Segment 2 -> structural B -> semantic Verse
Segment 3 -> structural C -> semantic Chorus
Segment 4 -> structural B -> semantic Verse
Segment 5 -> structural C -> semantic Chorus
```

Bu slaytta anlat:

> Structural label similarity claim'dir. B label'li iki segment birbirine audio descriptor acisindan benzer demektir. Semantic label muziksel role claim'idir. Bir section'a Chorus demek repetition, position ve energy gibi additional evidence gerektirir. Bu nedenle semantic label structural label'i overwrite etmez.

Descriptor'lari acikla:

- Chroma mean/std: Harmonic content.
- MFCC mean/std: Timbre.
- RMS mean/std: Energy.
- Onset density: Rhythmic activity.
- Duration ratio: Relative length.

Semantic heuristic'leri anlat:

- First content section near start -> Intro.
- Last content section near end -> Outro.
- Repeated higher-energy cluster -> Chorus candidate.
- Other repeated cluster -> Verse candidate.
- Unique middle section -> Bridge candidate.

#### Slayt 18 - Evaluation with SALAMI

Slaytta ground truth ve prediction timeline'i birlikte goster. Sonra metrics table koy:

| Metric | 0.5s | 3.0s |
|---|---:|---:|
| Precision | ... | ... |
| Recall | ... | ... |
| F1 | ... | ... |

Bu slaytta anlat:

> SALAMI dataset human-annotated music structure interval'lari saglar. Prediction ve reference segment'leri `(start, end)` interval matrix'e ceviriyoruz. `mir_eval.segment.detection(..., trim=True)` ile boundary detection metric'lerini hesapliyoruz.

Metric'leri basit dille acikla:

- `Precision`: Predict ettigimiz boundary'lerin ne kadari dogru?
- `Recall`: Ground-truth boundary'lerin ne kadarini bulduk?
- `F1`: Precision ve Recall'un harmonic mean'i.
- `0.5s`: Exact timing'e yakin strict evaluation.
- `3.0s`: Dogru transition region'ini bulmaya odaklanan lenient evaluation.

Interpretation example:

> F1@3s yuksek ama F1@0.5s dusukse algorithm structural transition region'ini buluyor fakat exact timestamp localization'i zayif. Ikisi de dusukse boundary detection veya boundary count problemi var.

#### Slayt 19 - Live Demo

Slaytta demo planini yaz; demo sirasinda bos slayt birakma:

1. Select a track.
2. Select four baselines and fusion.
3. Submit task.
4. Observe partial results.
5. Open fusion diagnostics.
6. Compare timelines.
7. Show evaluation if time permits.

Demo sirasinda anlat:

> Request gonderildiginde fusion worker hemen calismiyor. Once base worker'lar parallel dispatch ediliyor. Partial result'lar geldikce task processing kalmaya devam ediyor. Listener readiness condition saglandiginda fusion task'i olusturuyor. Final result'ta her fused boundary'nin sources ve raw_times alanlarini gorebiliriz.

Demo risk plani:

- Onceden basariyla tamamlanmis bir task ID hazir tut.
- Fusion diagnostics JSON'unun screenshot'ini hazir tut.
- Worker'lardan biri gecikirse bunu architecture'in asynchronous yapisini anlatmak icin kullan.
- Canli evaluation uzun surerse onceden kaydedilmis batch result goster.

#### Slayt 20 - Contributions, Limitations and Conclusion

Slaytta uc kolon kullan:

**Contributions**

- Distributed multi-algorithm pipeline
- Common result schema
- Feature-level fusion
- Algorithm-level fusion
- Multi-tolerance evaluation

**Limitations**

- Confidence calibration
- Static fusion weights
- Event-driven timeout limitation
- Heuristic semantic labels

**Future Work**

- Learned weights
- Genre-adaptive fusion
- Watchdog and retry
- Better visualization

Kapanis konusmasi:

> Bu projenin katkisi yalnizca yeni bir boundary detector yazmak degildir. Farkli acoustic feature'lari custom pipeline icinde, farkli segmentation algorithm'lerini ise ikinci fusion level'inda birlestiren end-to-end bir system kurduk. Tum result'lari common schema ile standardize ettik, decision diagnostics ekledik ve output'lari SALAMI reference annotation'lariyla strict ve lenient tolerance'larda evaluate ettik. Sistem deterministic, modular, explainable ve yeni worker'larla genisletilebilir durumdadir.

### 2.2 Temel Audio ve Segmentation Terimleri

Bu bolumdeki tanimlari ezberlemek yerine mantigini anlamaya calis. Hocanin "RMS tam olarak nedir?" veya "SSM neyi temsil ediyor?" sorularina bu aciklamalarla cevap verebilirsin.

#### Digital Audio Signal

Digital audio, air pressure variation'in zaman icinde sayisal sample'lara donusturulmus halidir. Bir waveform'da:

- Horizontal axis: time.
- Vertical axis: amplitude.
- Positive/negative amplitude speaker diaphragm'in iki yonlu hareketini temsil eder.

Tek bir sample muzik yapisini anlatmaz. Bu nedenle signal kisa frame'lere bolunur ve her frame'den feature cikarilir.

#### Sample Rate

`Sample rate`, bir saniyede kac audio sample alindigidir. `22050 Hz`, saniyede 22,050 sample demektir. Nyquist principle'a gore temsil edilebilen maximum frequency sample rate'in yarisidir; 22050 Hz sample rate yaklasik 11025 Hz'e kadar frequency content tasir.

Bu projede 22050 Hz neden kullaniliyor?

- Music structure icin gerekli harmonic/rhythmic information'in buyuk kismini korur.
- 44100 Hz'e gore memory ve computation cost'u azaltir.
- Librosa music information retrieval pipeline'larinda yaygin bir choice'tur.

#### Mono

Stereo audio left ve right olmak uzere iki channel tasir. Mono conversion bu channel'lari tek signal'da birlestirir. Structure analysis icin channel-specific spatial information genellikle primary evidence degildir; mono computation'i azaltir ve representation'i standardize eder.

#### Frame ve Hop Length

Audio feature'lari tum track icin tek deger olarak degil, kisa overlapping window'lar uzerinde hesaplanir.

- `Frame`: Feature hesaplanan kisa audio window.
- `Hop length`: Bir frame'den sonraki frame'e kac sample ilerledigimiz.

Projede raw feature extraction icin `hop_length=512` kullanilir. `sr=22050` icin yaklasik frame step:

```text
512 / 22050 ~= 0.0232 seconds
```

Yani raw feature timeline saniyede yaklasik 43 observation uretir. Daha sonra median pooling ile target FPS dusurulur.

#### Amplitude

Amplitude waveform'un instantaneous magnitude'udur. Tek basina perceived loudness ile birebir ayni degildir, fakat signal energy hesabinin temelidir.

#### RMS - Root Mean Square

RMS, bir audio frame icindeki sample amplitude'larinin effective magnitude'ini olcer. Formula:

```text
RMS = sqrt((x1^2 + x2^2 + ... + xN^2) / N)
```

Adimlar:

1. Her sample square edilir. Boylece negative ve positive amplitude birbirini goturmez.
2. Squared values'in mean'i alinir.
3. Square root alinarak tekrar amplitude scale'e donulur.

RMS neyi temsil eder?

- Short-time signal energy icin bir proxy.
- Genellikle perceived loudness ile iliskilidir, ancak psychoacoustic loudness'in tam modeli degildir.
- Silence veya low-energy region'da dusuktur.
- Loud chorus, drum entry veya instrumentation growth durumunda yukselebilir.

Projede RMS iki yerde kullanilir:

1. `active region detection`: Leading/trailing low-energy region'i bulmak.
2. `RMS boundary candidates`: RMS curve'deki ani change'leri boundary evidence olarak kullanmak.

Onemli savunma:

> High RMS tek basina Chorus demek degildir. RMS yalnizca energy evidence saglar. Bu nedenle weight'i dusuktur ve diger feature'larla fusion'a girer.

#### Decibel - dB

Audio amplitude cok genis dynamic range'e sahiptir. dB logarithmic scale kullanarak bu araligi daha okunabilir hale getirir.

Amplitude ratio icin genel ifade:

```text
dB = 20 * log10(amplitude / reference_amplitude)
```

Librosa'da reference olarak maximum amplitude kullanildiginda en yuksek nokta yaklasik `0 dB`, daha sessiz noktalar negative dB olur. Ornegin `-20 dB`, reference maximum'dan belirgin derecede dusuk level demektir.

Projede active threshold:

```text
P75(RMS_dB) - margin_db
```

Bu fixed threshold yerine track'in kendi dynamic range'ine uyum saglayan adaptive threshold uretir.

#### Percentile - P75

`P75`, degerlerin yuzde 75'inin altinda kaldigi point'tir. RMS distribution'da P75, track'in daha active energy region'larini temsil eden robust bir reference saglar. Maximum kullanmaktan daha dayaniklidir; tek bir loud transient threshold'u bozmaz.

#### Gaussian Smoothing

Raw feature curve'lari frame-level noise tasiyabilir. Gaussian smoothing yakin frame'leri weighted average ile yumusatir. Yakindaki frame daha yuksek, uzaktaki frame daha dusuk weight alir.

Amac:

- Small fluctuation'lari azaltmak.
- Meaningful trend ve peak'leri one cikarmak.

Risk:

- Fazla smoothing boundary timestamp'ini kaydirabilir veya iki yakin transition'i birlestirebilir.

#### Spectrogram

Spectrogram, audio signal'in time-frequency representation'idir:

- Horizontal axis: time.
- Vertical axis: frequency.
- Color/intensity: O frequency'deki energy.

MFCC ve bazi onset feature'lari frequency-domain representation'dan turetilir.

#### Chroma Feature

Chroma, frequency content'i 12 pitch class'a map eder:

```text
C, C#, D, D#, E, F, F#, G, G#, A, A#, B
```

Farkli octave'lardaki ayni note ayni class'ta toplanir. Ornegin low C ve high C ayni chroma bin'e katkida bulunur.

Neden useful?

- Chord ve harmonic progression'i compact temsil eder.
- Verse veya Chorus tekrarlarini harmonic pattern uzerinden yakalayabilir.
- Exact instrumentation'dan daha az etkilenebilir.

#### Chroma-CENS

`CENS`, Chroma Energy Normalized Statistics anlamina gelir. Standard chroma'ya gore quantization ve temporal smoothing kullanarak dynamics ve articulation degisimlerine daha robust harmonic representation hedefler.

Neden structure analysis icin uygun?

> Section repetition'i ararken tek tek note attack'larindan cok daha uzun sureli harmonic progression onemlidir. CENS bunu smooth ve normalized bicimde temsil eder.

#### MFCC - Mel-Frequency Cepstral Coefficients

MFCC, short-time spectrum'un spectral envelope'ini compact coefficient'larla temsil eder. Human auditory perception'a yaklasan Mel frequency scale kullanilir.

Basit yorum:

- Chroma "hangi pitch classes aktif?" sorusuna yakindir.
- MFCC "sound'un timbral shape'i nasil?" sorusuna yakindir.

MFCC hangi degisimleri yakalayabilir?

- Guitar'dan full band'e gecis.
- Vocal'in girmesi veya cikmasi.
- Instrumentation ve texture change.

Projede `MFCC0` neden atiliyor?

> MFCC0 overall log-energy ile cok iliskilidir. Bu coefficient similarity hesabini domine ederse timbre SSM aslinda loudness SSM'e donusebilir. Bu nedenle bir fazla coefficient hesaplanip ilki atilir.

#### Normalization

Feature dimension'larinin scale'leri farkli olabilir. Normalization, bir feature'in yalnizca numeric magnitude'i buyuk oldugu icin similarity'yi domine etmesini engeller.

- `Z-score`: Mean'i 0, standard deviation'i 1 yapar.
- `L2 normalization`: Vector length'i 1 yapar.
- `Min-max normalization`: Curve'u genellikle 0-1 range'e getirir.

#### Median Pooling

Birden fazla adjacent frame'in element-wise median'ini alarak frame rate dusurulur.

Neden median?

- Single-frame outlier ve transient'lere mean'den daha robust'tur.
- SSM matrix boyutunu ve `O(N^2)` computation cost'unu azaltir.
- Section-level analysis icin gereksiz high temporal resolution'i azaltir.

#### Timbre

Timbre, ayni pitch ve loudness'a sahip iki sound'u birbirinden ayiran tonal character'dir. Piano ile guitar ayni note'u calsa bile farkli duyulmasi timbre farkidir. MFCC timbre icin yaygin bir representation'dir.

#### Self-Similarity Matrix - SSM

SSM, track'teki her frame'i diger her frame ile karsilastiran square matrix'tir:

```text
S[n, m] = similarity(feature_at_n, feature_at_m)
```

Matrix neden square?

> Hem row hem column ayni track'in time axis'ini temsil eder.

Diagonal neden parlak?

> Her frame kendisiyle maximum similarity'ye sahiptir.

Off-diagonal block veya diagonal line ne demek?

> Track'in farkli time region'larinda benzer music material tekrar ediyor olabilir.

#### Cosine Similarity

Cosine similarity iki vector arasindaki angle'i olcer:

```text
cosine_similarity(a, b) = (a dot b) / (||a|| * ||b||)
```

L2-normalized vector'larda dot product cosine similarity'ye esittir. Direction benzerligini olctugu icin overall magnitude farkindan daha az etkilenir.

#### Transposition Invariance

Bir melody veya chord progression farkli key'de tekrar edebilir. Transposition-invariant chroma SSM, chroma vector'u 12 possible pitch shift ile kaydirir ve maximum similarity'yi alir.

Avantaj:

- Key change olsa da structural repetition'i yakalayabilir.

Risk:

- Farkli harmonic content bazen gerektiginden fazla benzer kabul edilebilir.

#### Diagonal Smoothing

SSM'de repeated sequence'ler diagonal path olusturur. Diagonal smoothing bu path boyunca similarity'yi average ederek consistent repetition'i guclendirir ve isolated noise'u azaltir.

Tempo-invariant smoothing farkli slope'lari dener. Cunku ayni sequence biraz daha hizli veya yavas tekrarlandiginda diagonal slope degisir.

#### Novelty Curve

Novelty curve, her timestamp icin "burada structural change olma ihtimali ne kadar?" sorusuna scalar response verir.

- Horizontal axis: time.
- Vertical axis: novelty score.
- High peak: Potential boundary.

Novelty probability olmak zorunda degildir. Normalize edilmis change response'tur.

#### Checkerboard Kernel

Checkerboard kernel, SSM diagonal'i etrafindaki before/after block structure'i test eder.

Ideal boundary etrafinda:

- Before-before similarity yuksek.
- After-after similarity yuksek.
- Before-after similarity dusuk.

Kernel'in ayni isaretli quadrant'lari within-section similarity'yi positive, farkli quadrant'lari cross-section similarity'yi negative agirliklandirir. Inner product yuksekse iki farkli homogeneous region arasinda transition olabilir.

#### Kernel Size

Kernel size novelty'nin baktigi temporal scale'i belirler.

- Cok kucuk kernel: Note veya beat-level change'lere fazla duyarli.
- Cok buyuk kernel: Kisa section'lari kacirabilir ve timestamp'i smooth edebilir.

Projede kernel seconds parametresi section-level transition hedeflenerek secilir.

#### Peak Detection

Novelty curve'deki local maximum'lar `find_peaks` ile bulunur. Her local maximum boundary degildir. Ek kosullar kullanilir:

- Minimum prominence.
- Minimum distance.
- Edge margin.

#### Prominence

Peak prominence, bir peak'in surrounding baseline'a gore ne kadar belirgin oldugunu olcer. Sadece absolute height degil, komsu valley'lere gore ne kadar one ciktigiyla ilgilidir.

- Dusuk prominence threshold: Daha fazla candidate, daha yuksek recall, daha fazla false positive riski.
- Yuksek prominence threshold: Daha az candidate, daha yuksek precision ihtimali, missed boundary riski.

#### Onset

Onset, bir sound event'in baslangic anidir. Note attack, drum hit veya vocal entry onset olabilir.

Onset ile boundary ayni sey degildir:

- Her boundary yakininda onset olabilir.
- Fakat her onset structural boundary degildir.
- Drum pattern icinde yuzlerce onset bulunabilir.

Bu nedenle onset secondary timing evidence olarak kullanilir.

#### Onset Strength ve Spectral Flux

Onset strength, spectrum'un frame'den frame'e ne kadar arttigini olcen curve'dur. Spectral flux, frequency bin energy'lerindeki positive change'i toplar. Yeni sound event geldiginde curve yukselebilir.

Projede iki amacla kullanilir:

- Boundary candidate uretmek.
- SSM ile bulunan region'daki timestamp'i precise onset'e snap etmek.

#### Beat ve Tempo

- `Beat`: Perceived rhythmic pulse timestamp'leri.
- `Tempo`: Beat frequency, genellikle BPM ile ifade edilir.
- `BPM`: Beats per minute.

Beat boundary'nin varligini kanitlamaz. Ancak music section transition'lari siklikla beat veya bar boundary'lerine hizalandigi icin timestamp refinement'ta useful olabilir.

#### Chord Proxy

Sistem full symbolic chord recognition yapmak zorunda degildir. Chroma vector'lar arasindaki similarity dususunu harmonic change proxy'si olarak kullanir.

"Proxy" ne demek?

> Dogrudan chord name tahmin etmiyoruz; chord change ile iliskili olabilecek measurable signal kullaniyoruz.

#### Candidate Boundary

Candidate boundary, henuz final karar olmayan potential timestamp'tir. Tipik alanlari:

```json
{
  "time": 42.3,
  "source": "rms",
  "confidence": 0.71
}
```

Feature-level fusion candidate'lari group'lar, score hesaplar ve accepted/rejected karari verir.

#### Confidence

Confidence, source'un kendi evidence strength'ini `0-1` range'inde ifade eder. Her algorithm veya feature icin ayni statistical calibration'a sahip olmak zorunda degildir.

Onemli nokta:

> Confidence ground-truth probability degildir. Bu projede normalized evidence strength olarak kullanilir.

#### Weight

Weight, system designer'in bir source veya algorithm'e verdigi prior importance'tir.

- Confidence: Bu specific boundary icin evidence ne kadar guclu?
- Weight: Bu source'a genel olarak ne kadar guveniyoruz?

Fusion contribution:

```text
contribution = weight * confidence
```

#### Threshold

Threshold, score'un accepted sayilmasi icin gecmesi gereken minimum degerdir.

- Threshold dusurse recall artabilir, false positive de artabilir.
- Threshold yukselirse precision artabilir, missed boundary de artabilir.

#### Merge Window

Farkli source veya algorithm'ler ayni transition'i tam ayni timestamp'te bulmayabilir. `merge_window_seconds`, birbirine ne kadar yakin vote'larin ayni boundary group sayilacagini belirler.

Window cok kucukse:

- Ayni transition farkli group'lara bolunebilir.

Window cok buyukse:

- Gercekte farkli iki transition tek group'ta birlesebilir.

#### Boundary Snapping

Snapping, coarse structural boundary'yi yakin bir precise event timestamp'ine tasimaktir. Bu projede nearby strong onset veya beat kullanilabilir. Snap distance limited tutulur.

#### Minimum Segment Duration

Iki boundary arasindaki minimum allowed duration'dir. Section-level analysis'te birkac yuz millisecond'luk segment genellikle meaningful structural section degildir.

Bu parametre:

- Over-segmentation'i azaltir.
- Fakat cok yuksek secilirse gercek short section'lari silebilir.

#### Clustering

Clustering, label bilinmeden benzer data point'leri group'lama islemidir. Segment descriptor'lari benzerse ayni structural cluster'a atanabilir.

#### Agglomerative Clustering

Bottom-up hierarchical clustering method'udur:

1. Her segment ayri cluster olarak baslar.
2. En benzer cluster'lar birlestirilir.
3. Istenen cluster count'a kadar devam edilir.

#### K-Means

Data point'leri `k` centroid etrafinda group'lar. Her point en yakin centroid'e atanir, centroid'ler tekrar hesaplanir. Custom pipeline'da fallback veya segment clustering context'inde kullanilabilir.

#### Silhouette Score

Bir point'in kendi cluster'ina ne kadar yakin, diger cluster'lara ne kadar uzak oldugunu olcer. Yaklasik range `-1` ile `1` arasindadir.

- Yuksek score: Cluster'lar daha ayrik.
- Zero civari: Cluster overlap.
- Negative: Point muhtemelen yanlis cluster'da.

System farkli `k` degerlerini deneyip daha iyi silhouette score veren cluster count'u secebilir.

#### Structural Label

`A`, `B`, `C` gibi label'lar segment similarity group'unu temsil eder. Semantic meaning garanti etmez.

Ornek:

```text
A B C B C D
```

Burada B ve C tekrar ediyor olabilir. B'nin Verse, C'nin Chorus oldugu ancak additional evidence ile tahmin edilebilir.

#### Semantic Label

`Intro`, `Verse`, `Chorus`, `Bridge`, `Outro` gibi human-interpretable music role label'idir. Bu projede heuristic ve conservative olarak atanir; structural label'dan ayri saklanir.

#### Heuristic

Heuristic, kesin mathematical guarantee yerine domain knowledge'e dayali practical rule'dur. Ornegin track'in ilk yuzde 20'sindeki unique first section'i Intro candidate saymak heuristic'tir.

#### Ground Truth ve Reference Annotation

Ground truth, evaluation'da dogru kabul edilen human annotation'dir. Music segmentation'da human disagreement olabilecegi icin "absolute truth" yerine `reference annotation` demek daha dikkatli bir ifadedir.

#### Precision

Predict edilen boundary'lerin ne kadarinin reference boundary ile match oldugunu olcer:

```text
Precision = TP / (TP + FP)
```

Low precision genellikle fazla boundary prediction'i, yani over-segmentation ile iliskilidir.

#### Recall

Reference boundary'lerin ne kadarinin detect edildigini olcer:

```text
Recall = TP / (TP + FN)
```

Low recall genellikle missed boundary, yani under-segmentation ile iliskilidir.

#### F1 Score

Precision ve Recall'un harmonic mean'idir:

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

Arithmetic mean yerine harmonic mean kullanilmasi, degerlerden biri cok dusukse score'un yuksek gorunmesini engeller.

#### Tolerance Window

Prediction timestamp'i reference timestamp'e tolerance kadar yakinsa match kabul edilir.

- `0.5s`: Strict localization.
- `3.0s`: Lenient structural-region detection.

Tolerance, ayni prediction'in birden fazla reference boundary ile eslesmesine izin vermemelidir; matching one-to-one yapilir.

#### Over-Segmentation

System'in gerektiginden fazla boundary uretmesidir. Tipik sonucu:

- False positive artar.
- Precision duser.
- Segment'ler gereksiz yere kuculur.

#### Under-Segmentation

System'in gercek boundary'leri kacirip gerektiginden az segment uretmesidir. Tipik sonucu:

- False negative artar.
- Recall duser.
- Farkli structural section'lar tek segment icinde kalir.

#### Baseline

Baseline, yeni method'un karsilastirildigi established veya simpler reference method'dur. Foote, CNMF ve SCluster burada farkli MSAF baseline'lari olarak kullanilir.

#### Deterministic

Ayni input ve params ile ayni computational steps'in calismasi ve reproducible result hedeflenmesidir. Floating-point, library veya environment farklari yine kucuk fark yaratabilir; deterministic ifadesi random generative decision olmadigini belirtir.

#### Diagnostics

Diagnostics, final output disinda decision process'i anlamaya yarayan metadata'dir. Fusion icin raw vote'lar, source listesi, score, acceptance, missing algorithm ve params burada tutulur.

#### Explainability

Explainability, system'in bir boundary'yi neden kabul ettigini trace edebilmesidir. Bu projede fusion boundary group diagnostics'i explainability'nin temelidir.

---

## 3. Uygulamanin Genel Mimarisi

### 3.1 Frontend

Frontend Svelte ile yazilmistir. Kullanici burada:

- Audio upload edebilir.
- Storage'daki bir parcayi secebilir.
- Calistirilacak algoritmalari belirleyebilir.
- Algoritmaya ozel parametreler gonderebilir.
- Task durumunu izleyebilir.
- Algoritma bazli segmentleri ve evaluation sonuclarini gorebilir.
- Batch evaluation baslatabilir.

Frontend'in teknik olarak bilmesi gereken en onemli ayrim, gercek algoritma key'leri ile metadata key'leridir. DB sonucunda `foote` gibi key segment listesini; `foote__result`, `foote__diagnostics` ve `foote__boundaries` gibi key'ler ayrintili metadata'yi tasir. UI render ederken `__` iceren alanlar normal algoritma sonucu gibi ele alinmaz.

### 3.2 FastAPI API katmani

API'nin sorumluluklari:

- HTTP request kabul etmek.
- Request body'sini veya multipart alanlarini validate etmek.
- Orchestrator'u cagirmak.
- Task ID dondurmek.
- Status ve evaluation endpoint'lerini sunmak.

API audio feature extraction yapmaz. Bu ayrim onemlidir: HTTP lifecycle'i uzun sureli DSP hesaplamalariyla bloke edilmez.

`backend/api/schemas.py` icindeki semalar parametrelerin sinirlarini belirler. Ornegin:

- `merge_window_seconds > 0`
- `threshold` degeri `0.0-1.0`
- `required_vote_count` degeri `1-4`
- `anchor_strategy`, `weighted_mean` veya `custom_snap`

Bu validation, hatali parametrenin worker'a kadar gidip gec fark edilmesini engeller.

### 3.3 SegmentationOrchestrator

Dosya: `backend/services/segmentation_orchestrator.py`

Bu sinif request ile worker execution arasindaki ana koordinasyon katmanidir.

#### `__init__()`

Algoritmalari RabbitMQ routing key'lerine map eder:

```text
custom_librosa -> segmentation.custom
foote          -> segmentation.foote
cnmf           -> segmentation.cnmf
scluster       -> segmentation.scluster
fusion         -> segmentation.fusion
llm            -> segmentation.llm
```

#### `_normalize_algorithms(requested_algos)`

- `custom` gibi legacy ismi `custom_librosa` canonical ismine cevirir.
- Sadece izin verilen algoritmalari kabul eder.
- Duplicate algoritmalari eler.
- Gecerli algoritma kalmazsa hata verir.

Neden gerekli: Sonuc key'leri, expected algoritmalar ve routing mantigi ayni canonical isimleri kullanmazsa task hic tamamlanmayabilir.

#### `_expand_requested_algorithms(algorithms)`

Fusion lifecycle'inin ilk kritik methodudur.

- `expected`: task tamamlanmadan once gelmesi beklenen tum sonuclar.
- `dispatch`: hemen RabbitMQ'ya gonderilecek algoritmalar.

Fusion istenmediyse iki liste aynidir. Fusion istenmisse:

- `fusion`, expected listesinde kalir.
- `custom_librosa`, `foote`, `cnmf`, `scluster` expected listesine eklenir.
- Fusion ilk anda dispatch edilmez.
- Dort base algoritma dispatch listesine eklenir.

Bu tasarimin nedeni fusion worker'inin ham audio'dan bagimsiz bir boundary seti cikarmamasi, tamamlanmis base sonucuna ihtiyac duymasidir.

#### `_validate_and_trim_params(params, algorithms)`

Yalnizca istenen algoritmalarla ilgili parametreleri task payload'inda tutar. Ornegin MSAF secilmediyse `msaf` parametreleri, fusion secilmediyse `fusion` parametreleri cikarilir.

#### `_create_task_record(...)`

Task'i PostgreSQL'e kaydeder:

- `status=processing`
- `results={}`
- `expected_algorithms`
- source bilgisi
- request parametreleri
- opsiyonel webhook URL

Task kaydi publish'ten once yapilir. Boylece worker cok hizli donse bile listener'in bulabilecegi bir task kaydi vardir.

#### `_publish_tasks(task_payload, algorithms)`

Her algoritma icin ayni task baglamini ilgili routing key'e publish eder. RabbitMQ topic exchange sayesinde her worker yalnizca kendi queue'suna bagli mesaji alir.

#### `process_upload(...)`

- Audio'yu chunk'lar halinde upload dizinine yazar.
- Task ID ve dosya adini olusturur.
- DB kaydini acar.
- Payload icine `file_path`, content type, algorithms ve params koyar.
- Base worker task'larini publish eder.

#### `process_from_storage(...)`

- `song_id` dogrular.
- Azure Blob icinde `songs/{song_id}.mp3` var mi kontrol eder.
- Source type'i `storage` olarak kaydeder.
- Worker'a local path yerine blob bilgisini iletir.

### 3.4 RabbitMQ

RabbitMQ bu projede yalnizca queue degil, servisler arasi execution boundary'sidir.

Kazandirdiklari:

- Backend worker'in bitmesini beklemez.
- Her algoritma bagimsiz scale edilebilir.
- Custom worker sayisi CPU ihtiyacina gore artirilabilir.
- Bir algoritmanin yavasligi diger algoritmanin sonuc uretmesini engellemez.
- Sonuclar tek `segmentation.result` kanalinda listener'a akar.

### 3.5 BaseWorker

Dosya: `workers/BaseWorker.py`

Tum worker'larin ortak lifecycle'ini yonetir:

1. Queue'dan task alma.
2. Storage girdisiyse audio'yu local cache'e indirme.
3. Alt sinifin `process_task()` methodunu cagirma.
4. Sonucu `segmentation.result` routing key'i ile publish etme.
5. Mesaji ACK etme.
6. Exception durumunda failed result publish etme.

Failed result publish edilmesi fusion icin kritiktir. Worker sadece exception firlatip kaybolursa listener bu algoritmanin hala calistigini mi yoksa tamamen basarisiz oldugunu mu anlayamaz. Explicit failed payload, algoritmayi `resolved but failed` hale getirir.

### 3.6 ResultListener

Dosya: `backend/services/result_listener.py`

Result listener sistemin result aggregation ve fusion coordination merkezidir.

#### `_result_key(worker_type, algorithm)`

Legacy `custom` worker type'ini `custom_librosa` olarak saklar; digerlerini canonical hale getirir.

#### `_visible_result_keys(current_results)`

Yalnizca kullaniciya gosterilen algoritma segment listelerini sayar. `__result`, `__diagnostics` gibi metadata key'lerini expected result olarak saymaz.

#### `_normalized_result(data, key)`

Worker zaten common schema'da result vermisse onu korur. Legacy bir segment listesi veya eksik schema geldiyse `normalize_algorithm_result()` ile standard format'a getirir.

#### `_process_result(...)`

Her sonucu aldiginda:

1. Task ve algorithm key'i bulur.
2. Sonucu normalize eder.
3. Segment listesini `results[algorithm]` altina yazar.
4. Full sonucu `results[algorithm__result]` altina yazar.
5. Diagnostics, boundaries ve duration alanlarini ayri metadata key'lerinde saklar.
6. `_maybe_dispatch_fusion()` cagrisi yapar.
7. Gelen visible algoritmalar expected set'i kapsiyorsa task'i `completed` yapar.
8. Partial veya final update'i SSE/webhook mekanizmasina yollar.
9. DB transaction'ini commit eder ve RabbitMQ mesajini ACK eder.

Bu yapida backwards compatibility korunur: eski frontend segment listesini okumaya devam ederken yeni kod full normalized result ve diagnostics okuyabilir.

---

## 4. Common Segmentation Data Model

### 4.1 `Boundary`

Dosya: `shared/segmentation_models.py`

Bir boundary su field'lari tasiyabilir:

- `time`: saniye cinsinden konum.
- `confidence`: `0-1` arasi confidence score.
- `source`: boundary'yi ureten ana kaynak.
- `sources`: fusion varsa katkida bulunan kaynaklar.
- `metadata`: score veya raw vote gibi aciklanabilirlik bilgisi.

Boundary, segment'ten farklidir. Boundary tek bir timestamp; segment iki boundary arasindaki interval'dir.

### 4.2 `Segment`

Temel alanlar:

- `start`, `end`
- `label`, `structural_label`
- `semantic_label`, `section_type`
- `label_confidence`, `label_method`
- `semantic_confidence`, `semantic_reason`
- `confidence`, `sources`

`label` ve `section_type` alanlari eski frontend contract'ini korur. Yeni ve daha acik alanlar `structural_label` ve `semantic_label`'dir.

### 4.3 `AlgorithmResult`

Her worker result'i ayni top-level contract'a donusturulur:

- Hangi worker uretti?
- Hangi algorithm?
- Status nedir?
- Audio duration nedir?
- Boundary'ler neler?
- Segment'ler neler?
- Hangi diagnostics uretildi?

Bu normalization olmadan fusion servisinin her algoritma icin ayri parser yazmasi gerekirdi.

### 4.4 `normalize_boundaries()`

Dosya: `shared/segmentation_utils.py`

Methodun amaci boundary listelerini valid ve consistent hale getirmektir:

- Numeric olmayan degerleri eler.
- Negatif veya duration disi zamanlari eler.
- Siralar.
- Birbirine cok yakin boundary'leri deduplicate eder.
- Baslangic `0.0` ve bitis `duration` edge'lerini ekler.

### 4.5 `segments_to_intervals()`

Segment dict'lerini dogrudan NumPy `(n, 2)` interval matrix'e cevirir:

```text
[{start: 0, end: 10}, {start: 10, end: 25}]
                    |
                    v
[[0, 10], [10, 25]]
```

Evaluation bu gercek segment interval'lari uzerinden yapilir.

### 4.6 `segments_to_internal_boundaries()`

Segment'lerin outer edge'lerini degil, aradaki transition'lari cikarir. Ornegin `[0-10], [10-20], [20-30]` icin internal boundary listesi `[10, 20]` olur. `0` ve `30`, track edge'i oldugu icin detection boundary sayilmaz.

### 4.7 `boundaries_to_segments()`

Internal boundary'leri `0` ve duration ile birlestirip segment interval'lari olusturur. Boundary metadata varsa ilgili segment'lere confidence/source bilgisi aktarilabilir.

### 4.8 `enforce_min_segment_duration()`

Cok kisa segment'leri neighbor segment'lerle birlestirir. Bu post-processing iki amaca hizmet eder:

- Muziksel olarak anlamsiz micro-segment'leri azaltmak.
- Birbirine yakin yuksek novelty noktalarinin over-segmentation uretmesini engellemek.

### 4.9 `normalize_algorithm_result()`

Sistemdeki en merkezi compatibility helper'larindan biridir:

- Algorithm adini canonical hale getirir.
- Segment ve boundary alanlarini tamamlar.
- Structural ve semantic alanlari korur.
- Diagnostics'i tek yerde standardize eder.
- Failed result icin sahte boundary uretmez.

---

## 5. `custom_librosa` Pipeline'i

Dosyalar:

- `workers/segmenters/custom_worker.py`
- `workers/segmenters/segmentation_service.py`
- `workers/segmenters/multi_feature_fusion.py`

`custom_librosa`, projenin deterministic ve multi-feature segmentation pipeline'idir. `custom` adi input compatibility icin kabul edilir; canonical output adi `custom_librosa`'dir.

### 5.1 Pipeline ozeti

```text
Audio load
  -> active region detection
  -> Chroma-CENS + MFCC extraction
  -> temporal pooling and normalization
  -> self-similarity matrix
  -> SSM enhancement and novelty
  -> RMS/onset/chord/beat/lyrics candidates
  -> feature-level fusion
  -> onset/beat snapping
  -> segment construction
  -> structural clustering
  -> conservative semantic labeling
  -> full-track timestamp correction
```

### 5.2 Audio loading

#### `_find_ffmpeg()`

Sistemde kullanilabilir ffmpeg binary'sini arar. Container ve local ortam icin bilinen path'leri kontrol eder.

#### `_load_audio_ffmpeg(file_path, sr=22050)`

- ffmpeg ile audio'yu decode eder.
- Mono, `22050 Hz`, float32 PCM cikisi alir.
- Python seviyesinde frame-frame MP3 decode overhead'ini azaltir.
- Uzun parcalarda Librosa/audioread yoluna gore daha hizli olmasi hedeflenmistir.

#### `_load_audio_from_bytes(...)`

File path kullanilamayan durumlar icin Librosa ile byte stream'den fallback loading saglar.

### 5.3 Active region detection

#### `_detect_active_region(y, sr, ...)`

Track basindaki veya sonundaki sessizligi tum analizi bozmayacak sekilde ayirmaya calisir:

1. RMS enerji hesaplanir.
2. dB domain'ine cevrilir.
3. Gaussian smoothing uygulanir.
4. Dinamik threshold `P75(RMS_dB) - margin_db` olarak belirlenir.
5. Threshold ustundeki ilk ve son frame aktif bolge olarak alinir.
6. Aktif bolge cok kisaysa guvenli fallback olarak tum track kullanilir.

Bu crop nedeniyle uretilen boundary zamanlari gecici olarak active-region koordinatindadir. Pipeline sonunda `active_start` yeniden eklenerek full-track timeline'a donulur.

### 5.4 Feature extraction

#### `_median_pool(feat, pw)`

Raw feature frame'lerini temporal bloklarda median ile havuzlar. Median, tekil gurultu ve transient'lere mean'den daha dayaniklidir.

#### `_extract_downsampled_features(...)`

Iki temel representation uretir:

- `Chroma-CENS`: 12 pitch class uzerinden harmonik yapi ve tekrar bilgisi.
- `MFCC`: timbral ve enstrumantasyon karakteri.

Islem:

1. Chroma ve MFCC raw hop rate'te cikarilir.
2. MFCC0 enerji etkisini fazla tasidigi icin bir ekstra coefficient istenir ve ilk coefficient atilir.
3. Feature'lar target FPS'e median-pool edilir.
4. Chroma frame'leri L2 normalize edilir.
5. MFCC coefficient'lari z-score standardize edilir ve L2 normalize edilir.
6. Uniform `frame_times` olusturulur.

Uniform time grid, boundary timestamp'inin beat uzunluguna bagli degisken resolution'a sahip olmasini engeller.

### 5.5 Self-Similarity Matrix

#### `_compute_raw_ssm(feat)`

L2-normalized feature frame'leri arasinda cosine similarity benzeri dot product matrisi hesaplar:

```text
S[n,m] = feature(n) dot feature(m)
```

Matrisin diagonal disindaki yuksek bloklari, parcada birbirine benzeyen zaman bolgelerini gosterir.

#### `_compute_ti_chroma_ssm(chroma)`

Transposition-invariant chroma SSM uretir. Chroma vektoru 12 olasi pitch shift ile roll edilir ve her frame ciftinde maksimum benzerlik alinir:

```text
S_TI[n,m] = max over c in 0..11 of <roll_c(chroma_n), chroma_m>
```

Boylece ayni motif farkli tona transpose edilmisse de tekrar olarak gorulebilir.

#### `_build_combined_ssm(...)`

- Harmonik tekrar icin chroma SSM.
- Timbre degisimi icin MFCC SSM.
- MFCC aktifse ikisini esit oranda blend eder.

### 5.6 SSM enhancement

#### `_diagonal_smooth_theta(...)`

SSM uzerindeki diagonal tekrar yollarini belirli tempo oraninda smooth eder. Forward ve backward iki yon kullanilmasi, tek yonlu smoothing'in boundary'yi sistematik olarak gec kaydirmasini azaltir.

#### `_smooth_ssm(...)`

Birden fazla tempo ratio icin diagonal smoothing yapar:

```text
[0.66, 0.81, 1.0, 1.22, 1.50]
```

Tum versiyonlarin cell-wise maximum'u alinir. Amac, benzer muzik materyali farkli tempoda tekrarlandiginda diagonal path'i korumaktir.

#### `_threshold_ssm(...)`

Global relative threshold ile SSM'nin en guclu `rho` oranindaki hucrelerini tutar. Varsayilan `rho=0.20`, matrisin en anlamli benzerliklerini one cikarip zayif background similarity'yi bastirir.

### 5.7 Novelty curves

#### `_compute_novelty_ssm(S, L, var)`

Gaussian checkerboard kernel'i SSM diagonal'i boyunca kaydirir. Bir boundary etrafinda:

- Boundary oncesi frame'ler kendi icinde benzerdir.
- Boundary sonrasi frame'ler kendi icinde benzerdir.
- Iki taraf birbirinden farklidir.

Checkerboard kernel tam bu blok yapisina yuksek cevap verir. Novelty curve'deki peak, olasi segment gecisidir.

#### `_structure_feature_novelty(...)`

Local checkerboard novelty'ye ek olarak global repetition context change'i olcer. Circular time-lag representation'daki ard arda column'larin L2 distance'ini kullanir. Benzer instrumentation'da verse'ten chorus'a transition gibi local energy change'i zayif ama repetition context'i farkli boundary'leri yakalamaya yardim eder.

### 5.8 Diger candidate kaynaklari

`multi_feature_fusion.py` her kaynagi ayni candidate semasina cevirir:

```json
{"time": 42.3, "source": "rms", "confidence": 0.71}
```

#### `find_boundaries(...)`

- Novelty curve'u opsiyonel Gaussian filter ile smooth eder.
- `scipy.signal.find_peaks` ile peak bulur.
- Peak'ler arasinda minimum segment duration kadar mesafe ister.
- Track edge'lerine cok yakin peak'leri eler.
- Frame index'i saniyeye cevirir.

#### `normalise_curve(values)`

NaN ve infinity degerlerini temizler, minimumu sifira tasir ve curve'u `0-1` araligina normalize eder.

#### `curve_confidence(curve, frame_times, t)`

Bir candidate timestamp'ine en yakin curve frame'indeki normalize novelty degerini confidence olarak kullanir.

#### `rms_boundary_candidates(...)`

- RMS energy curve hesaplar.
- dB'e cevirir.
- Zamansal turevin mutlak degerini alir.
- Buyuk energy change'lerini boundary candidate yapar.

#### `onset_boundary_candidates(...)`

- Onset strength envelope hesaplar.
- Smooth ve normalize eder.
- Yogun atak/transient yapisi degisen bolgeleri candidate yapar.

#### `chord_proxy_boundary_candidates(...)`

Chroma frame'lerini yaklasik `+/-0.5s` etrafinda karsilastirir. Similarity dususu armonik degisimin proxy'sidir. CENS cok smooth oldugu icin direkt adjacent frame farki yerine centered lag kullanilir.

#### `tempo_and_beats(...)`

Tempo ve beat zamanlarini cikarir. Beat bilgisi tek basina ana structural detector degildir; phrase candidate ve final snapping icin yardimci zaman grid'i saglar.

#### `beat_phrase_boundary_candidates(...)`

Beat sequence uzerinden periyodik phrase-level candidate'lar uretir. Agirligi dusuktur; amaci boundary'yi ritmik olarak anlamli bolgeye desteklemektir.

#### `lyrics_boundary_candidates(...)`

Optional timed lyric line'larini candidate'a cevirir. Lyrics yoksa pipeline tamamen deterministic audio feature'lariyla devam eder. Lyrics candidate confidence'i kontrollu tutulur; lyric line change otomatik olarak structural boundary kabul edilmez.

---

## 6. Feature-Level Fusion - Tam Teknik Aciklama

Dosya: `workers/segmenters/multi_feature_fusion.py`

Feature-level fusion, `custom_librosa` pipeline'inin icindedir. Farkli algoritma worker'larini birlestirmez. Ayni audio analizinden elde edilen farkli feature kaynaklarinin candidate'larini birlestirir.

### 6.1 Varsayilan source agirliklari

```text
ssm          = 0.42
chord_proxy  = 0.18
onset_flux   = 0.06
rms          = 0.06
lyrics       = 0.10
beat         = 0.02
```

Bu degerler normalize edilir. Kullanici `feature_weights` ile override edebilir. `spectral_flux_weight`, geriye uyumluluk veya kolay tuning icin onset flux agirligini ayri degistirebilir.

SSM en yuksek agirligi alir; cunku hedef frame-level degisim degil, section-level yapi degisimidir. RMS veya onset tek basina bir davul fill'i ya da loudness jump'i yapisal boundary sanabilir.

### 6.2 `normalise_feature_weights(...)`

1. Default agirliklari kopyalar.
2. Bilinen source key'leri icin user override uygular.
3. Negatif agirligi sifira clamp eder.
4. Toplami sifirsa default'a doner.
5. Tum agirliklari toplamlari `1.0` olacak sekilde normalize eder.

### 6.3 `fuse_feature_candidates(...)`

Methodun adimlari:

#### Adim 1 - Edge filtering

Candidate ancak su aralikta ise tutulur:

```text
min_segment_duration * 0.5 <= time <= duration - min_segment_duration * 0.5
```

Bu, track baslangic/bitis edge'ine cok yakin anlamsiz candidate'lari azaltir.

#### Adim 2 - Temporal grouping

Candidate'lar zamana gore siralanir. Yeni candidate, son grubun ortalamasina `merge_window_s` kadar yakinsa ayni gruba girer. Varsayilan pencere `2.75s`'dir.

Ornek:

```text
SSM   : 30.8
RMS   : 31.5
Onset : 31.1

=> ayni yapisal gecisin uc farkli tahmini olarak tek grup
```

#### Adim 3 - Her source'tan tek oy

Ayni source ayni temporal grup icinde birden fazla candidate uretebilir. Fusion sadece confidence'i en yuksek adayi alir. Bu kural tek bir feature'in cok sayida peak ureterek oylamayi domine etmesini engeller.

#### Adim 4 - Weighted score

Her grubun temel skoru:

```text
weighted_sum = sum(source_weight * candidate_confidence)
```

Ardindan source cesitliligine kucuk bir agreement bonus eklenir:

```text
bonus = min(0.15, 0.035 * (source_count - 1))
score = min(1.0, weighted_sum + bonus)
```

Bu bonus, birbirinden bagimsiz birden fazla feature'in ayni zaman bolgesini gostermesini odullendirir.

#### Adim 5 - Acceptance

Grup su kosullardan biriyle kabul edilir:

- `score >= threshold`, varsayilan threshold `0.30`.
- SSM candidate'i varsa ve confidence'i en az `0.5` ise.

Ikinci kural bilincli bir exception'dir. SSM ana structural signal oldugu icin, diger low-level feature'lar desteklemese bile guclu bir SSM boundary kaybolmaz.

#### Adim 6 - Anchor secimi

Kabul edilen grubun hangi timestamp'e yerlestirilecegi `_choose_boundary_anchor()` ile belirlenir. Burada source agirligi ve confidence dikkate alinir; amac grup ortalamasini korurken zayif source'un timestamp'i kaydirmasini engellemektir.

#### Adim 7 - Minimum distance

Kabul edilmis boundary'ler zamana gore siralanir. Iki boundary arasinda `min_seg_dur` yoksa confidence'i daha yuksek olan tutulur.

#### Adim 8 - Maximum boundary count

`max_boundaries` verilmisse en guclu boundary'ler tutulur ve tekrar kronolojik siraya konur.

### 6.4 `snap_fused_boundaries(...)`

SSM structural bolgeyi iyi bulabilir fakat tam timestamp'i frame resolution veya smoothing nedeniyle biraz kayabilir. Snapping adimi fused boundary'yi yakin bir guclu onset veya beat'e tasir.

Buradaki prensip:

- Fusion, gecisin hangi bolgede oldugunu belirler.
- Onset/beat, timestamp'i daha hassas bir muzik olayina hizalar.
- Snap window limited tutulur; dogru region'daki boundary uzak bir transient'e tasinmaz.

### 6.5 Feature-level fusion fallback'i

Fusion threshold'undan hic candidate gecmez ama SSM novelty mevcutsa en guclu SSM peak'lerinden fallback boundary uretilir. Bu, secondary feature'larin zayif oldugu parcalarda pipeline'in tamamen bos sonuc vermesini engeller.

### 6.6 Sunumda kullanilacak kisa ayrim

> Feature-level fusion, ayni custom algoritmanin icinde SSM, chord, RMS, onset, beat ve lyrics sinyallerini birlestirir. Buradaki oy verenler algoritmalar degil, feature kaynaklaridir.

---

## 7. MSAF Algoritmalari

Dosya: `workers/segmenters/msaf_worker.py`

MSAF worker ayni worker sinifi uzerinden `foote`, `cnmf` veya `scluster` calistirir. `boundaries_id`, hangi MSAF boundary detector'in kullanilacagini belirler.

### 7.1 Foote

Foote yaklasimi self-similarity representation uzerindeki novelty mantigina dayanir. Checkerboard benzeri lokal degisim yapisini kullanarak iki yapi bolgesi arasindaki gecisleri arar.

Sunum mesaji:

> Foote lokal struktur degisimlerine odaklanan klasik ve aciklanabilir bir baseline'dir. Custom pipeline'imiz daha fazla feature birlestirirken Foote daha sade bir ikinci gorus saglar.

### 7.2 CNMF

CNMF, convex non-negative matrix factorization tabanli structure analysis method'udur. Music representation'indaki recurring latent pattern'leri factorize ederek section'lari cikarir.

Sunum mesaji:

> CNMF'in hata profili novelty peak detector'dan farklidir. Bu fark, fusion icin degerlidir; cunku ayni sinyali kopyalamak yerine farkli matematiksel varsayima sahip bir oy elde ederiz.

### 7.3 SCluster

SCluster, spectral clustering ile music structure'i section'lara ayirir. Similarity graph veya affinity structure'indaki global cluster organization'dan yararlanir.

Sunum mesaji:

> SCluster local change'den cok global similarity structure'ini kullanir. Recurring section'larin global organization'ini yakalamasi nedeniyle algorithm-level fusion'da ikinci en yuksek default weight'i alir.

### 7.4 MSAF worker normalization

MSAF raw output'u dogrudan frontend'e verilmez. Worker:

1. `boundaries_id` degerini validate eder.
2. MSAF'i file path uzerinde calistirir.
3. Boundary time'larini numeric ve duration-safe hale getirir.
4. Baslangic ve bitis edge'lerini normalize eder.
5. Raw label sayisi segment sayisina uymuyorsa pad/trim yapar.
6. MSAF label'larini otomatik Verse/Chorus kabul etmez.
7. Structural ve semantic katmanlari ayirir.
8. Diagnostics ile raw ve normalized count'lari raporlar.
9. Sonucu ortak `AlgorithmResult` semasina cevirir.

Bu adim algorithm-level fusion'in Foote/CNMF/SCluster sonucunu ayni contract ile okuyabilmesini saglar.

---

## 8. Algorithm-Level Fusion - Tam Teknik Aciklama

Dosyalar:

- `backend/services/segmentation_orchestrator.py`
- `backend/services/result_listener.py`
- `workers/segmenters/fusion_worker.py`
- `workers/segmenters/fusion_service.py`

Algorithm-level fusion, projenin en onemli ozelligidir. Bu servis audio'yu bastan analiz eden besinci bir detector degildir. Dort base algoritmanin tamamlanmis boundary tahminlerini post-processing asamasinda birlestirir.

### 8.1 Neden ayri bir fusion seviyesi var?

Her base algoritmanin farkli guclu ve zayif yonu vardir:

- `custom_librosa`: cok feature'li, domain'e ozel ve aciklanabilir.
- `foote`: lokal novelty degisimleri.
- `cnmf`: latent tekrar pattern'leri.
- `scluster`: global spectral clustering yapisi.

Bir boundary yalnizca tek algoritmada goruluyorsa false positive olabilir. Birden fazla farkli algoritma yakin zamanlarda boundary buluyorsa guven artar. Ancak algoritmalar esit guvenilir kabul edilmez; bu nedenle basit majority vote yerine weighted voting kullanilir.

### 8.2 Varsayilan algoritma agirliklari

```text
custom_librosa = 0.35
scluster       = 0.30
cnmf           = 0.20
foote          = 0.15
```

Toplam `1.0`'dir. Kullanici `params.fusion.weights` ile override edebilir. Negatif degerler sifira clamp edilir; bilinmeyen algoritma key'leri yok sayilir.

Bu agirliklar mutlak dogruluk garantisi degildir. Sistemin mevcut onceliklendirmesidir ve batch evaluation ile yeniden kalibre edilebilir.

### 8.3 Fusion request lifecycle

Kullanici su request'i gonderebilir:

```json
{
  "song_id": "1013",
  "algorithms": ["fusion"],
  "params": {
    "fusion": {
      "merge_window_seconds": 2.5,
      "threshold": 0.45,
      "required_vote_count": 2,
      "anchor_strategy": "custom_snap"
    }
  }
}
```

Yalnizca `fusion` yazilsa bile orchestrator dort base algoritmayi otomatik dispatch eder. Expected result listesi base algoritmalarla fusion'i birlikte icerir.

Fusion worker ilk anda calistirilmaz. Cunku payload icinde `algorithm_results` henuz yoktur.

### 8.4 `_maybe_dispatch_fusion()`

Bu method ResultListener icinde fusion'in ne zaman baslayacagini belirler.

#### Guard kosullari

Fusion dispatch edilmez eger:

- Task fusion beklemiyorsa.
- Fusion sonucu zaten geldiyse.
- `fusion__dispatched` flag'i daha once set edildiyse.

Bu kontroller duplicate fusion task olusmasini engeller.

#### Base result toplama

Her baseline algoritma icin:

- `algorithm__result` varsa resolved kabul edilir.
- Status `completed` ve segment listesi doluysa fusion input'una eklenir.
- Failed veya bos result ise failed listesine girer.
- Legacy `results[algorithm]` segment listesi varsa normalized result yeniden olusturulur.

Bu ayrim onemlidir:

- `resolved`: worker cevap verdi.
- `successful`: fusion'da kullanilabilir segment uretti.

#### Dispatch condition

Fusion ancak dort baseline'in tamami resolved oldugunda dispatch edilir:

```text
resolved algorithms = custom_librosa + foote + cnmf + scluster
```

Burada `resolved`, worker'in completed veya failed result publish etmis olmasidir. Dort result da gelmeden fusion baslatilmaz. Tum baseline'lar resolved olduktan sonra en az iki successful result varsa available result'larla fusion calisir.

#### Yetersiz sonuc

Tum gerekli durumlar resolved oldugu halde ikiden az basarili base sonuc varsa fusion worker'a gereksiz task gonderilmez. Listener dogrudan failed normalized fusion sonucu olusturur:

```text
Fusion requires at least two successful base algorithm results.
```

#### Fusion payload

Fusion worker'a su baglam gonderilir:

- `task_id`
- `algorithm=fusion`
- `worker_type=fusion`
- Basarili normalized `algorithm_results`
- Fusion params
- Failed/missing algoritmalar
- Slow/pending algoritmalar
- Mumkunse audio source bilgisi

Audio source, fusion boundary voting icin zorunlu degildir; labeling descriptor'lari icin kullanilabilir.

### 8.5 `FusionWorker.process_task()`

Fusion worker ince bir adapter'dir:

1. Payload'dan `algorithm_results` alir.
2. Params ve audio path'i hazirlar.
3. `fuse_algorithm_results()` servisini cagirir.
4. Normalized fusion sonucunu BaseWorker lifecycle'iyle publish eder.

Asil domain logic `fusion_service.py` icindedir. Bu ayrim worker messaging kodu ile fusion matematigini birbirinden ayirir ve unit test yazmayi kolaylastirir.

### 8.6 `_duration_from_results()`

Fusion track suresini su kaynaklardan cikarir:

- Result icindeki `duration_seconds`.
- Segmentlerin en buyuk `end` degeri.

Mevcut degerlerin maksimumunu alir. Boylece bir worker duration yazmadiysa diger result veya segment end'i fallback olur.

### 8.7 `_internal_boundaries_from_result()`

Bir algoritmanin fusion'a verecegi oylar cikarilir:

1. Full result icinde boundary listesi varsa onu kullanir.
2. `time <= 0.5s` olan start-edge boundary'leri eler.
3. Duration biliniyorsa `duration - 0.5s` sonrasindaki end-edge boundary'leri eler.
4. Confidence'i korur.
5. Boundary listesi yoksa segment interval'larindan internal boundary reconstruct eder ve confidence'i `1.0` kabul eder.

Track baslangici ve bitisi zaten segment edge'idir; algoritma oylamasina dahil edilmez.

### 8.8 `_group_votes(votes, merge_window_seconds)`

Tum algoritma oylarini timestamp'e gore siralar. Her yeni vote icin son grubun mevcut mean timestamp'i hesaplanir. Vote bu merkeze merge window kadar yakinsa ayni gruba eklenir.

Varsayilan:

```text
merge_window_seconds = 2.5
```

Ornek:

```text
custom_librosa: 60.2
scluster      : 61.0
cnmf          : 59.7
foote         : 74.3

Ilk uc oy ayni boundary group olur.
74.3 ayri group olur.
```

Neden tam timestamp equality kullanilmiyor: Farkli frame grid'leri, smoothing ve model varsayimlari ayni muziksel gecisi birkac yuz milisaniye veya saniye farkla tahmin edebilir.

### 8.9 Grup icinde algoritma deduplication

Ayni algoritma ayni group icinde birden fazla boundary uretebilir. `best_by_algorithm`, sadece en yuksek confidence'li vote'u tutar.

Bu kural cok onemlidir:

> Bir algoritma bir grup icinde kac peak uretirse uretsin tek oy hakkina sahiptir.

Aksi halde over-segment eden bir algoritma weighted score'u yapay olarak yukseltebilir.

### 8.10 Weighted score

Bir grubun skoru:

```text
score = sum(weight_algorithm * confidence_algorithm)
```

Ornek:

```text
custom_librosa confidence = 0.90, weight = 0.35
scluster       confidence = 0.80, weight = 0.30

score = 0.35*0.90 + 0.30*0.80
      = 0.315 + 0.240
      = 0.555
```

Varsayilan threshold `0.45` oldugu icin bu grup score ile kabul edilir.

### 8.11 Acceptance kuralinin iki yolu

Bir boundary group su kosullardan biriyle kabul edilir:

```text
score >= threshold
OR
unique_source_count >= required_vote_count
```

Varsayilanlar:

```text
threshold = 0.45
required_vote_count = 2
```

Bu OR kurali neden var?

- Weighted threshold, yuksek guvenli ve yuksek agirlikli algoritmalari odullendirir.
- Vote count, iki farkli algoritmanin ayni bolgeyi gostermesini korur; confidence calibration algoritmalar arasinda mukemmel olmasa da consensus kaybolmaz.

Trade-off: Dusuk confidence'li iki algoritma da `required_vote_count=2` ile boundary kabul ettirebilir. Bu nedenle parametre evaluation sonucuna gore ayarlanabilir.

### 8.12 `_choose_fused_time()`

Kabul edilen grubun final timestamp'ini belirler.

#### `weighted_mean`

```text
fused_time = sum(weight_i * confidence_i * time_i)
             / sum(weight_i * confidence_i)
```

Agirlikli paydaya sahip oldugu icin daha guvenilir algoritma final timestamp'i kendine daha fazla ceker.

#### `custom_snap`

Grup icinde `custom_librosa` vote'u varsa en guclu custom vote timestamp'i anchor olarak secilir. Custom yoksa weighted mean'e duser.

Bu stratejinin gerekcesi custom pipeline'in final candidate'lari onset/beat'e snap etmis olabilmesi ve timestamp precision'inin daha yuksek olmasidir. Base algoritma consensus'u boundary'nin varligini; custom timestamp'i hassas konumu belirler.

Not: Kodda `FusionSegmentationParams` schema default'u `weighted_mean`, servis params gelmezse internal default `custom_snap` kullanir. Sunumda aktif request'in hangi degeri gonderdigini goster; bu iki default farki ileride tek degerde birlestirilmesi gereken bir configuration tutarsizligidir.

### 8.13 Accepted boundary yapisi

Her kabul edilen boundary su aciklanabilirlik bilgisini tasir:

```json
{
  "time": 60.2,
  "confidence": 0.555,
  "source": "algorithm_fusion",
  "sources": ["custom_librosa", "scluster"],
  "metadata": {
    "score": 0.555,
    "raw_times": [
      {"algorithm": "custom_librosa", "time": 60.2, "confidence": 0.9},
      {"algorithm": "scluster", "time": 61.0, "confidence": 0.8}
    ]
  }
}
```

Bu veri, "fusion bu boundary'yi neden koydu?" sorusunu cevaplar.

### 8.14 `_dedupe_and_enforce_boundaries()`

Accepted boundary'lere final zaman kurallari uygulanir:

- Track basindan `min_segment_duration` kadar yakin boundary atilir.
- Track sonundan `min_segment_duration` kadar yakin boundary atilir.
- Iki accepted boundary arasinda minimum duration yoksa confidence'i daha yuksek olan tutulur.

Varsayilan minimum segment duration `8.0s`'dir.

Bu adim voting kabul etse bile final segmentasyonun mikro segmentlere ayrilmasini engeller.

### 8.15 Segment olusturma ve ikinci minimum-duration kontrolu

Accepted boundary'ler `boundaries_to_segments()` ile araliklara cevrilir. Ardindan `enforce_min_segment_duration()` segment seviyesinde de calisir.

Iki kontrolun farki:

- Boundary-level kontrol, cok yakin boundary point'lerinden birini secer.
- Segment-level kontrol, olusan gercek segment interval'larini gerekirse neighbor segment ile birlestirir.

### 8.16 Leading silence korumasi

#### `_collect_leading_silence_ends()`

Base algoritma segmentlerinde track basinda `Silence` semantic label'i varsa end timestamp'lerini toplar. `2s` icindeki yakin zamanlari cluster edip median ile tek degere indirger.

#### `_reinsert_silence_segments()`

Fusion min-duration veya voting asamasinda leading silence boundary'si kaybolmussa final segmenti bu noktada tekrar boler ve ilk parcayi `Silence` olarak isaretler.

Bu ozel kural, muzikal content boundary voting'inin track basindaki sessizligi yutmamasini saglar.

### 8.17 Structural ve semantic labeling

Fusion segmentleri `apply_two_layer_labels()` ile etiketlenir:

- Ilk katman: tekrar benzerligine dayali `A/B/C` structural labels.
- Ikinci katman: opsiyonel, konservatif semantic labels.

Fusion audio path alabiliyorsa descriptor extraction daha zengin olur. Audio yoksa mevcut label veya deterministic fallback kullanilir.

### 8.18 Fusion diagnostics

Final result icinde su diagnostics alanlari bulunur:

- `weights`
- `merge_window_seconds`
- `threshold`
- `min_segment_duration_seconds`
- `anchor_strategy`
- `required_vote_count`
- `input_algorithms`
- `available_algorithms`
- `failed_or_missing_algorithms`
- `boundary_groups`

Her `boundary_group` icinde:

- `fused_time`
- `score`
- `sources`
- Tum raw timestamp ve confidence'lar
- `accepted` karari

Sunumda diagnostics'i mutlaka goster. Fusion'in kara kutu olmadigini kanitlayan ana ciktidir.

### 8.19 Bilinen lifecycle siniri

BaseWorker yakaladigi exception icin failed result publish eder. Ancak process aniden oldurulurse, container tamamen cokmeden once result publish edemezse veya mesaj hic teslim edilmezse listener o algoritmayi `resolved` olarak goremeyebilir.

Fusion tum baseline'lari bekledigi icin bir worker hic result veya failure payload publish etmezse `_maybe_dispatch_fusion()` readiness condition'ini saglayamaz ve task processing durumunda kalir. Bu, sunumda saklanmamasi gereken bilinen bir sinirdir.

Gelecek gelistirme:

- Periyodik task watchdog.
- Worker heartbeat veya lease.
- Task-level hard timeout.
- Dead-letter queue ve retry policy.

### 8.20 Feature-level ve algorithm-level fusion karsilastirmasi

| Ozellik | Feature-level fusion | Algorithm-level fusion |
|---|---|---|
| Konum | `custom_librosa` pipeline ici | Ayri fusion worker/service |
| Girdi | SSM, RMS, onset, chord, beat, lyrics candidate'lari | Tamamlanmis base algorithm result'lari |
| Oy veren | Feature source | Segmentasyon algoritmasi |
| Ana method | `fuse_feature_candidates()` | `fuse_algorithm_results()` |
| Default merge window | 2.75s | 2.5s |
| Default threshold | 0.30 | 0.45 |
| Ozel kabul | Guclu SSM tek basina gecebilir | Score veya gerekli algoritma oy sayisi |
| Timestamp | Feature anchor + onset/beat snapping | Weighted mean veya custom snap |
| Amac | Custom detector icinde sinyal cesitliligi | Bagimsiz detector'lar arasinda consensus |

Sunumda birebir kullanilabilecek cumle:

> Projede fusion kelimesi iki farkli seviyeyi ifade ediyor. Birincisinde tek algoritmanin farkli akustik gozlemlerini birlestiriyoruz. Ikincisinde ise farkli segmentasyon algoritmalarinin tamamlanmis kararlarini birlestiriyoruz. Bunlar ayni islem degil ve kodda da ayri servisler olarak tutuluyor.

---

## 9. Structural ve Semantic Labeling

Dosya: `shared/labeling.py`

### 9.1 Neden iki katman?

Boundary detection "section nerede degisti?" sorusunu cevaplar. Structural labeling "hangi segment'ler birbirine benziyor?" sorusunu cevaplar. Semantic labeling ise "bu segment muziksel olarak Chorus mu?" sorusunu cevaplamaya calisir.

Son soru daha guclu bir iddiadir. Bu nedenle sistem:

- Structural label'i ana ve stabil label olarak tutar.
- Semantic label'i confidence ve reason ile ayri tutar.
- Semantik kanit zayifsa `Unknown`, `Early`, `Middle` veya `Late` gibi daha muhafazakar sonuc verebilir.

### 9.2 `build_segment_descriptors()`

Her segment icin descriptor uretir:

- Chroma mean ve standard deviation.
- MFCC mean ve standard deviation.
- RMS mean ve standard deviation.
- Onset density.
- Normalize segment duration.

Bu descriptor segmentleri timbre, harmony, energy, rhythm ve duration acisindan karsilastirmaya yarar.

### 9.3 `_cluster_descriptors()`

- Descriptor'lari standardize eder.
- `k=2..6` arasinda Agglomerative Clustering dener.
- Silhouette score ile en uygun sonucu secer.
- Score'u label confidence'e donusturur.
- Dependency veya clustering sorunu olursa cosine-similarity tabanli deterministic fallback kullanir.

### 9.4 `assign_structural_labels()`

- Cluster ID'lerini frequency order'a gore `A/B/C...` harflerine map eder.
- `structural_label` ve backwards-compatible `label` alanlarini birlikte yazar.
- Cluster yoksa mevcut raw labels'i normalize eder.
- Hic bilgi yoksa segment sirasi bazli deterministic fallback verir.

`A` "Verse" demek degildir. En sik veya ilk structural cluster'i temsil eder.

### 9.5 `assign_semantic_labels()`

Heuristic kanitlar:

- Cok dusuk RMS: `Silence`.
- Ilk non-silence ve track'in ilk yuzde 20'sinde: `Intro`.
- Son non-silence ve son yuzde 25'te: `Outro`.
- Tekrar eden ve daha yuksek energy'li structural cluster: `Chorus`.
- Diger tekrar eden cluster: `Verse`.
- Ortada, unique ve yeterince uzun segment: `Bridge`.
- Kanit yetersizse position-based `Early/Middle/Late` veya `Unknown`.

Her semantic label icin `semantic_confidence` ve `semantic_reason` yazilir.

### 9.6 `apply_two_layer_labels()`

Descriptor'i bir kez olusturup iki katmanla paylasir. Boylece structural clustering ve semantic energy analizi ayni audio representation'ini kullanir.

---

## 10. Evaluation Sistemi

Dosyalar:

- `shared/evaluation_metrics.py`
- `backend/services/evaluation_service.py`
- `backend/api/evaluation.py`

### 10.1 Ground truth

SALAMI annotation'lari human-labeled reference segment interval'larini saglar. Sistem prediction'i `estimated`, human annotation ise `reference` olarak ele alinir.

### 10.2 Neden interval-based evaluation?

Reference ve estimated sonuclar once gercek segment interval'lerine cevrilir:

```text
reference: [[0, 15.2], [15.2, 42.0], ...]
estimated: [[0, 14.9], [14.9, 43.1], ...]
```

Ardindan `mir_eval.segment.detection(..., trim=True)` kullanilir. Eski ve kirilgan yaklasim olan starts/ends listesinden ad hoc boundary reconstruction yerine segment contract'i dogrudan kullanilir.

### 10.3 Tolerans

Bir estimated boundary reference boundary'ye belirlenen pencere kadar yakinsa match sayilir.

- `+/-0.5s`: strict timing precision.
- `+/-3.0s`: dogru structural region'i bulma.

Iki metrigi birlikte okumak gerekir:

- `F1@3.0` iyi, `F1@0.5` dusukse model dogru bolgeyi buluyor ama timestamp hassasiyeti zayif.
- Ikisi de dusukse boundary detection veya boundary sayisi problemli.

### 10.4 Precision

```text
Precision = TP / (TP + FP)
```

Tahmin edilen boundary'lerin ne kadari dogru? Dusuk precision genellikle over-segmentation belirtisidir.

### 10.5 Recall

```text
Recall = TP / (TP + FN)
```

Gercek boundary'lerin ne kadari bulundu? Dusuk recall genellikle under-segmentation belirtisidir.

### 10.6 F1

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

Precision ve recall'u dengeler. Cok boundary tahmin ederek recall'u yapay yukseltmeyi veya cok az boundary ile precision'i korumayi cezalandirir.

### 10.7 `compute_boundary_metrics_at_tolerance()`

- Segmentleri interval'e cevirir.
- Internal boundary count'larini hesaplar.
- mir_eval detection uygular.
- Precision, recall, F1 ve count bilgilerini dondurur.
- Gerekirse guvenli greedy matching fallback'i vardir.

### 10.8 `compute_multi_tolerance_metrics()`

Ayni reference/estimated sonucunu birden fazla toleransta olcer ve key'leri su formatta dondurur:

```text
precision_0_5, recall_0_5, f1_0_5
precision_3_0, recall_3_0, f1_3_0
```

Ayrica estimated/reference ratio ile over/under-segmentation egilimini raporlar.

### 10.9 Batch evaluation

Batch akisi:

1. MinIO'daki track ID'leri listelenir.
2. Lokal SALAMI anotasyonu olanlarla intersection alinir.
3. `max_tracks=0` ise tum mevcut dataset kullanilir.
4. Her track belirtilen concurrency ile task olarak dispatch edilir.
5. Secilen her algoritma icin ayri evaluation row'u olusturulur.
6. Fusion secildiyse dort baseline ve fusion birlikte raporlanir.
7. Algorithm bazinda average strict/lenient F1, precision, recall ve estimated/reference ratio ozetlenir.

Batch evaluation sayesinde fusion weight ve threshold'lari tek bir ornek yerine dataset geneli uzerinden tartisilabilir.

---

## 11. API ve Demo Ornekleri

### 11.1 Storage'dan tum base algoritmalar ve fusion

```bash
curl -X POST http://localhost:8000/segmentation/from-storage \
  -H "Content-Type: application/json" \
  -d '{
    "song_id": "1013",
    "algorithms": ["custom_librosa", "foote", "cnmf", "scluster", "fusion"],
    "params": {
      "fusion": {
        "merge_window_seconds": 2.5,
        "threshold": 0.45,
        "required_vote_count": 2,
        "anchor_strategy": "custom_snap"
      }
    }
  }'
```

### 11.2 Status kontrolu

```bash
curl http://localhost:8000/segmentation/status/TASK_ID
```

Kontrol edilecek alanlar:

- `status`
- Base algorithm segment listeleri
- `fusion`
- `fusion__result`
- `fusion__diagnostics`
- `fusion__boundaries`

### 11.3 Batch evaluation

```bash
curl -X POST http://localhost:8000/evaluation/batch \
  -H "Content-Type: application/json" \
  -d '{
    "max_tracks": 20,
    "algorithms": ["custom_librosa", "foote", "cnmf", "scluster", "fusion"],
    "tolerances": [0.5, 3.0],
    "concurrency": 3
  }'
```

### 11.4 Demo sirasinda anlatilacaklar

Request'i gonderirken:

> Fusion'i sectigimiz icin backend fusion worker'ini hemen calistirmiyor. Once dort base algoritmayi RabbitMQ uzerinden paralel dispatch ediyor.

Partial result gorurken:

> Burada task hala processing. Cunku custom ve Foote sonucu gelmis olsa da expected algoritmalarin tamami tamamlanmadi. Listener her sonucu ayri kaydediyor.

Fusion sonucu gelince:

> Bu boundary'nin sources field'inda hangi algorithm'lerin vote verdigini, raw_times field'inda original timestamp'leri ve score field'inda weighted confidence'i gorebiliyoruz.

Evaluation gosterirken:

> Strict F1 timestamp hassasiyetini, lenient F1 ise dogru structural bolgeyi bulup bulmadigimizi gosteriyor. Bu ikisini birlikte yorumluyoruz.

---

## 12. Testler ve Guvenilirlik

Dosya: `tests/test_segmentation_core.py`

Mevcut core testlerin kapsadigi davranislar:

- Segmentlerin gercek interval matrisine donusmesi.
- Internal boundary'lerin start/end edge'lerini dislamasi.
- Boundary normalization'in invalid zamanlari atmasi ve duplicate'leri temizlemesi.
- Boundary'den segment olusturma.
- `0.5s` ve `3.0s` evaluation key'leri.
- MSAF-style sonuc normalization'i.
- Failed result icin sahte boundary uretilmemesi.
- Structural/semantic label ayrimi.
- Algorithm-level weighted voting'in boundary group kabul etmesi.

Sunumda testleri su sekilde konumlandir:

> Testler yalnizca endpoint'in 200 donmesini kontrol etmiyor. Fusion ve evaluation'in temel invariants'larini kontrol ediyor: edge boundary dislama, timestamp normalization, multi-tolerance metric ve multi-algorithm vote acceptance.

---

## 13. Tasarim Kararlari ve Savunmalari

### Neden microservice/worker mimarisi?

Algoritmalar CPU ve memory bakimindan farkli maliyetlere sahiptir. Asenkron worker mimarisi backend'i bloke etmeden paralel calisma ve algoritma bazli scaling saglar.

### Neden tek fusion worker?

Fusion deterministik post-processing servisidir. Her feature veya algoritma icin yeni agent olusturmak gereksiz karmasiklik yaratirdi. Bir domain service ve ince worker adapter'i yeterlidir.

### Neden common schema?

MSAF, custom ve fusion farkli raw formatlar uretebilir. Common schema frontend, evaluation ve fusion'in algoritmaya ozel branching yapmasini azaltir.

### Neden structural ve semantic label ayri?

Benzer section'lari cluster etmek ile o section'a Chorus demek ayni evidence level'da degildir. Bu separation, sistemin bilmedigi seyi biliyormus gibi gostermesini engeller.

### Neden weighted voting?

Algoritmalarin ayni guvenilirlikte olmadigi varsayilir. Majority vote her algoritmayi esit kabul eder; weighted voting hem algoritma prior'ini hem de boundary confidence'i kullanir.

### Neden iki acceptance kosulu var?

Confidence degerleri algoritmalar arasinda tam calibrated olmayabilir. Weighted threshold kalite sinyali saglarken vote count consensus'u korur.

### Neden minimum segment duration?

Peak detector'lar kisa aralikta birden fazla degisim gorebilir. Muziksel section segmentasyonu icin saniyelik mikro segmentler genellikle over-segmentation'dir.

### Neden iki tolerans?

Tek tolerans modelin problemini gizleyebilir. `3s` iyi ama `0.5s` kotuyse detector semantic region'i buluyor; localization precision gelistirilmelidir.

---

## 14. Muhtemel Juri Sorulari ve Cevaplari

### "Fusion sonucunun her base algoritmadan daha iyi oldugunu garanti ediyor musunuz?"

Hayir. Fusion bir garanti degil, farkli hata profilleri arasinda consensus arayan bir ensemble stratejisidir. Basarisi SALAMI batch evaluation ile algoritma ve track bazinda olculur. Weight ve threshold'lar dataset sonucuna gore kalibre edilmelidir.

### "Neden custom algoritmaya en yuksek agirligi verdiniz?"

Custom pipeline yapisal SSM sinyalini farkli feature candidate'lariyla birlestiriyor ve final timestamp'i onset/beat'e snap ediyor. Bu nedenle mevcut konfigurasyonda en yuksek prior ona verildi. Ancak bu deger hard scientific constant degildir; evaluation ile tune edilebilir.

### "Iki algoritma dusuk confidence ile ayni boundary'yi bulursa neden kabul ediliyor?"

Default `required_vote_count=2`, confidence calibration farklarina karsi consensus'u korur. Daha konservatif davranis icin vote count artirilabilir veya acceptance kuralinin yalnizca threshold olmasi tercih edilebilir. Mevcut secim recall ile precision arasinda ayarlanabilir bir trade-off'tur.

### "Feature fusion ile algorithm fusion ayni sey degil mi?"

Hayir. Feature fusion tek `custom_librosa` calismasindaki SSM/RMS/onset/chord gibi gozlemleri birlestirir. Algorithm fusion bagimsiz segmenter'larin tamamlanmis boundary setlerini birlestirir.

### "MSAF label'larini neden Chorus olarak kullanmiyorsunuz?"

MSAF'in raw labels'i yapisal cluster veya algorithm-specific label olabilir. Otomatik olarak semantic section name kabul etmek guclu ve desteklenmeyen bir iddia olur. Bu nedenle raw/yapisal bilgi korunur, semantic layer ayri ve konservatif uygulanir.

### "Bir worker fail olursa ne oluyor?"

BaseWorker failed normalized result publish eder. Listener bunu resolved fakat basarisiz sonuc olarak isaretler. En az iki basarili base sonuc varsa fusion calisabilir. Ikiden azsa acik bir failed fusion sonucu uretilir.

### "Bir worker hic cevap vermezse ne oluyor?"

Fusion tum baseline result'larini bekledigi icin task processing durumunda kalir ve fusion dispatch edilmez. Normal exception durumunda BaseWorker failed result publish ederek bunu engeller. Worker result publish edemeden tamamen kaybolursa watchdog, retry veya hard timeout gerekir; bunlar gelecek gelistirme alanidir.

### "Boundary confidence'lari algoritmalar arasinda calibrated mi?"

Tam olarak degil. Sistem confidence ile algorithm prior'ini birlikte kullanir ve vote-count fallback'i bulundurur. Daha ileri calismada validation set uzerinde calibration veya learned stacking uygulanabilir.

### "Neden deep learning kullanmadiniz?"

Bu sistemin hedefi aciklanabilir, deterministik, modul ve baseline'larla karsilastirilabilir bir pipeline kurmaktir. Deep model ileride yeni bir worker olarak eklenebilir ve ortak schema sayesinde fusion'a katilabilir. Mevcut mimari bu genislemeye kapali degildir.

### "LLM boundary uretiyor mu?"

Ana deterministik pipeline ve algorithm fusion LLM'e bagli degildir. LLM opsiyonel ayri worker'dir ve baseline fusion setinin parcasi degildir. Grounded timestamp uretimi DSP/segmentasyon servislerinde tutulur.

### "Evaluation label dogrulugunu da olcuyor mu?"

Ana evaluation boundary detection'a odaklanir. Structural labeling ikincil, semantic naming ucuncul hedeftir. Semantik label evaluation ancak ayni label taxonomy'sine sahip guvenilir ground truth ile anlamlidir.

### "Neden `trim=True`?"

Track edge'lerinin boundary detection score'unu yapay olarak etkilemesini engellemek ve reference/estimated interval kapsam farklarini standart mir_eval davranisiyla ele almak icin kullanilir.

### "Neden 0.5 ve 3 saniye?"

`0.5s` hassas zamanlama, `3s` ise daha toleransli structural localization olcumudur. Ikisi birlikte detector'in boundary'yi kacirip kacirmadigini ve bulduysa ne kadar hassas yerlestirdigini ayirir.

---

## 15. Bilinen Sinirlamalar ve Gelecek Calismalar

Sunumda sinirlamalari saklamak yerine tasarim farkindaligi olarak anlat:

1. Algorithm confidence degerleri tam calibrated degildir.
2. Default fusion weights mevcut bir configuration'dir; daha genis validation ile tune edilmelidir.
3. Fusion tum baseline result'larini bekledigi icin sessizce kaybolan worker task'i bloklayabilir; periyodik watchdog henuz yoktur.
4. Semantic labels heuristic'tir ve boundary metrics kadar guvenilir kabul edilmez.
5. Farkli muzik turlerinde ayni minimum segment duration optimal olmayabilir.
6. `anchor_strategy` schema ve service default'lari su anda ayni degildir; configuration tek kaynaga indirgenmelidir.
7. En az iki algoritmanin ayni sistematik hatayi yapmasi consensus ile false positive kabul ettirebilir.
8. Learned ensemble veya genre-adaptive weights henuz yoktur.

Gelecek gelistirmeler:

- Dataset bazli automatic weight optimization.
- Confidence calibration.
- Genre veya track descriptor'a gore dynamic algorithm weights.
- Periodic task watchdog ve hard timeout.
- Dead-letter queue ve kontrollu retry.
- Boundary group'lari icin learned meta-classifier.
- Semantic labeling icin ayrica annotated taxonomy evaluation.
- Frontend'de algorithm vote timeline visualization.

---

## 16. Sunumda Kullanilacak Ana Cumleler

Acillis:

> Muzik segmentasyonunda temel problem bir parcaya isim vermek degil, yapisal degisimlerin gerceklestigi zaman noktalarini guvenilir bicimde bulmaktir.

Mimari:

> API hesaplamayi kendi icinde yapmiyor; task'i kaydediyor ve algoritmalari RabbitMQ uzerinden bagimsiz worker'lara dagitiyor.

Normalization:

> Farkli algorithm'lerin birlikte calisabilmesi icin once hepsini boundary, segment ve diagnostics iceren common result schema'ya donusturuyoruz.

Feature fusion:

> Tek bir akustik ozellik her gecisi goremedigi icin custom pipeline icinde harmonik, timbral, enerji, onset, ritim ve opsiyonel lyrics kanitlarini birlestiriyoruz.

Algorithm fusion:

> Daha sonra tek bir algorithm'e guvenmek yerine dort independent segmentation method'unun yakin timestamp'lerdeki vote'larini weighted voting ile birlestiriyoruz.

Aciklanabilirlik:

> Her fused boundary icin hangi algoritmalarin oy verdigini, ham timestamp'leri, confidence degerlerini ve kabul skorunu diagnostics icinde sakliyoruz.

Labeling:

> A/B/C yapisal benzerliktir; Verse/Chorus ise semantik yorumdur. Sistem bu iki iddiayi ayni alan icinde karistirmiyor.

Evaluation:

> Sonuclari tek bir ornek uzerinden yorumlamiyoruz; SALAMI anotasyonlariyla strict ve lenient toleranslarda precision, recall ve F1 olcuyoruz.

Kapanis:

> Projenin ana katkisi, coklu feature ve coklu algoritma kararlarini dagitik ama ortak bir veri modeli uzerinde, aciklanabilir iki seviyeli fusion ve tekrarlanabilir evaluation ile birlestirmesidir.

---

## 17. Son Kontrol Listesi

Sunumdan once:

- Docker servislerinin ayakta oldugunu kontrol et.
- Backend health endpoint'ini kontrol et.
- RabbitMQ, PostgreSQL ve gerekli worker container'larini kontrol et.
- Demo track'inin storage'da bulundugunu onceden dogrula.
- Ayni track icin bir yedek screenshot veya JSON result hazir tut.
- Fusion diagnostics'te accepted ve rejected boundary group ornekleri bul.
- `0.5s` ve `3.0s` metriklerini karistirmamak icin slaytta acik etiketle.
- `feature fusion` ve `algorithm fusion` terimlerini ilk kullanimda tanimla.
- Semantic label'lari ground truth gibi sunma.
- Fusion'in garanti degil ensemble stratejisi oldugunu acik soyle.
- Tum baseline'lari bekleme davranisini ve gelecek watchdog gelistirmesini bil.

Sunum sirasinda:

- Once problem, sonra mimari, sonra method ayrintisi anlat.
- Kod method isimlerini sadece ne yaptiklarini aciklarken kullan.
- Formulu verdikten sonra mutlaka sayisal ornek goster.
- Diagnostics ile aciklanabilirligi kanitla.
- Fusion'dan evaluation'a gecisi net kur: "Birlesimi yaptik, simdi gercekten ise yariyor mu olcuyoruz."

Sunum sonunda dinleyicinin su dort noktayi anlamis olmasi gerekir:

1. Uygulama dagitik ve asenkron bir segmentasyon platformudur.
2. `custom_librosa` kendi icinde multi-feature fusion yapar.
3. `fusion` worker'i bagimsiz algoritma sonuclarini weighted voting ile birlestirir.
4. Output'lar common schema'ya normalize edilip SALAMI ile multi-tolerance evaluation'a sokulur.
