# Eksiksiz Method İndeksi

Bu indeks dört dosyadaki her methodu hızlı tekrar için listeler. Ayrıntılı anlatım diğer bölümlerdedir.

## segmentation_service.py

| Method | Girdi → çıktı | Neden var? |
|---|---|---|
| _find_ffmpeg | ortam → binary yolu | Hızlı decoder seçmek |
| _load_audio_ffmpeg | dosya, sr → y, sr | Decode/resample/mono yapmak |
| _load_audio_from_bytes | bytes, sr → y, sr | Bellek girdisini yüklemek |
| _detect_active_region | waveform → start, end | Baş/son sessizliğini ayırmak |
| _median_pool | feature, pencere → küçük feature | Gürültü ve SSM maliyetini azaltmak |
| _extract_downsampled_features | waveform → chroma, MFCC, times, FPS | Ortak feature grid'i kurmak |
| _l2_norm | feature → normalize feature | Cosine hesabını iç çarpıma dönüştürmek |
| _compute_ti_chroma_ssm | chroma → SSM | Ton değişimine dayanıklı tekrar bulmak |
| _compute_raw_ssm | feature → SSM | Standart cosine benzerliği |
| _build_combined_ssm | chroma, MFCC → SSM | Armoni ve tınıyı birleştirmek |
| _diagonal_smooth_theta | SSM, L, theta → SSM | Tempo oranındaki diagonal yolu güçlendirmek |
| _smooth_ssm | raw SSM → smooth SSM | Çoklu tempo/yön desteği |
| _threshold_ssm | smooth SSM → sparse SSM | Zayıf benzerliği bastırmak |
| _compute_novelty_ssm | SSM, kernel → curve | Lokal blok değişimini ölçmek |
| _structure_feature_novelty | sparse SSM → curve | Global tekrar bağlamı değişimini ölçmek |
| _select_n_clusters | descriptor matrix → k | Silhouette ile küme sayısı seçmek |
| _ssm_segment_labels | SSM, spans → cluster IDs | Segment tekrar benzerliğini gruplamak |
| _segment_feature_vector | segment feature → vector | Mean/delta/std özeti üretmek |
| _enforce_min_segment_duration | segments → merged segments | Mikro segmentleri kaldırmak |
| _merge | iki komşu segment → bir segment | İçteki birleştirme işlemini tutarlı yapmak |
| _assign_section_types | segments → semantic segments | Eski çağrılar için label uyumluluğu |
| _cluster_and_label_segments | features, boundaries → segments | Segment kurma ve label akışını yürütmek |
| _boundary_context | segment uçları → confidence, sources | Yakın boundary metadata'yı segmente taşımak |
| _novelty_snr | curve → 0..1 | Eğrinin ne kadar tepeli olduğunu ölçmek |
| _compute_dynamic_weights | kaynak sinyalleri → weights, confidences | Şarkıya özel weight uyarlamak |
| _beat_regularity | beat times → 0..1 | Ritmik grid güvenini ölçmek |
| _lyrics_confidence | lyric candidates, süre → 0..1 | Lyric kanıt yoğunluğunu ölçmek |
| process_file_path | path, params → result | Worker'ın public girişini sağlamak |
| _analyze_content | audio/path, params → result | Stage 0–8 pipeline'ını yürütmek |
| _empty_result | dosya bilgisi → boş result | Aktif müzik yokken geçerli cevap vermek |

_l2_norm, _merge ve _boundary_context dışarıdan çağrılan API değildir; üst method içinde tanımlı küçük yardımcı closure'lardır. Yine de davranışın parçasıdır.

## multi_feature_fusion.py

| Method | Görevi |
|---|---|
| find_boundaries | Novelty peak'lerini zamanlara çevirir |
| normalise_curve | Eğriyi güvenli biçimde 0–1 yapar |
| curve_confidence | Zamanın yerel novelty değerini bulur |
| candidates_from_boundaries | Ortak candidate şeması üretir |
| rms_boundary_candidates | Enerji değişimi adayları |
| onset_boundary_candidates | Onset adayları ve snapping eğrisi |
| tempo_and_beats | BPM ve beat zamanları |
| beat_phrase_boundary_candidates | Phrase-grid adayları |
| _dedupe_candidates | Yakın adaylardan güçlüyü tutar |
| chord_proxy_boundary_candidates | Armonik değişim proxy'si |
| lyrics_boundary_candidates | Timed-lyrics adayları |
| normalise_feature_weights | Weight override ve toplam normalization |
| fuse_feature_candidates | Feature-level fusion |
| fuse_boundary_candidates | Geriye uyumlu alias |
| _choose_boundary_anchor | Final yapısal zamanı seçer |
| snap_fused_boundaries | Beat/onset'e hassas hizalama |

## msaf_worker.py

| Method | Görevi |
|---|---|
| MSAFWorker.__init__ | Environment'tan algoritma/queue kurmak |
| MSAFWorker.process_task | MSAF sonucunu ortak şemaya uyarlamak |
| _normalize_msaf_labels | Label listesini segment sayısına uydurmak |

## fusion_service.py

| Method | Görevi |
|---|---|
| _collect_leading_silence_ends | Algoritmaların baş sessizlik bitişini toplamak |
| _reinsert_silence_segments | Kaybolan baş sessizliğini geri bölmek |
| _duration_from_results | Ortak track süresini belirlemek |
| _internal_boundaries_from_result | Track edge'leri dışındaki oyları çıkarmak |
| _group_votes | Zamanca yakın algoritma oylarını gruplamak |
| _choose_fused_time | Custom snap veya weighted mean zamanı |
| fuse_algorithm_results | Algorithm-level fusion akışını yürütmek |
| _dedupe_and_enforce_boundaries | Duplicate ve süre kısıtını uygulamak |

## Her methodu öğrenme şablonu

Bir methodu çalışırken şu beş cümleyi tamamla:

1. Girdisi ...
2. Çıktısı ...
3. Çözdüğü problem ...
4. Bu method olmasa oluşacak hata ...
5. En önemli parametre/trade-off ...
