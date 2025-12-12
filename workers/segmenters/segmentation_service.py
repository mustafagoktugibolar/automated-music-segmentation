import io
from collections import Counter

import librosa
import numpy as np
from fastapi import UploadFile
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from shared.logger import get_ogger

logger = get_ogger()

# --- Tunable Parameters (Quality-Oriented) ---

# Base hop length for Chroma CQT. Affects the time resolution of the features.
CQT_HOP_LENGTH = 512

# Time downsampling factor for chroma features.
# 1  → no downsampling (highest quality, slower)
# 2+ → faster, lower time resolution
DOWNSAMPLE_FACTOR = 1

# Size of the kernel for novelty curve calculation (in seconds).
NOVELTY_KERNEL_SIZE_SECONDS = 3.0

# Maximum kernel block size used in checkerboard expansion.
# Effective kernel will be at most (2 * MAX_KERNEL_BLOCK_SIZE) x (2 * MAX_KERNEL_BLOCK_SIZE).
MAX_KERNEL_BLOCK_SIZE = 4

# Number of clusters to find for labeling (e.g., verse, chorus, bridge).
# Smaller = simpler structure; 4–5 is good for typical pop songs.
N_CLUSTERS = 4

# Minimum segment duration (seconds). Shorter segments will be merged with neighbors.
MIN_SEGMENT_DURATION_SECONDS = 5.0

# Parameters for detecting active music region (start/end of "real" music).
# - ACTIVE_MARGIN_DB: how far below typical loudness we still consider "active"
# - MIN_ACTIVE_REGION_SECONDS: ignore tiny blips, need at least this much active audio
ACTIVE_MARGIN_DB = 20.0
MIN_ACTIVE_REGION_SECONDS = 3.0


# -------------------------------------------------------------------------
# Helper: detect active (non-silent) region based on RMS energy
# -------------------------------------------------------------------------

def _detect_active_region(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    margin_db: float = ACTIVE_MARGIN_DB,
    min_region_s: float = MIN_ACTIVE_REGION_SECONDS,
) -> tuple[float, float]:
    """
    Detects the approximate active music region [start_time, end_time] in seconds,
    based on frame-wise RMS energy.

    This is ADAPTIVE per track:
      - We compute RMS over frames,
      - Convert to dB and smooth,
      - Use a percentile-based threshold to decide what counts as 'music'.

    Returns:
      (active_start_s, active_end_s)
    If no significant active region is found, returns (0.0, total_duration).
    """
    if y.size == 0:
        return 0.0, 0.0

    # Frame-wise RMS with the same hop_length as chroma/STFT.
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]  # shape: (n_frames,)

    # Convert to dB relative to max RMS.
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Smooth to avoid spiky behavior.
    if rms_db.size > 3:
        rms_db = gaussian_filter1d(rms_db, sigma=2.0)

    # Dynamic threshold:
    # Take a "typical loud" level as the 75th percentile,
    # then allow margin_db below it as "still active".
    p75 = np.percentile(rms_db, 75)
    threshold = p75 - margin_db

    # Active frames: where energy is above threshold.
    active_mask = rms_db > threshold

    if not np.any(active_mask):
        # Could not find a clearly active region; fallback to full track.
        total_duration = librosa.get_duration(y=y, sr=sr)
        return 0.0, total_duration

    # Find first and last active frame, but enforce some continuity.
    active_indices = np.where(active_mask)[0]
    first_idx = int(active_indices[0])
    last_idx = int(active_indices[-1])

    # Convert frame indices to time.
    active_start_s = librosa.frames_to_time(first_idx, sr=sr, hop_length=hop_length)
    active_end_s = librosa.frames_to_time(last_idx, sr=sr, hop_length=hop_length)

    # Enforce a minimum region length; if too small, fallback to full track.
    if (active_end_s - active_start_s) < min_region_s:
        total_duration = librosa.get_duration(y=y, sr=sr)
        return 0.0, total_duration

    return active_start_s, active_end_s


# -------------------------------------------------------------------------
# Feature extraction & core MIR pieces
# -------------------------------------------------------------------------

def _load_audio_from_bytes(content: bytes, sr: int | None = None) -> tuple[np.ndarray, int]:
    """
    Loads an audio waveform from an in-memory byte buffer.
    """
    try:
        audio_stream = io.BytesIO(content)
        y, sr = librosa.load(audio_stream, sr=sr)
        return y, sr
    except Exception as e:
        logger.error("Failed to load audio from bytes.", exc_info=e)
        raise


def _extract_chroma_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extracts Chroma CQT features from an audio waveform.
    Uses harmonic separation for more stable chroma.
    """
    y_harmonic, _ = librosa.effects.hpss(y)

    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic,
        sr=sr,
        hop_length=CQT_HOP_LENGTH,
    )
    return chroma


def _compute_ssm(features: np.ndarray) -> np.ndarray:
    """
    Computes a self-similarity matrix from a feature matrix.
    Features should be shape (n_features, n_frames).
    """
    features_transposed = features.T
    ssm = cosine_similarity(features_transposed)
    return ssm


def _compute_novelty_curve(ssm: np.ndarray, kernel_size_frames: int = 1) -> np.ndarray:
    """
    Computes a novelty curve from a self-similarity matrix using a checkerboard kernel.

    NOTE: To keep performance reasonable, the kernel size is clamped and kept small.
    """
    kernel = np.array(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
        ],
        dtype=float,
    )

    block_size = max(1, min(kernel_size_frames, MAX_KERNEL_BLOCK_SIZE))

    if block_size > 1:
        kernel = np.kron(kernel, np.ones((block_size, block_size), dtype=float))

    novelty_2d = correlate2d(ssm, kernel, mode="same", boundary="symm")

    novelty_curve = np.diag(novelty_2d)
    novelty_curve = np.maximum(novelty_curve, 0.0)

    max_val = float(np.max(novelty_curve)) if novelty_curve.size > 0 else 1.0
    if max_val > 0:
        novelty_curve = novelty_curve / max_val

    # Smooth to reduce spurious peaks.
    if novelty_curve.size > 3:
        novelty_curve = gaussian_filter1d(novelty_curve, sigma=2.0)

    return novelty_curve


def _find_boundaries(
    novelty_curve: np.ndarray,
    sr: int,
    hop_length: int,
    min_segment_duration_s: float,
) -> list[float]:
    """
    Finds boundary timestamps from a novelty curve by picking peaks.

    Peak picking parameters are derived from the desired minimum segment duration,
    so we don't get crazy-dense boundaries.
    """
    if novelty_curve.size == 0:
        return []

    frames_per_second = sr / hop_length
    min_frames = max(1, int(min_segment_duration_s * frames_per_second))

    pre_max = max(1, min_frames // 4)
    post_max = pre_max
    pre_avg = max(1, min_frames // 2)
    post_avg = pre_avg
    wait = max(1, min_frames // 2)
    delta = 0.25  # slightly higher to ignore small bumps

    peaks = librosa.util.peak_pick(
        novelty_curve,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=wait,
    )

    if len(peaks) == 0:
        return []

    boundary_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    return boundary_times.tolist()


# -------------------------------------------------------------------------
# Segment post-processing & labeling
# -------------------------------------------------------------------------

def _merge_consecutive_same_labels(segments: list[dict]) -> list[dict]:
    """
    Merges consecutive segments that share the same label.
    """
    if not segments:
        return segments

    merged: list[dict] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue

        last = merged[-1]
        if last["label"] == seg["label"]:
            last["end"] = seg["end"]
        else:
            merged.append(seg)

    return merged


def _enforce_min_segment_duration(
    segments: list[dict],
    min_duration: float,
    total_duration: float,
) -> list[dict]:
    """
    Ensures that no segment is shorter than min_duration seconds.
    Short segments are merged with neighbors (prefer longer neighbor).
    """
    if not segments:
        return segments

    changed = True
    segs = segments

    while changed:
        changed = False
        new_segments: list[dict] = []
        i = 0

        while i < len(segs):
            seg = segs[i]
            dur = seg["end"] - seg["start"]

            if dur >= min_duration or len(segs) == 1:
                new_segments.append(seg)
                i += 1
                continue

            # Segment too short → merge with neighbor.
            if i == 0:
                neighbor = segs[i + 1]
                merged = {
                    "start": seg["start"],
                    "end": neighbor["end"],
                    "label": neighbor["label"],
                }
                new_segments.append(merged)
                i += 2
            elif i == len(segs) - 1:
                prev = new_segments[-1]
                prev["end"] = seg["end"]
                i += 1
            else:
                left = new_segments[-1]
                right = segs[i + 1]
                left_dur = left["end"] - left["start"]
                right_dur = right["end"] - right["start"]

                if left_dur >= right_dur:
                    left["end"] = seg["end"]
                    i += 1
                else:
                    merged = {
                        "start": seg["start"],
                        "end": right["end"],
                        "label": right["label"],
                    }
                    new_segments[-1] = left
                    new_segments.append(merged)
                    i += 2

            changed = True

        segs = new_segments

    if segs:
        segs[-1]["end"] = min(segs[-1]["end"], total_duration)

    return segs


def _assign_section_types(
    segments: list[dict],
    total_duration: float,
) -> list[dict]:
    """
    Assign human-friendly section types (Intro, Verse, Chorus, Outro, Other).

    Mantık:
      - Label toplam sürelerine göre: en uzun = Chorus, ikinci = Verse, diğerleri = Other.
      - Eğer 1'den fazla segment varsa:
          * ilk segment => Intro
          * son segment => Outro
      - Sadece 1 segment varsa: FullTrack gibi davranmak daha mantıklı olabilir (ama burada label'a göre kalıyor).
    """
    if not segments:
        return segments

    # 1) Label bazında toplam süre
    label_total_duration: dict[str, float] = {}
    for seg in segments:
        d = seg["end"] - seg["start"]
        label_total_duration[seg["label"]] = label_total_duration.get(seg["label"], 0.0) + d

    # En uzun label'ları sırala
    sorted_labels = sorted(label_total_duration.items(), key=lambda x: x[1], reverse=True)
    sorted_label_ids = [lab for lab, _ in sorted_labels]

    # 2) Label -> default section_type (Intro/Outro’dan bağımsız)
    label_default_type: dict[str, str] = {}
    if sorted_label_ids:
        label_default_type[sorted_label_ids[0]] = "Chorus"
    if len(sorted_label_ids) > 1:
        label_default_type[sorted_label_ids[1]] = "Verse"
    for lab in sorted_label_ids[2:]:
        label_default_type[lab] = "Other"

    enriched: list[dict] = []
    n = len(segments)

    for i, seg in enumerate(segments):
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Başlangıçta label'a göre tip ver
        section_type = label_default_type.get(seg["label"], "Other")

        if n > 1:
            # İlk segment: Intro
            if i == 0:
                section_type = "Intro"
            # Son segment: Outro
            elif i == n - 1:
                section_type = "Outro"

        enriched.append(
            {
                "start": seg_start,
                "end": seg_end,
                "label": seg["label"],
                "section_type": section_type,
            }
        )

    return enriched


def _cluster_and_label_segments(
    chroma_features: np.ndarray,
    boundaries_seconds: list[float],
    sr: int,
    hop_length: int,
    n_clusters: int,
    total_duration: float,
) -> list[dict]:
    """
    Clusters segments based on their chroma features and assigns labels.
    total_duration burada aktif müzik bölgesinin süresi (trim sonrası).
    """
    if chroma_features.size == 0:
        return []

    n_frames = chroma_features.shape[1]

    if not boundaries_seconds:
        base_segment = {
            "start": 0.0,
            "end": round(total_duration, 2),
            "label": "A",
            "section_type": "FullTrack",
        }
        return [base_segment]

    boundary_frames = librosa.time_to_frames(boundaries_seconds, sr=sr, hop_length=hop_length)

    full_boundary_frames = np.concatenate(
        (
            np.array([0], dtype=int),
            boundary_frames.astype(int),
            np.array([n_frames], dtype=int),
        )
    )

    full_boundary_frames = np.clip(full_boundary_frames, 0, n_frames)
    full_boundary_frames = np.unique(full_boundary_frames)

    segment_features: list[np.ndarray] = []
    segment_frame_spans: list[tuple[int, int]] = []

    for start_frame, end_frame in zip(full_boundary_frames[:-1], full_boundary_frames[1:]):
        if end_frame <= start_frame:
            continue

        segment_chroma = chroma_features[:, int(start_frame):int(end_frame)]
        if segment_chroma.size == 0:
            continue

        segment_median = np.median(segment_chroma, axis=1)
        segment_features.append(segment_median)
        segment_frame_spans.append((int(start_frame), int(end_frame)))

    if not segment_features:
        base_segment = {
            "start": 0.0,
            "end": round(total_duration, 2),
            "label": "A",
            "section_type": "FullTrack",
        }
        return [base_segment]

    n_effective_clusters = min(n_clusters, len(segment_features))
    kmeans = KMeans(
        n_clusters=n_effective_clusters,
        random_state=0,
        n_init=10,
    )
    labels = kmeans.fit_predict(segment_features)

    label_counts = Counter(labels)
    sorted_label_ids = [label for label, _ in label_counts.most_common()]
    numeric_to_char: dict[int, str] = {label: chr(65 + i) for i, label in enumerate(sorted_label_ids)}

    raw_segments: list[dict] = []
    for (start_frame, end_frame), numeric_label in zip(segment_frame_spans, labels):
        start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
        end_time = librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length)

        if end_time <= start_time:
            continue

        raw_segments.append(
            {
                "start": round(float(start_time), 2),
                "end": round(float(end_time), 2),
                "label": numeric_to_char[numeric_label],
            }
        )

    merged_segments = _merge_consecutive_same_labels(raw_segments)

    merged_segments = _enforce_min_segment_duration(
        merged_segments,
        MIN_SEGMENT_DURATION_SECONDS,
        total_duration,
    )

    enriched_segments = _assign_section_types(merged_segments, total_duration)

    return enriched_segments


# -------------------------------------------------------------------------
# Main entrypoint
# -------------------------------------------------------------------------

async def analyze_and_segment_audio(file: UploadFile):
    """
    Orchestrates the audio segmentation process.
    Pipeline:
      1. Load audio.
      2. Detect active music region [active_start, active_end] via RMS.
      3. Run segmentation ONLY on y_active = y[active_start:active_end].
      4. Cluster + label segments (Intro/Verse/Chorus/Outro/Other).
      5. Shift segment times back to original timeline.
    """
    logger.info(f"Starting analysis for: {file.filename}")

    try:
        content = await file.read()
        y, sr = _load_audio_from_bytes(content)

        original_duration = librosa.get_duration(y=y, sr=sr)
        logger.info(f"Loaded audio: sr={sr}, original_duration≈{original_duration:.2f}s")

        # --- 1) Detect active music region (adaptive, per track) ---
        active_start_s, active_end_s = _detect_active_region(
            y,
            sr,
            hop_length=CQT_HOP_LENGTH,
            margin_db=ACTIVE_MARGIN_DB,
            min_region_s=MIN_ACTIVE_REGION_SECONDS,
        )
        core_duration = max(0.0, active_end_s - active_start_s)

        logger.info(
            f"Active music region: start={active_start_s:.2f}s, "
            f"end={active_end_s:.2f}s, core_duration≈{core_duration:.2f}s"
        )

        if core_duration <= 0.0:
            logger.warning("No clear active region detected; returning empty segmentation.")
            return {
                "filename": file.filename,
                "content_type": file.content_type,
                "duration_seconds": round(original_duration, 2),
                "segments": [],
                "status": "No clear active music region detected.",
            }

        # Slice out only the active region for analysis.
        start_sample = int(active_start_s * sr)
        end_sample = int(active_end_s * sr)
        y_active = y[start_sample:end_sample]

        # --- 2) Extract chroma on active region ---
        logger.info("Extracting chroma features (active region only)...")
        chroma_features = _extract_chroma_features(y_active, sr)
        logger.info(f"Chroma shape (n_chroma, n_frames): {chroma_features.shape}")

        if DOWNSAMPLE_FACTOR > 1:
            chroma_features_ds = chroma_features[:, ::DOWNSAMPLE_FACTOR]
        else:
            chroma_features_ds = chroma_features

        effective_hop_length = CQT_HOP_LENGTH * DOWNSAMPLE_FACTOR
        logger.info(
            f"Downsampled chroma shape: {chroma_features_ds.shape}, "
            f"effective_hop_length={effective_hop_length}"
        )

        # --- 3) SSM & novelty ---
        logger.info("Computing self-similarity matrix...")
        ssm = _compute_ssm(chroma_features_ds)
        logger.info(f"SSM shape: {ssm.shape}")

        logger.info("Computing novelty curve and finding boundaries...")
        frames_per_second = sr / effective_hop_length
        kernel_size_frames = int(NOVELTY_KERNEL_SIZE_SECONDS * frames_per_second)
        kernel_size_frames = max(1, min(kernel_size_frames, MAX_KERNEL_BLOCK_SIZE))

        novelty_curve = _compute_novelty_curve(ssm, kernel_size_frames=kernel_size_frames)
        boundaries_core = _find_boundaries(
            novelty_curve,
            sr=sr,
            hop_length=effective_hop_length,
            min_segment_duration_s=MIN_SEGMENT_DURATION_SECONDS,
        )
        logger.info(f"Detected {len(boundaries_core)} boundaries inside active region.")

        # --- 4) Cluster & label segments in active region ---
        logger.info("Clustering and labeling segments...")
        segments_core = _cluster_and_label_segments(
            chroma_features_ds,
            boundaries_core,
            sr,
            effective_hop_length,
            N_CLUSTERS,
            total_duration=core_duration,
        )

        # --- 5) Shift times back to original timeline ---
        for seg in segments_core:
            seg["start"] = round(seg["start"] + active_start_s, 2)
            seg["end"] = round(seg["end"] + active_start_s, 2)

        logger.info(f"Successfully processed {file.filename}")

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "duration_seconds": round(original_duration, 2),
            "segments": segments_core,
            "status": "Segmentation and labeling complete.",
        }
    except Exception as e:
        logger.error(f"Error processing file {file.filename}", exc_info=e)
        raise e
