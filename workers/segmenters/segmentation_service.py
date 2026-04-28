import io
from collections import Counter

import librosa
import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from shared.logger import get_logger

logger = get_logger()

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

# --- New optimisation defaults ---

# Number of MFCC coefficients to include in fused features.
MFCC_N_COMPONENTS = 13

# Blend weight for spectral flux novelty curve (0 = pure SSM, 1 = pure spectral flux).
SPECTRAL_FLUX_WEIGHT = 0.4

# When True, use silhouette score to auto-select k (range 2..8).
AUTO_N_CLUSTERS = True

# Hard cap on the number of frames used for the SSM path.
# If a track produces more frames, we downsample before the expensive all-pairs similarity step.
MAX_SSM_FRAMES = 3500


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
    except Exception:
        logger.error("Failed to load audio from bytes.", exc_info=True)
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


def _extract_fused_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = MFCC_N_COMPONENTS,
) -> np.ndarray:
    """
    Fused feature matrix: L2-normalised Chroma CQT + L2-normalised MFCC.

    Combining harmonic content (chroma) with timbral texture (MFCC) makes the
    Self-Similarity Matrix sensitive to both pitch-class changes (verse→chorus
    key changes) AND timbral changes (sparse guitar vs. full-band sections),
    resulting in more accurate boundary detection.

    Returns shape (n_chroma + n_mfcc, n_frames).
    """
    y_harmonic, _ = librosa.effects.hpss(y)

    chroma = librosa.feature.chroma_cqt(
        y=y_harmonic,
        sr=sr,
        hop_length=CQT_HOP_LENGTH,
    )  # (12, n_frames)

    mfcc = librosa.feature.mfcc(
        y=y_harmonic,
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=CQT_HOP_LENGTH,
    )  # (n_mfcc, n_frames)

    # L2-normalise each feature type independently (axis=0 = per frame)
    chroma_norm = normalize(chroma, norm="l2", axis=0)
    mfcc_norm = normalize(mfcc, norm="l2", axis=0)

    return np.vstack([chroma_norm, mfcc_norm])  # (12 + n_mfcc, n_frames)


def _compute_ssm(features: np.ndarray) -> np.ndarray:
    """
    Computes a self-similarity matrix from a feature matrix.
    Features should be shape (n_features, n_frames).
    """
    # Use float32 to reduce memory pressure on large matrices.
    features_transposed = np.ascontiguousarray(features.T, dtype=np.float32)
    ssm = features_transposed @ features_transposed.T
    return ssm


def _downsample_features_for_ssm(features: np.ndarray, max_frames: int = MAX_SSM_FRAMES) -> tuple[np.ndarray, int]:
    """Downsample frames so the expensive SSM step stays within a bounded size."""
    n_frames = features.shape[1]
    if n_frames <= max_frames:
        return features, 1

    factor = int(np.ceil(n_frames / max_frames))
    factor = max(1, factor)
    downsampled = features[:, ::factor]
    logger.info(
        f"Downsampling features for SSM: n_frames={n_frames}, max_frames={max_frames}, factor={factor}, downsampled_frames={downsampled.shape[1]}"
    )
    return downsampled, factor


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


def _compute_spectral_flux_novelty(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute a spectral flux novelty curve using librosa onset strength.
    Returns a 1-D array normalised to [0, 1] with the same frame grid as CQT_HOP_LENGTH.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=CQT_HOP_LENGTH)
    if onset_env.max() > 0:
        onset_env = onset_env / onset_env.max()
    return onset_env


def _blend_novelty_curves(
    ssm_novelty: np.ndarray,
    flux_novelty: np.ndarray,
    flux_weight: float,
) -> np.ndarray:
    """Blend SSM-based and spectral-flux novelty curves with a given weight."""
    # Align lengths (SSM curve may differ by 1 frame)
    min_len = min(len(ssm_novelty), len(flux_novelty))
    ssm_novelty = ssm_novelty[:min_len]
    flux_novelty = flux_novelty[:min_len]

    blended = (1.0 - flux_weight) * ssm_novelty + flux_weight * flux_novelty
    max_val = blended.max()
    if max_val > 0:
        blended = blended / max_val
    return blended


def _find_boundaries(
    novelty_curve: np.ndarray,
    sr: int,
    hop_length: int,
    min_segment_duration_s: float,
    y_active: np.ndarray | None = None,
    snap_to_rms_dip: bool = True,
) -> list[float]:
    """
    Finds boundary timestamps from a novelty curve by picking peaks.

    Improvements over the original:
      1. Adaptive delta: threshold scales with the curve's 75th percentile,
         avoiding both over-segmentation on loud tracks and missed boundaries
         on quiet ones.
      2. RMS-dip snapping (optional): each detected boundary is shifted by up
         to ±1 second to the nearest local energy minimum, which is where
         music transitions typically occur.

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

    # Adaptive delta: scale to the dynamic range of the curve
    delta = max(0.1, float(np.percentile(novelty_curve, 75)) * 0.3)

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

    # Optional RMS-dip snapping
    if snap_to_rms_dip and y_active is not None and len(y_active) > 0:
        snap_window_frames = max(1, int(1.0 * frames_per_second))  # ±1 second
        rms = librosa.feature.rms(y=y_active, hop_length=hop_length)[0]
        snapped_peaks = []
        for p in peaks:
            lo = max(0, p - snap_window_frames)
            hi = min(len(rms), p + snap_window_frames)
            if hi > lo:
                local_min = lo + int(np.argmin(rms[lo:hi]))
                snapped_peaks.append(local_min)
            else:
                snapped_peaks.append(p)
        peaks = np.array(snapped_peaks)
        peaks = np.unique(peaks)

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


def _select_n_clusters(X: np.ndarray, max_k: int = 8) -> int:
    """
    Auto-select number of clusters using silhouette score (range 2..max_k).
    Falls back to 2 if there are fewer samples than max_k.
    """
    from sklearn.metrics import silhouette_score

    n_samples = len(X)
    if n_samples < 3:
        return min(2, n_samples)

    upper = min(max_k, n_samples - 1)
    if upper < 2:
        return 2

    best_k = 2
    best_score = -1.0
    for k in range(2, upper + 1):
        try:
            km = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = km.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    return best_k


def _cluster_and_label_segments(
    features: np.ndarray,
    boundaries_seconds: list[float],
    sr: int,
    hop_length: int,
    n_clusters: int,
    min_segment_duration_seconds: float,
    total_duration: float,
    auto_n_clusters: bool = False,
) -> list[dict]:
    """
    Clusters segments based on their features and assigns labels.

    Uses mean + std of features per segment for richer representation
    (std captures textural density, e.g. sparse guitar vs. full band).
    When auto_n_clusters=True, silhouette score selects k automatically.

    total_duration is the active music region duration (after trim).
    """
    if features.size == 0:
        return []

    n_frames = features.shape[1]

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

        seg_feats = features[:, int(start_frame):int(end_frame)]
        if seg_feats.size == 0:
            continue

        # Richer representation: mean + std per feature dimension
        seg_mean = np.mean(seg_feats, axis=1)
        seg_std = np.std(seg_feats, axis=1)
        seg_descriptor = np.concatenate([seg_mean, seg_std])

        segment_features.append(seg_descriptor)
        segment_frame_spans.append((int(start_frame), int(end_frame)))

    if not segment_features:
        base_segment = {
            "start": 0.0,
            "end": round(total_duration, 2),
            "label": "A",
            "section_type": "FullTrack",
        }
        return [base_segment]

    X = np.array(segment_features)

    if auto_n_clusters:
        n_effective_clusters = _select_n_clusters(X, max_k=min(8, len(segment_features)))
        logger.info(f"Auto-selected n_clusters={n_effective_clusters} via silhouette score")
    else:
        n_effective_clusters = min(n_clusters, len(segment_features))

    kmeans = KMeans(
        n_clusters=n_effective_clusters,
        random_state=0,
        n_init=10,
    )
    labels = kmeans.fit_predict(X)

    label_counts = Counter(labels.tolist())
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
        min_segment_duration_seconds,
        total_duration,
    )

    enriched_segments = _assign_section_types(merged_segments, total_duration)

    return enriched_segments


# -------------------------------------------------------------------------
# Main entrypoint
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Main entrypoints
# -------------------------------------------------------------------------

def process_file_path(file_path: str, params: dict | None = None):
    """
    Worker-friendly entry point. Reads file from disk.
    """
    logger.info(f"Starting analysis for file path: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        
        filename = file_path.split("/")[-1]
        
        # Reuse the core logic by passing content bytes
        return _analyze_content(content, filename, params=params)
    except Exception:
        logger.error(f"Error processing file path {file_path}", exc_info=True)
        raise


def _analyze_content(
    content: bytes,
    filename: str,
    content_type: str = "audio/wav",
    params: dict | None = None,
):
    """
    Shared core logic for segmentation.

    Optimisation parameters (all optional, with sensible defaults):
      use_mfcc (bool):               Fuse MFCC with Chroma CQT for richer features.
      mfcc_n_components (int):       Number of MFCC coefficients (default 13).
      spectral_flux_weight (float):  Blend weight for spectral-flux novelty (0-1).
      auto_n_clusters (bool):        Auto-select k via silhouette score.
    """
    y, sr = _load_audio_from_bytes(content)
    params = params or {}

    min_segment_duration_seconds = float(
        params.get("min_segment_duration_seconds", MIN_SEGMENT_DURATION_SECONDS)
    )
    novelty_kernel_size_seconds = float(
        params.get("novelty_kernel_size_seconds", NOVELTY_KERNEL_SIZE_SECONDS)
    )
    n_clusters = int(params.get("n_clusters", N_CLUSTERS))
    use_mfcc = bool(params.get("use_mfcc", True))
    mfcc_n_components = int(params.get("mfcc_n_components", MFCC_N_COMPONENTS))
    spectral_flux_weight = float(params.get("spectral_flux_weight", SPECTRAL_FLUX_WEIGHT))
    spectral_flux_weight = max(0.0, min(1.0, spectral_flux_weight))
    auto_n_clusters = bool(params.get("auto_n_clusters", AUTO_N_CLUSTERS))

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
            "filename": filename,
            "content_type": content_type,
            "duration_seconds": round(original_duration, 2),
            "segments": [],
            "status": "No clear active music region detected.",
        }

    # Slice out only the active region for analysis.
    start_sample = int(active_start_s * sr)
    end_sample = int(active_end_s * sr)
    y_active = y[start_sample:end_sample]

    # --- 2) Feature extraction on active region ---
    if use_mfcc:
        logger.info("Extracting fused Chroma+MFCC features (active region only)...")
        features = _extract_fused_features(y_active, sr, n_mfcc=mfcc_n_components)
        logger.info(f"Fused feature shape (n_feats, n_frames): {features.shape}")
    else:
        logger.info("Extracting Chroma CQT features only (active region only)...")
        features = _extract_chroma_features(y_active, sr)
        logger.info(f"Chroma shape (n_chroma, n_frames): {features.shape}")

    if DOWNSAMPLE_FACTOR > 1:
        features_ds = features[:, ::DOWNSAMPLE_FACTOR]
        hop_after_static_ds = CQT_HOP_LENGTH * DOWNSAMPLE_FACTOR
    else:
        features_ds = features
        hop_after_static_ds = CQT_HOP_LENGTH

    features_ds, adaptive_factor = _downsample_features_for_ssm(features_ds, max_frames=MAX_SSM_FRAMES)
    effective_hop_length = hop_after_static_ds * adaptive_factor

    # --- 3) SSM & novelty ---
    logger.info("Computing self-similarity matrix...")
    ssm = _compute_ssm(features_ds)
    logger.info(f"SSM shape: {ssm.shape}")

    logger.info("Computing novelty curve and finding boundaries...")
    frames_per_second = sr / effective_hop_length
    kernel_size_frames = int(novelty_kernel_size_seconds * frames_per_second)
    kernel_size_frames = max(1, min(kernel_size_frames, MAX_KERNEL_BLOCK_SIZE))

    ssm_novelty = _compute_novelty_curve(ssm, kernel_size_frames=kernel_size_frames)

    # Blend with spectral flux for more robust boundary detection
    if spectral_flux_weight > 0.0:
        flux_novelty = _compute_spectral_flux_novelty(y_active, sr)
        novelty_curve = _blend_novelty_curves(ssm_novelty, flux_novelty, spectral_flux_weight)
    else:
        novelty_curve = ssm_novelty

    boundaries_core = _find_boundaries(
        novelty_curve,
        sr=sr,
        hop_length=effective_hop_length,
        min_segment_duration_s=min_segment_duration_seconds,
        y_active=y_active,
        snap_to_rms_dip=True,
    )
    logger.info(f"Detected {len(boundaries_core)} boundaries inside active region.")

    # --- 4) Cluster & label segments in active region ---
    logger.info("Clustering and labeling segments...")
    segments_core = _cluster_and_label_segments(
        features_ds,
        boundaries_core,
        sr,
        effective_hop_length,
        n_clusters,
        min_segment_duration_seconds=min_segment_duration_seconds,
        total_duration=core_duration,
        auto_n_clusters=auto_n_clusters,
    )

    # --- 5) Shift times back to original timeline ---
    for seg in segments_core:
        seg["start"] = round(seg["start"] + active_start_s, 2)
        seg["end"] = round(seg["end"] + active_start_s, 2)

    logger.info(f"Successfully processed {filename}")

    return {
        "filename": filename,
        "content_type": content_type,
        "duration_seconds": round(original_duration, 2),
        "segments": segments_core,
        "status": "Segmentation and labeling complete.",
    }
