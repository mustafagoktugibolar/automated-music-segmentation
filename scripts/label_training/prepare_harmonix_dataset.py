#!/usr/bin/env python
"""
Harmonix mel-spectrogram → segment-label training parquet.

Reads:
    data/harmonix/Harmonix_melspecs.tar   (or --tar path)
    data/harmonix/segments/               (downloaded from GitHub on first run)
    data/harmonix/beats/                  (same)

Writes:
    data/label_training/harmonix_segments.parquet
        columns: song_id, raw_track_id, annotator_id, dataset,
                 segment_idx, start, end, label,
                 + 87 feature columns (same schema as segments.parquet)

Feature extraction from mel spectrograms
-----------------------------------------
  MFCCs            exact   (DCT of log-mel via librosa)
  RMS              exact   (sqrt of mean mel power per frame)
  Spectral centroid exact   (weighted mean of mel-bin frequencies)
  Onset density    good    (mel spectral flux)
  Tempo / beats    exact   (from Harmonix beat annotations)
  Beat regularity  exact   (from beat intervals)
  Chroma           approx  (mel-bin → pitch-class projection)
  ZCR              missing (waveform required; set to 0.0)

Usage (inside Docker container):
    python scripts/label_training/prepare_harmonix_dataset.py
    python scripts/label_training/prepare_harmonix_dataset.py --tar /app/data/harmonix/Harmonix_melspecs.tar
    python scripts/label_training/prepare_harmonix_dataset.py --max-songs 50  # quick test
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

_here     = os.path.dirname(os.path.abspath(__file__))
_app_root = os.path.abspath(os.path.join(_here, "..", ".."))
if _app_root not in sys.path:
    sys.path.insert(0, _app_root)

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
HARMONIX_DIR    = os.path.join(_app_root, "data", "harmonix")
DEFAULT_TAR     = os.path.join(HARMONIX_DIR, "Harmonix_melspecs.tar")
SEGMENTS_CACHE  = os.path.join(HARMONIX_DIR, "segments")
BEATS_CACHE     = os.path.join(HARMONIX_DIR, "beats")
OUTPUT_PARQUET  = os.path.join(_app_root, "data", "label_training", "harmonix_segments.parquet")

# ── Mel spectrogram parameters (from tar's info.json) ─────────────────────────
_SR          = 22050
_HOP_LENGTH  = 1024
_N_MELS      = 80
_N_FFT       = 2048
_FPS         = _SR / _HOP_LENGTH   # ≈ 21.53 frames per second

# ── Label mapping Harmonix → canonical ────────────────────────────────────────
_LABEL_MAP: dict[str, str | None] = {
    "intro":                "Intro",
    "verse":                "Verse",
    "chorus":               "Chorus",
    "chorus_instrumental":  "Chorus",
    "bridge":               "Bridge",
    "prechorus":            "Pre-Chorus",
    "outro":                "Outro",
    "instrumental":         "Instrumental",
    "solo":                 "Instrumental",
    "solo2":                "Instrumental",
    "breakdown":            "Other",
    "postchorus":           "Other",
    "bre":                  "Other",
    "silence":              "Silence",
    "end":                  None,   # end-of-song marker — discarded
}

# ── GitHub raw URLs ────────────────────────────────────────────────────────────
_GH_BASE     = "https://raw.githubusercontent.com/urinieto/harmonixset/main/dataset"
_SEGMENTS_URL = f"{_GH_BASE}/segments"
_BEATS_URL    = f"{_GH_BASE}/beats_and_downbeats"


# ── Annotation download ───────────────────────────────────────────────────────

def _fetch_url(url: str) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return r.read().decode()
    except Exception as e:
        print(f"  [warn] download failed: {url}  ({e})")
        return None


def _ensure_annotations(song_names: list[str], workers: int = 16) -> None:
    """Download segment + beat annotations if not already cached."""
    os.makedirs(SEGMENTS_CACHE, exist_ok=True)
    os.makedirs(BEATS_CACHE, exist_ok=True)

    to_fetch: list[tuple[str, str, str]] = []  # (url, cache_path, kind)
    for name in song_names:
        seg_path  = os.path.join(SEGMENTS_CACHE, f"{name}.txt")
        beat_path = os.path.join(BEATS_CACHE,    f"{name}.txt")
        if not os.path.exists(seg_path):
            to_fetch.append((f"{_SEGMENTS_URL}/{name}.txt",      seg_path,  "seg"))
        if not os.path.exists(beat_path):
            to_fetch.append((f"{_BEATS_URL}/{name}.txt", beat_path, "beat"))

    if not to_fetch:
        print(f"  [cache] All {len(song_names)} annotation files already cached.")
        return

    print(f"  Downloading {len(to_fetch)} annotation files from GitHub …")
    done = 0

    def _dl(args):
        url, path, kind = args
        text = _fetch_url(url)
        if text:
            with open(path, "w") as f:
                f.write(text)
        return path, text is not None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, ok in pool.map(_dl, to_fetch):
            done += 1
            if done % 100 == 0:
                print(f"    {done}/{len(to_fetch)} downloaded")

    missing = sum(1 for _, _, kind in to_fetch
                  if not os.path.exists(os.path.join(
                      SEGMENTS_CACHE if kind == "seg" else BEATS_CACHE,
                      os.path.basename(_.split("/")[-1] if isinstance(_, str) else "")
                  )))


# ── Annotation parsing ────────────────────────────────────────────────────────

def _parse_segments(path: str) -> list[tuple[float, float, str]]:
    """Return [(start, end, canonical_label), ...] with 'end' marker removed."""
    lines = open(path).read().strip().splitlines()
    times: list[float] = []
    labels: list[str]  = []
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        times.append(float(parts[0]))
        labels.append(parts[-1].lower())

    segments: list[tuple[float, float, str]] = []
    for i, (t, lbl) in enumerate(zip(times, labels)):
        canonical = _LABEL_MAP.get(lbl)
        if canonical is None:
            continue  # skip "end" and unknown labels
        end_t = times[i + 1] if i + 1 < len(times) else t + 10.0
        segments.append((t, end_t, canonical))
    return segments


def _parse_beats(path: str) -> np.ndarray:
    """Return sorted array of beat times in seconds."""
    beats: list[float] = []
    if not os.path.exists(path):
        return np.array([], dtype=np.float64)
    for line in open(path).read().strip().splitlines():
        parts = line.split()
        if parts:
            try:
                beats.append(float(parts[0]))
            except ValueError:
                pass
    return np.array(beats, dtype=np.float64)


# ── Mel → acoustic descriptor ─────────────────────────────────────────────────

# (12, 80) projection matrix built once at module level
def _build_mel_chroma_matrix() -> np.ndarray:
    """Map 80 mel bins → 12 chroma pitch classes.

    Uses librosa if available (more accurate filterbank), otherwise falls back
    to HTK mel-scale formula so the script works without librosa on the host.
    """
    try:
        import librosa
        mel_fb = librosa.filters.mel(
            sr=_SR, n_fft=_N_FFT, n_mels=_N_MELS, fmin=0, fmax=_SR / 2,
        )  # (80, 1025)
        freqs = librosa.fft_frequencies(sr=_SR, n_fft=_N_FFT)  # (1025,)
        center_freqs = (mel_fb * freqs[None, :]).sum(axis=1) / (mel_fb.sum(axis=1) + 1e-8)
    except ImportError:
        # HTK mel → Hz: f = 700*(10^(m/2595) - 1)
        mel_min = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
        mel_max = 2595.0 * np.log10(1.0 + (_SR / 2.0) / 700.0)
        mel_pts = np.linspace(mel_min, mel_max, _N_MELS + 2)[1:-1]
        center_freqs = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)

    mat = np.zeros((12, _N_MELS), dtype=np.float32)
    for k, f in enumerate(center_freqs):
        if f > 0:
            pc = int(round(12.0 * np.log2(max(f, 1.0) / 440.0))) % 12
            mat[pc, k] = 1.0
    return mat  # (12, 80)


_MEL_CHROMA_MATRIX: np.ndarray | None = None


def _get_chroma_matrix() -> np.ndarray:
    global _MEL_CHROMA_MATRIX
    if _MEL_CHROMA_MATRIX is None:
        _MEL_CHROMA_MATRIX = _build_mel_chroma_matrix()
    return _MEL_CHROMA_MATRIX


def _segment_descriptor(
    mel: np.ndarray,
    start_s: float,
    end_s: float,
    beats: np.ndarray,
    total_dur_s: float,
) -> np.ndarray:
    """Return 60-dim acoustic descriptor matching workers/infrastructure/audio/features.py layout."""
    # Frame indices for this segment
    f0 = max(0, int(round(start_s * _FPS)))
    f1 = min(mel.shape[1], int(round(end_s * _FPS)))
    if f1 <= f0:
        f1 = f0 + 1

    seg_mel = mel[:, f0:f1].astype(np.float64)  # (80, T)
    T = seg_mel.shape[1]
    dur_s = (f1 - f0) / _FPS

    # ── Chroma (approximate via pitch-class projection) ──────────────────────
    chroma_mat = _get_chroma_matrix().astype(np.float64)  # (12, 80)
    chroma_raw = chroma_mat @ seg_mel  # (12, T)
    col_norms  = np.linalg.norm(chroma_raw, axis=0, keepdims=True)
    chroma_raw = chroma_raw / (col_norms + 1e-8)
    chroma_mean = chroma_raw.mean(axis=1).astype(np.float32)  # (12,)
    chroma_std  = chroma_raw.std(axis=1).astype(np.float32)   # (12,)

    # ── MFCCs (exact: DCT of log-mel) ────────────────────────────────────────
    log_mel = np.log(seg_mel + 1e-6)  # (80, T)
    try:
        import librosa as _lib
        mfcc_all = _lib.feature.mfcc(S=log_mel, n_mfcc=13)  # (13, T)
    except ImportError:
        from scipy.fft import dct
        mfcc_all = dct(log_mel, axis=0, norm="ortho")[:13]   # (13, T)
    mfcc_mean = mfcc_all.mean(axis=1).astype(np.float32)
    mfcc_std  = mfcc_all.std(axis=1).astype(np.float32)

    # ── RMS ───────────────────────────────────────────────────────────────────
    rms_frames = np.sqrt(seg_mel.mean(axis=0))  # (T,)
    rms_mean   = float(rms_frames.mean())
    rms_std    = float(rms_frames.std())

    # ── Onset density (mel spectral flux) ─────────────────────────────────────
    if T > 1:
        flux = np.maximum(0.0, np.diff(seg_mel, axis=1)).sum(axis=0)  # (T-1,)
        onset_density = float(flux.mean() / max(dur_s, 1e-3))
    else:
        onset_density = 0.0

    # ── Norm duration ─────────────────────────────────────────────────────────
    norm_duration = float(np.clip(dur_s / max(total_dur_s, 1.0), 0.0, 1.0))

    # ── Tempo & beat features (from annotator beats) ──────────────────────────
    seg_beats = beats[(beats >= start_s) & (beats < end_s)]
    n_beats   = len(seg_beats)
    if n_beats > 1 and dur_s > 0:
        intervals       = np.diff(seg_beats)
        mean_interval   = float(intervals.mean())
        tempo           = 60.0 / max(mean_interval, 1e-3)
        beat_density    = n_beats / dur_s
        beat_regularity = float(np.clip(
            1.0 - intervals.std() / (mean_interval + 1e-8), 0.0, 1.0
        ))
    else:
        tempo           = 0.0
        beat_density    = 0.0
        beat_regularity = 0.0

    tempo_norm        = float(np.clip(tempo / 200.0, 0.0, 2.0))
    beat_density_norm = float(np.clip(beat_density / 5.0, 0.0, 2.0))

    # ── Spectral centroid (from mel bin frequencies) ──────────────────────────
    try:
        import librosa as _lib
        mel_freqs = _lib.mel_frequencies(n_mels=_N_MELS + 2, fmin=0, fmax=_SR / 2)[1:-1]
    except ImportError:
        mel_min   = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
        mel_max   = 2595.0 * np.log10(1.0 + (_SR / 2.0) / 700.0)
        mel_pts   = np.linspace(mel_min, mel_max, _N_MELS + 2)[1:-1]
        mel_freqs = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    total_pow  = seg_mel.sum(axis=0) + 1e-8
    centroid_f = (mel_freqs[:, None] * seg_mel).sum(axis=0) / total_pow
    fmax       = _SR / 2.0
    centroid_mean_norm = float(centroid_f.mean() / fmax)
    centroid_std_norm  = float(centroid_f.std()  / fmax)

    # ── ZCR (not available from mel → 0.0) ───────────────────────────────────
    zcr_mean = 0.0

    # ── Assemble [0:60] ───────────────────────────────────────────────────────
    descriptor = np.concatenate([
        chroma_mean,    # [0:12]
        chroma_std,     # [12:24]
        mfcc_mean,      # [24:37]
        mfcc_std,       # [37:50]
        np.array([
            rms_mean, rms_std, onset_density, norm_duration,
            tempo_norm, beat_density_norm, beat_regularity,
            centroid_mean_norm, centroid_std_norm, zcr_mean,
        ], dtype=np.float32),  # [50:60]
    ]).astype(np.float32)

    return descriptor  # (60,)


# ── Per-song processing ───────────────────────────────────────────────────────

def _get_build_fn():
    """Return build_segment_label_vectors; works in Docker and on host."""
    try:
        from workers.core.labeling.features import build_segment_label_vectors
        return build_segment_label_vectors
    except ImportError:
        # Host fallback: workers package not importable; use local copy of features module.
        import importlib.util, pathlib
        feat_path = pathlib.Path(_app_root) / "workers" / "core" / "labeling" / "features.py"
        spec = importlib.util.spec_from_file_location("_features", feat_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.build_segment_label_vectors


def _process_song(
    song_name: str,
    mel: np.ndarray,
) -> list[dict] | None:
    """Return list of row dicts (one per segment) or None on failure."""
    build_segment_label_vectors = _get_build_fn()

    seg_path  = os.path.join(SEGMENTS_CACHE, f"{song_name}.txt")
    beat_path = os.path.join(BEATS_CACHE,    f"{song_name}.txt")

    if not os.path.exists(seg_path):
        return None  # annotation not downloaded

    raw_segs = _parse_segments(seg_path)
    if not raw_segs:
        return None

    beats = _parse_beats(beat_path)

    total_dur_s = mel.shape[1] / _FPS

    # Build segment dicts and descriptor matrix
    segments: list[dict] = []
    descs:    list[np.ndarray] = []
    for start_s, end_s, canonical in raw_segs:
        segments.append({"start": start_s, "end": end_s})
        desc = _segment_descriptor(mel, start_s, end_s, beats, total_dur_s)
        descs.append(desc)

    descriptors = np.stack(descs, axis=0)  # (N, 60)

    # Compute all 87 features using the same function as SALAMI
    X, feat_names = build_segment_label_vectors(segments, descriptors=descriptors)

    song_id = f"harmonix_{song_name}"
    rows: list[dict] = []
    for idx, ((start_s, end_s, canonical), feat_vec) in enumerate(
        zip(raw_segs, X)
    ):
        row: dict = {
            "song_id":      song_id,
            "raw_track_id": song_id,
            "annotator_id": 0,
            "dataset":      "harmonix",
            "segment_idx":  idx,
            "start":        round(start_s, 3),
            "end":          round(end_s,   3),
            "label":        canonical,
        }
        for fname, fval in zip(feat_names, feat_vec.tolist()):
            row[fname] = fval
        rows.append(row)

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        description="Build Harmonix mel-spec → segment-label parquet."
    )
    parser.add_argument(
        "--tar", default=DEFAULT_TAR,
        help=f"Path to Harmonix_melspecs.tar (default: {DEFAULT_TAR})",
    )
    parser.add_argument("--output",    default=OUTPUT_PARQUET)
    parser.add_argument("--max-songs", type=int, default=0,
                        help="Limit songs for quick testing (0 = all).")
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--dl-workers", type=int, default=20,
                        help="Parallel download threads for annotations.")
    args = parser.parse_args()

    if not os.path.exists(args.tar):
        print(f"[error] Tar not found: {args.tar}")
        print(f"  Copy it with:  docker cp /Users/goktugibolar/Downloads/Harmonix_melspecs.tar "
              f"music-segmentation-worker-custom-5:{args.tar}")
        sys.exit(1)

    # ── Enumerate songs in tar ────────────────────────────────────────────────
    print(f"Opening tar: {args.tar} …")
    tf = tarfile.open(args.tar)
    npy_members = [
        m for m in tf.getmembers()
        if m.name.endswith("-mel.npy") and not os.path.basename(m.name).startswith("._")
    ]
    npy_members.sort(key=lambda m: m.name)

    song_names: list[str] = []
    member_map: dict[str, tarfile.TarInfo] = {}
    for m in npy_members:
        # melspecs/NNNN_songname-mel.npy → NNNN_songname
        name = os.path.basename(m.name).replace("-mel.npy", "")
        song_names.append(name)
        member_map[name] = m

    if args.max_songs > 0:
        song_names = song_names[: args.max_songs]
    print(f"Found {len(song_names)} songs in tar.")

    # ── Download annotations ──────────────────────────────────────────────────
    _ensure_annotations(song_names, workers=args.dl_workers)

    # ── Process songs ─────────────────────────────────────────────────────────
    print(f"\nExtracting features  (workers={args.workers}) …", flush=True)
    all_rows: list[dict] = []
    skipped  = 0
    t0       = time.perf_counter()

    # Pipeline: submit to pool as each mel is extracted; avoids loading all into RAM at once.
    pending: dict = {}
    total = len(song_names)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name in song_names:
            m   = member_map[name]
            buf = tf.extractfile(m)
            if buf is None:
                skipped += 1
                continue
            mel = np.load(io.BytesIO(buf.read()), allow_pickle=False)
            pending[pool.submit(_process_song, name, mel)] = name

        tf.close()

        done = 0
        for fut in as_completed(pending):
            rows = fut.result()
            done += 1
            if rows:
                all_rows.extend(rows)
            else:
                skipped += 1
            if done % 50 == 0 or done == len(pending):
                elapsed = time.perf_counter() - t0
                print(
                    f"  {done}/{total}  "
                    f"segments: {len(all_rows)}  "
                    f"skipped: {skipped}  "
                    f"elapsed: {elapsed:.1f}s",
                    flush=True,
                )

    if not all_rows:
        print("No rows produced. Check annotation downloads and tar path.")
        sys.exit(1)

    # ── Save parquet ──────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"\nWrote {len(df)} rows  ({df['song_id'].nunique()} songs)  → {args.output}")
    print("Label distribution:")
    print(df["label"].value_counts().to_string())
    print(f"\nFeature columns: {df.shape[1] - 8} (expected 87)")


if __name__ == "__main__":
    main()
