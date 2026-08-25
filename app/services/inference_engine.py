"""
ML Inference Engine - Dialflo Voice Demographic Service

Uses acoustic features and embeddings for age and gender prediction.
Primary: Pitch (F0) based gender detection + acoustic features for age
Fallback: Simple acoustic analysis

TARGET LATENCY: ~50-200ms on CPU for 5-second audio

FIXES in v2:
    1. Removed unused import (scipy.io.wavfile)
    2. Shared pitch computation - compute once, use for pitch AND jitter
    3. Spectral centroid uses full audio (windowed average, not just first 2048 samples)
    4. Added named constants for magic numbers
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("dialflo-backend")


# =============================================================================
# CONFIGURATION
# =============================================================================

AGE_BRACKETS = [
    (0, 30, "18-30"),
    (30, 45, "31-45"),
    (45, 60, "46-60"),
    (60, 120, "60+")
]

MIN_CONFIDENCE_THRESHOLD = 0.4

# =============================================================================
# ACOUSTIC CONSTANTS (research-backed values)
# =============================================================================

# Pitch (F0) ranges for gender detection
# Male: 85-180 Hz (average ~120 Hz)
# Female: 165-255 Hz (average ~200 Hz)
# Overlap region: 165-180 Hz
PITCH_MIN_HZ = 50       # Below this = not human speech
PITCH_MAX_HZ = 400      # Above this = not fundamental (harmonic)
PITCH_MALE_TYPICAL = 120
PITCH_FEMALE_TYPICAL = 200

# Gender decision boundaries (Hz)
PITCH_DEFINITELY_MALE = 140      # Below this = almost certainly male
PITCH_LIKELY_MALE = 160          # Below this = likely male
PITCH_AMBIGUOUS_HIGH = 180       # Below this = ambiguous
PITCH_LIKELY_FEMALE = 200        # Below this = likely female
# Above 200 Hz = almost certainly female

# Frame analysis parameters
FRAME_LENGTH_MS = 30    # 30ms frames (standard for speech)
HOP_LENGTH_MS = 10      # 10ms hop (standard overlap)

# Minimum voiced frame ratio to trust pitch estimate
MIN_PITCH_CONFIDENCE = 0.15  # At least 15% of frames should have pitch

# Autocorrelation peak threshold (relative to zero-lag)
AUTOCORR_PEAK_THRESHOLD = 0.3

# Energy threshold for voiced frame detection
VOICED_ENERGY_THRESHOLD = 0.01

# Speaking rate boundaries (syllables per second)
SPEAKING_RATE_FAST = 5.0    # > 5 syl/s = younger
SPEAKING_RATE_MODERATE = 4.0
SPEAKING_RATE_SLOW = 3.0    # < 3 syl/s = older

# Spectral centroid boundaries (Hz)
CENTROID_BRIGHT = 1500      # > 1500 Hz = younger voice
CENTROID_MODERATE = 1200
CENTROID_DARK = 900         # < 900 Hz = older voice

# Jitter boundaries (ratio)
JITTER_LOW = 0.015          # < 0.015 = stable pitch (younger)
JITTER_MODERATE = 0.025
JITTER_HIGH = 0.04          # > 0.04 = unstable pitch (older)


@dataclass
class InferenceResult:
    """Container for inference results."""
    gender_prediction: str
    gender_confidence: float
    gender_scores: Dict[str, float]
    age_bracket: str
    age_confidence: float
    age_raw: float
    inference_ms: float


# =============================================================================
# SHARED PITCH COMPUTATION (used by pitch AND jitter)
# =============================================================================

def _compute_frame_pitches(audio: np.ndarray, sample_rate: int = 16000) -> Tuple[List[float], int, int]:
    """
    Compute pitch for all voiced frames using autocorrelation.

    This is the SHARED computation used by both:
    - estimate_pitch() → returns median pitch
    - estimate_jitter() → returns pitch variability

    Algorithm: Autocorrelation-based pitch detection
    - For each frame, compute self-correlation
    - Find peak in lag range corresponding to [PITCH_MIN_HZ, PITCH_MAX_HZ]
    - Peak at lag L means period = L samples → frequency = sample_rate / L

    Returns:
        tuple: (list of pitches, voiced_frame_count, total_frame_count)
    """
    frame_length = int(FRAME_LENGTH_MS / 1000 * sample_rate)
    hop_length = int(HOP_LENGTH_MS / 1000 * sample_rate)

    pitches = []
    voiced_frames = 0
    total_frames = 0

    # Precompute lag bounds
    min_lag = int(sample_rate / PITCH_MAX_HZ)  # 400 Hz → ~40 samples at 16kHz
    max_lag = int(sample_rate / PITCH_MIN_HZ)  # 50 Hz → ~320 samples at 16kHz

    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        total_frames += 1

        # Check if frame has enough energy (voiced speech)
        energy = np.sqrt(np.mean(frame ** 2))
        if energy < VOICED_ENERGY_THRESHOLD:
            continue

        # Autocorrelation for pitch detection
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]  # Keep only positive lags

        if max_lag > len(corr):
            continue

        # Search for peak in the valid pitch range
        search_region = corr[min_lag:max_lag]
        if len(search_region) == 0:
            continue

        peak_idx = np.argmax(search_region) + min_lag

        # Validate peak: must be strong relative to zero-lag (unvoiced has no peak)
        if peak_idx > 0 and corr[peak_idx] > AUTOCORR_PEAK_THRESHOLD * corr[0]:
            pitch = sample_rate / peak_idx
            if PITCH_MIN_HZ < pitch < PITCH_MAX_HZ:
                pitches.append(pitch)
                voiced_frames += 1

    return pitches, voiced_frames, total_frames


# =============================================================================
# ACOUSTIC FEATURE EXTRACTION
# =============================================================================

def estimate_pitch(audio: np.ndarray, sample_rate: int = 16000,
                   precomputed: Optional[Tuple[List[float], int, int]] = None) -> Tuple[float, float]:
    """
    Estimate fundamental frequency (F0) using autocorrelation.

    F0 is the primary acoustic feature for gender detection:
    - Male: typically 85-180 Hz (average ~120 Hz)
    - Female: typically 165-255 Hz (average ~200 Hz)

    Args:
        audio: Audio samples
        sample_rate: Sample rate (default 16000)
        precomputed: Optional precomputed (pitches, voiced_frames, total_frames)
                     to avoid redundant computation

    Returns:
        tuple: (pitch_hz, confidence) where confidence indicates
               how reliable the pitch estimate is (0-1)
    """
    if precomputed is not None:
        pitches, voiced_frames, total_frames = precomputed
    else:
        pitches, voiced_frames, total_frames = _compute_frame_pitches(audio, sample_rate)

    # Calculate confidence based on how many frames had detectable pitch
    if total_frames > 0:
        pitch_confidence = voiced_frames / total_frames
    else:
        pitch_confidence = 0.0

    if len(pitches) > 0:
        # Use median to be robust to outliers
        return float(np.median(pitches)), pitch_confidence

    return 0.0, 0.0  # No pitch detected = no speech


def estimate_speaking_rate(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Estimate speaking rate using syllable-like energy peaks.

    Younger speakers tend to speak faster (~5+ syl/s).
    Older speakers tend to speak slower (~3 syl/s).

    Algorithm:
    1. Compute energy envelope (25ms frames, 10ms hop)
    2. Smooth to remove micro-variations
    3. Count peaks above threshold = syllable nuclei
    4. syllable_rate = peaks / duration

    Returns:
        Syllables per second (typical range: 2-7)
    """
    # Envelope detection (25ms frames for syllable-level resolution)
    frame_length = int(0.025 * sample_rate)
    hop_length = int(HOP_LENGTH_MS / 1000 * sample_rate)

    # Vectorized envelope computation
    num_frames = (len(audio) - frame_length) // hop_length
    if num_frames < 10:
        return 4.0  # Default syllables per second

    envelope = np.array([
        np.sqrt(np.mean(audio[i:i + frame_length] ** 2))
        for i in range(0, num_frames * hop_length, hop_length)
    ])

    # Smooth the envelope (5-point moving average)
    kernel_size = 5
    envelope = np.convolve(envelope, np.ones(kernel_size) / kernel_size, mode='same')

    # Find peaks (syllable nuclei) above threshold
    threshold = np.mean(envelope) * 0.5
    peaks = 0
    for i in range(1, len(envelope) - 1):
        if envelope[i] > envelope[i - 1] and envelope[i] > envelope[i + 1]:
            if envelope[i] > threshold:
                peaks += 1

    duration = len(audio) / sample_rate
    syllable_rate = peaks / duration if duration > 0 else 4.0

    return float(syllable_rate)


def estimate_spectral_centroid(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """
    Estimate spectral centroid (brightness of voice) across FULL audio.

    Higher values often correlate with younger voices.
    - Younger: > 1500 Hz (brighter, more energy in upper frequencies)
    - Older: < 900 Hz (darker, less high-frequency content)

    Algorithm:
    1. Compute FFT for overlapping windows across entire audio
    2. Calculate centroid for each window
    3. Return median (robust to outliers from silence/noise)

    FIX: Previous version only used first 2048 samples (~128ms).
    Now uses windowed analysis across full audio.

    Returns:
        Spectral centroid in Hz (typical range: 500-2500)
    """
    n_fft = 2048
    hop = n_fft // 2  # 50% overlap

    # Need at least one full window
    if len(audio) < n_fft:
        # Short audio: use what we have
        spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
        if np.sum(spectrum) > 0:
            return float(np.sum(freqs * spectrum) / np.sum(spectrum))
        return 1000.0

    # Compute centroid for each window
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    centroids = []

    for i in range(0, len(audio) - n_fft, hop):
        window = audio[i:i + n_fft]

        # Apply Hann window to reduce spectral leakage
        window = window * np.hanning(n_fft)

        spectrum = np.abs(np.fft.rfft(window))
        total_energy = np.sum(spectrum)

        # Only include windows with sufficient energy (voiced segments)
        if total_energy > 0.01:
            centroid = np.sum(freqs * spectrum) / total_energy
            centroids.append(centroid)

    if len(centroids) > 0:
        # Use median to be robust to outliers
        return float(np.median(centroids))

    return 1000.0  # Default if no valid windows


def estimate_jitter(audio: np.ndarray, sample_rate: int = 16000,
                    precomputed: Optional[Tuple[List[float], int, int]] = None) -> float:
    """
    Estimate jitter (pitch variability) using SHARED pitch computation.

    Higher jitter often correlates with older voices:
    - Young: < 0.015 (stable pitch)
    - Middle: 0.015-0.025
    - Older: > 0.04 (unstable pitch)

    Jitter = mean(|pitch[i+1] - pitch[i]|) / mean(pitch)

    Args:
        audio: Audio samples
        sample_rate: Sample rate (default 16000)
        precomputed: Optional precomputed (pitches, voiced_frames, total_frames)
                     to avoid redundant computation

    Returns:
        Jitter ratio (typical range: 0.01-0.1)
    """
    if precomputed is not None:
        pitches, _, _ = precomputed
    else:
        pitches, _, _ = _compute_frame_pitches(audio, sample_rate)

    if len(pitches) < 3:
        return 0.02  # Default (not enough data)

    # Jitter = average absolute difference between consecutive pitches
    # normalized by mean pitch
    pitches_arr = np.array(pitches)
    diffs = np.abs(np.diff(pitches_arr))
    mean_pitch = np.mean(pitches_arr)

    if mean_pitch > 0:
        jitter = np.mean(diffs) / mean_pitch
    else:
        jitter = 0.02

    return float(jitter)


# =============================================================================
# INFERENCE ENGINE
# =============================================================================

class InferenceEngine:
    """
    Age and gender prediction using acoustic features.

    This uses proven acoustic correlates:
    - Gender: Fundamental frequency (F0/pitch)
    - Age: Speaking rate, spectral features, jitter

    While not as accurate as deep learning models, this approach:
    - Has no external dependencies to download
    - Runs very fast (<100ms)
    - Provides reasonable baseline predictions
    """

    def __init__(self):
        logger.info("Initializing acoustic feature-based inference engine...")
        self._model_type = "acoustic"
        logger.info("Inference engine ready (acoustic features)")

    def predict(self, audio: np.ndarray, sample_rate: int = 16000) -> InferenceResult:
        """
        Run inference on audio to predict age and gender.

        OPTIMIZATION: Pitch computation is done ONCE and shared between
        estimate_pitch() and estimate_jitter() to avoid redundant work.

        Args:
            audio: 16kHz mono float32 numpy array
            sample_rate: Must be 16000

        Returns:
            InferenceResult with gender and age predictions
        """
        start_time = time.perf_counter()

        if sample_rate != 16000:
            raise ValueError(f"Expected 16kHz, got {sample_rate}Hz")

        # === SHARED COMPUTATION: Compute frame pitches ONCE ===
        # This is used by both estimate_pitch() and estimate_jitter()
        precomputed_pitches = _compute_frame_pitches(audio, sample_rate)

        # === EXTRACT ACOUSTIC FEATURES ===
        pitch, pitch_confidence = estimate_pitch(audio, sample_rate, precomputed=precomputed_pitches)
        speaking_rate = estimate_speaking_rate(audio, sample_rate)
        spectral_centroid = estimate_spectral_centroid(audio, sample_rate)
        jitter = estimate_jitter(audio, sample_rate, precomputed=precomputed_pitches)

        logger.debug(
            f"Features: pitch={pitch:.1f}Hz (conf={pitch_confidence:.2f}), "
            f"rate={speaking_rate:.1f}syl/s, centroid={spectral_centroid:.0f}Hz, "
            f"jitter={jitter:.3f}"
        )

        # === CHECK FOR SPEECH ===
        # If pitch confidence is too low, there's likely no speech
        if pitch_confidence < MIN_PITCH_CONFIDENCE or pitch == 0:
            logger.info("No speech detected - returning unknown predictions")
            return InferenceResult(
                gender_prediction="unknown",
                gender_confidence=0.0,
                gender_scores={"male": 0.5, "female": 0.5},
                age_bracket="unknown",
                age_confidence=0.0,
                age_raw=0.0,
                inference_ms=0
            )

        # === GENDER PREDICTION FROM PITCH ===
        # Uses named constants for decision boundaries
        if pitch < PITCH_DEFINITELY_MALE:
            female_prob = 0.1
        elif pitch < PITCH_LIKELY_MALE:
            female_prob = 0.3
        elif pitch < PITCH_AMBIGUOUS_HIGH:
            female_prob = 0.5
        elif pitch < PITCH_LIKELY_FEMALE:
            female_prob = 0.7
        else:
            female_prob = 0.9

        # Confidence based on pitch reliability and distance from boundary
        base_confidence = abs(female_prob - 0.5) * 2  # 0 to 1
        gender_confidence = (0.5 + base_confidence * 0.4) * pitch_confidence

        # === AGE ESTIMATION FROM MULTIPLE FEATURES ===
        # NOTE: Acoustic-only age estimation is limited in accuracy.
        # This uses simplified heuristics calibrated toward moderate ages.

        # Speaking rate: younger people tend to speak faster
        if speaking_rate > SPEAKING_RATE_FAST:
            age_from_rate = 25  # Fast = younger
        elif speaking_rate > SPEAKING_RATE_MODERATE:
            age_from_rate = 32
        elif speaking_rate > SPEAKING_RATE_SLOW:
            age_from_rate = 40
        else:
            age_from_rate = 50  # Slow = older

        # Spectral centroid: younger voices tend to be brighter
        if spectral_centroid > CENTROID_BRIGHT:
            age_from_centroid = 25
        elif spectral_centroid > CENTROID_MODERATE:
            age_from_centroid = 32
        elif spectral_centroid > CENTROID_DARK:
            age_from_centroid = 40
        else:
            age_from_centroid = 50

        # Jitter: increases with age
        if jitter < JITTER_LOW:
            age_from_jitter = 25
        elif jitter < JITTER_MODERATE:
            age_from_jitter = 35
        elif jitter < JITTER_HIGH:
            age_from_jitter = 45
        else:
            age_from_jitter = 55

        # Combine estimates (weighted average)
        age_raw = (
            age_from_rate * 0.35 +
            age_from_centroid * 0.35 +
            age_from_jitter * 0.30
        )

        # Bias toward middle brackets (most common in call centers)
        age_raw = age_raw * 0.8 + 35 * 0.2  # Pull toward 35

        # Clamp to reasonable range
        age_raw = max(20, min(70, age_raw))

        # === BUILD RESULT ===
        result = self._postprocess(age_raw, female_prob)
        result.inference_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Inference: gender={result.gender_prediction} "
            f"({result.gender_confidence:.2f}), "
            f"age={result.age_bracket} ({result.age_raw:.1f}y), "
            f"pitch={pitch:.0f}Hz, took={result.inference_ms:.0f}ms"
        )

        return result

    def _postprocess(self, age_raw: float, female_prob: float) -> InferenceResult:
        """Convert raw predictions to structured result."""
        # Gender prediction
        if female_prob > 0.5:
            gender_prediction = "female"
            gender_confidence = female_prob
        else:
            gender_prediction = "male"
            gender_confidence = 1 - female_prob

        gender_scores = {
            "male": round(1 - female_prob, 3),
            "female": round(female_prob, 3)
        }

        if gender_confidence < MIN_CONFIDENCE_THRESHOLD:
            gender_prediction = "unknown"

        # Age bracket
        age_bracket = "unknown"
        for min_age, max_age, bracket_name in AGE_BRACKETS:
            if min_age <= age_raw < max_age:
                age_bracket = bracket_name
                break

        age_confidence = self._calc_age_confidence(age_raw)
        if age_confidence < MIN_CONFIDENCE_THRESHOLD:
            age_bracket = "unknown"

        return InferenceResult(
            gender_prediction=gender_prediction,
            gender_confidence=gender_confidence,
            gender_scores=gender_scores,
            age_bracket=age_bracket,
            age_confidence=age_confidence,
            age_raw=age_raw,
            inference_ms=0
        )

    def _calc_age_confidence(self, age: float) -> float:
        """Calculate confidence based on distance from bracket center."""
        for min_age, max_age, _ in AGE_BRACKETS:
            if min_age <= age < max_age:
                center = (min_age + max_age) / 2
                width = max_age - min_age
                dist = abs(age - center)
                return max(0.5, 0.85 - (dist / width) * 0.35)
        return 0.5


# =============================================================================
# GLOBAL ENGINE
# =============================================================================

_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        raise RuntimeError("Engine not initialized. Call init_engine() first.")
    return _engine


def init_engine() -> InferenceEngine:
    global _engine
    _engine = InferenceEngine()
    return _engine
