"""
Audio Processor Module — Real Signal Analysis
Processes raw audio data to extract pitch, energy, stability, and confidence metrics.
Uses numpy + scipy for actual DSP — no mocks.
"""

import numpy as np
from scipy.signal import find_peaks, stft
from scipy.io import wavfile
import io
import struct
from typing import Dict, Tuple


class AudioProcessor:
    """Processes raw audio bytes for confidence scoring via real DSP."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def decode_audio(self, audio_bytes: bytes) -> np.ndarray:
        """
        Decode raw audio bytes into a numpy float32 array.
        Handles WAV format and raw PCM.
        """
        try:
            # Try WAV format first
            wav_io = io.BytesIO(audio_bytes)
            rate, samples = wavfile.read(wav_io)
            self.sample_rate = rate
            if samples.dtype == np.int16:
                samples = samples.astype(np.float32) / 32768.0
            elif samples.dtype == np.int32:
                samples = samples.astype(np.float32) / 2147483648.0
            elif samples.dtype != np.float32:
                samples = samples.astype(np.float32)
            # Convert stereo to mono
            if len(samples.shape) > 1:
                samples = np.mean(samples, axis=1)
            return samples
        except Exception:
            # Fallback: raw PCM int16
            try:
                samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                return samples
            except Exception:
                return np.array([], dtype=np.float32)

    def compute_rms_energy(self, samples: np.ndarray) -> float:
        """Compute Root Mean Square energy (loudness)."""
        if len(samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples ** 2)))

    def compute_energy_over_time(self, samples: np.ndarray, frame_size: int = 2048, hop: int = 512) -> np.ndarray:
        """Compute energy per frame for variability analysis."""
        if len(samples) < frame_size:
            return np.array([self.compute_rms_energy(samples)])
        n_frames = (len(samples) - frame_size) // hop + 1
        energies = np.zeros(n_frames)
        for i in range(n_frames):
            frame = samples[i * hop: i * hop + frame_size]
            energies[i] = np.sqrt(np.mean(frame ** 2))
        return energies

    def compute_pitch_autocorrelation(self, samples: np.ndarray) -> Tuple[float, float]:
        """
        Estimate fundamental frequency (pitch) using autocorrelation method.
        Returns (pitch_hz, pitch_confidence).
        """
        if len(samples) < 1024:
            return 0.0, 0.0

        # Window to reduce artifacts
        windowed = samples * np.hanning(len(samples))

        # Autocorrelation via FFT (fast)
        n = len(windowed)
        fft = np.fft.rfft(windowed, n=2 * n)
        autocorr = np.fft.irfft(fft * np.conj(fft))[:n]

        # Normalize
        if autocorr[0] != 0:
            autocorr = autocorr / autocorr[0]

        # Find first peak after initial trough (skip lag 0)
        # Human pitch range: 80-400 Hz → lag range
        min_lag = int(self.sample_rate / 400)  # ~110 for 44100 Hz
        max_lag = int(self.sample_rate / 80)   # ~551 for 44100 Hz

        if max_lag > len(autocorr):
            max_lag = len(autocorr) - 1
        if min_lag >= max_lag:
            return 0.0, 0.0

        search_region = autocorr[min_lag:max_lag]
        if len(search_region) == 0:
            return 0.0, 0.0

        peak_idx = np.argmax(search_region) + min_lag
        pitch_confidence = float(autocorr[peak_idx])
        pitch_hz = self.sample_rate / peak_idx if peak_idx > 0 else 0.0

        return float(pitch_hz), max(0.0, pitch_confidence)

    def compute_pitch_over_time(self, samples: np.ndarray, frame_size: int = 4096, hop: int = 2048) -> np.ndarray:
        """Compute pitch per frame for stability analysis."""
        if len(samples) < frame_size:
            pitch, _ = self.compute_pitch_autocorrelation(samples)
            return np.array([pitch])

        n_frames = (len(samples) - frame_size) // hop + 1
        pitches = np.zeros(n_frames)
        for i in range(n_frames):
            frame = samples[i * hop: i * hop + frame_size]
            pitches[i], _ = self.compute_pitch_autocorrelation(frame)
        return pitches

    def compute_zero_crossing_rate(self, samples: np.ndarray) -> float:
        """Zero crossing rate — higher means more noise or consonant sounds."""
        if len(samples) < 2:
            return 0.0
        crossings = np.sum(np.abs(np.diff(np.sign(samples))) > 0)
        return float(crossings / len(samples))

    def compute_spectral_centroid(self, samples: np.ndarray) -> float:
        """
        Spectral centroid — 'brightness' of the sound.
        Confident speech tends to have a moderate, stable centroid.
        """
        if len(samples) < 256:
            return 0.0
        spectrum = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), d=1.0 / self.sample_rate)
        if np.sum(spectrum) == 0:
            return 0.0
        centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum))
        return centroid

    def analyze(self, audio_bytes: bytes) -> Dict:
        """
        Full audio analysis pipeline.
        Strictly monitors for background noise and vocal clarity.
        """
        samples = self.decode_audio(audio_bytes)

        # ─── STRICT NOISE DETECTION ───
        # Production threshold: > 0.15 RMS while silent/wait is generally unacceptable
        rms = self.compute_rms_energy(samples)
        noise_violation = False
        noise_reason = None
        
        # Simple thresholding: if no speech is present but RMS is high, it's environmental noise.
        # If speech is present but SNR is poor, it's also a violation in strict mode.
        if rms > 0.08 and len(samples) > 2000:
            # Check ZCR - high ZCR usually means hiss/static/fans
            zcr = self.compute_zero_crossing_rate(samples)
            if zcr > 0.3:
                noise_violation = True
                noise_reason = "Excessive background hiss or fan noise detected."
            elif rms > 0.2:
                noise_violation = True
                noise_reason = "Sudden loud noise or persistent background chatter detected."

        if len(samples) < 256:
            return {
                "rms_energy": 0.0,
                "pitch_hz": 0.0,
                "pitch_stability": 0.0,
                "energy_stability": 0.0,
                "zero_crossing_rate": 0.0,
                "spectral_centroid": 0.0,
                "confidence_score": 0.0,
                "duration_seconds": 0.0,
                "insight": "Insufficient audio data for analysis.",
                "has_speech": False
            }

        duration = len(samples) / self.sample_rate
        rms = self.compute_rms_energy(samples)
        pitch, pitch_conf = self.compute_pitch_autocorrelation(samples)
        zcr = self.compute_zero_crossing_rate(samples)
        centroid = self.compute_spectral_centroid(samples)

        # Time-series analysis for stability
        energies = self.compute_energy_over_time(samples)
        pitches = self.compute_pitch_over_time(samples)

        # Filter out zero pitches (unvoiced frames)
        voiced_pitches = pitches[pitches > 50]
        pitch_stability = float(np.std(voiced_pitches)) if len(voiced_pitches) > 1 else 0.0
        energy_stability = float(np.std(energies)) if len(energies) > 1 else 0.0

        # Detect if there's actual speech (not just silence/noise)
        has_speech = rms > 0.01 and len(voiced_pitches) > 2

        # ─── CONFIDENCE SCORING (real composite) ───
        # Volume score: too quiet = low confidence, moderate = good, too loud = shouting
        vol_score = min(100, max(0, (rms / 0.15) * 100))
        if rms > 0.3:  # penalize shouting
            vol_score = max(40, 100 - (rms - 0.3) * 200)

        # Pitch stability score: lower std dev = more confident, stable voice
        if pitch_stability > 0 and has_speech:
            stab_score = max(0, min(100, 100 - (pitch_stability / 50) * 100))
        else:
            stab_score = 50.0

        # Speaking rate score (ZCR proxy): moderate = good
        zcr_score = max(0, min(100, 100 - abs(zcr - 0.05) * 2000))

        # Pitch presence score: having a clear pitch = confident
        presence_score = min(100, pitch_conf * 100) if has_speech else 20.0

        # Weighted composite
        confidence = (
            vol_score * 0.25 +
            stab_score * 0.30 +
            zcr_score * 0.15 +
            presence_score * 0.30
        )
        confidence = round(max(0, min(100, confidence)), 1)

        # Generate insight
        if not has_speech:
            insight = "No clear speech detected in audio."
        elif confidence >= 80:
            insight = "Strong, confident speaking voice with stable pitch and good volume."
        elif confidence >= 60:
            insight = "Steady presentation with moderate confidence. Some variation in tone."
        elif confidence >= 40:
            insight = "Noticeable hesitation or uneven pacing. Volume could be stronger."
        else:
            insight = "Appears nervous — low volume, unstable pitch, or significant pausing."

        mean_pitch = float(np.mean(voiced_pitches)) if len(voiced_pitches) > 0 else 0.0

        return {
            "rms_energy": round(float(rms), 4),
            "pitch_hz": round(mean_pitch, 1),
            "pitch_stability": round(pitch_stability, 2),
            "energy_stability": round(energy_stability, 4),
            "zero_crossing_rate": round(zcr, 4),
            "spectral_centroid": round(centroid, 1),
            "confidence_score": confidence,
            "duration_seconds": round(duration, 2),
            "insight": insight,
            "has_speech": has_speech,
            "volume_score": round(vol_score, 1),
            "stability_score": round(stab_score, 1),
            "presence_score": round(presence_score, 1),
        }
