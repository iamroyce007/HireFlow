import os
import io
import time
import asyncio
import numpy as np

# Lazy imports — these are heavy/optional dependencies
whisper = None
AudioSegment = None

def _ensure_whisper():
    global whisper
    if whisper is None:
        import whisper as _whisper
        whisper = _whisper

def _ensure_pydub():
    global AudioSegment
    if AudioSegment is None:
        from pydub import AudioSegment as _AS
        AudioSegment = _AS

class STTService:
    """Speech-to-Text using optimized local Whisper model."""
    _model = None

    def __init__(self, api_key: str = None):
        # Lazy load to avoid blocking startup; using 'base' for speed
        if STTService._model is None:
            try:
                _ensure_whisper()
                print("[STTService] Loading Whisper 'base' model...")
                STTService._model = whisper.load_model("base")
                print("[STTService] Whisper model loaded.")
            except Exception as e:
                print(f"[STTService] Whisper not available: {e}. STT will return empty transcripts.")

    def preprocess_audio(self, audio_bytes: bytes) -> np.ndarray:
        """
        Convert WAV to 16kHz Mono numpy array.
        Bypasses ffmpeg by using scipy.io.wavfile for WAV data.
        """
        try:
            import io
            from scipy.io import wavfile
            
            # 1. Try reading as WAV directly (bypasses ffmpeg)
            try:
                wav_io = io.BytesIO(audio_bytes)
                rate, samples = wavfile.read(wav_io)
                
                # Convert to float32
                if samples.dtype == np.int16:
                    samples = samples.astype(np.float32) / 32768.0
                elif samples.dtype == np.int32:
                    samples = samples.astype(np.float32) / 2147483648.0
                
                # Resample if not 16kHz
                if rate != 16000:
                    from scipy.signal import resample
                    num_samples = int(len(samples) * 16000 / rate)
                    samples = resample(samples, num_samples)
                
                # Mono conversion
                if len(samples.shape) > 1:
                    samples = np.mean(samples, axis=1)
                
                return samples.astype(np.float32)
            except Exception as wav_err:
                print(f"[STTService] Native WAV read failed, falling back to pydub: {wav_err}")

            # 2. Fallback to Pydub (requires ffmpeg)
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            samples = np.array(audio_segment.get_array_of_samples())
            
            if audio_segment.sample_width == 2:
                samples = samples.astype(np.float32) / 32768.0
            elif audio_segment.sample_width == 4:
                samples = samples.astype(np.float32) / 2147483648.0
                
            return samples
        except Exception as e:
            print(f"[STTService] Preprocessing error: {e}")
            return np.array([])

    async def transcribe_async(self, audio_bytes: bytes) -> dict:
        """Execute transcription off the main thread."""
        return await asyncio.to_thread(self._transcribe_sync, audio_bytes)

    def _transcribe_sync(self, audio_bytes: bytes) -> dict:
        if not audio_bytes or len(audio_bytes) < 100:
            return {"text": "", "language": "unknown", "word_count": 0, "has_speech": False, "error": None}

        start_time = time.time()
        
        # 1. Preprocess: mono + 16kHz
        audio_array = self.preprocess_audio(audio_bytes)
        
        if len(audio_array) == 0:
            return {"text": "", "language": "unknown", "word_count": 0, "has_speech": False, "error": "Invalid Audio Format or FFmpeg missing"}

        try:
            # 2. Transcribe (fp16=False for compatibility on CPU)
            result = STTService._model.transcribe(audio_array, fp16=False)
            text = result.get("text", "").strip()
            
            latency = time.time() - start_time
            print(f"[Timing] Whisper STT Latency: {latency:.2f} seconds")
            
            has_speech = len(text) > 2
            
            return {
                "text": text,
                "language": result.get("language", "en"),
                "word_count": len(text.split()),
                "has_speech": has_speech,
                "latency": latency,
                "error": None
            }
        except Exception as e:
            print(f"[STTService] Transcription failed: {str(e)}")
            return {"text": "", "language": "unknown", "word_count": 0, "has_speech": False, "error": str(e)}
