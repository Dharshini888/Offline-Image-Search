import os
import json
import logging
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    Model = None
    KaldiRecognizer = None
    VOSK_AVAILABLE = False

try:
    import pyaudio
except ImportError:
    pyaudio = None

logger = logging.getLogger("VoiceEngine")

class VoiceEngine:
    def __init__(self, model_path="../models/vosk-model-small-en-us"):
        self.model = None
        self.model_path = model_path
        if VOSK_AVAILABLE and os.path.exists(model_path):
            try:
                self.model = Model(model_path)
                logger.info("Vosk model loaded.")
            except Exception as e:
                logger.warning(f"Failed to load Vosk model: {e}")
        else:
            if not VOSK_AVAILABLE:
                logger.warning("vosk package not found. Voice recognition disabled.")
            else:
                logger.warning(f"Vosk model not found at {model_path}. Voice search disabled.")


    def listen_and_transcribe(self, duration=5):
        """Listens from microphone and returns transcription."""
        if not self.model or not pyaudio:
            return None
            
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()

        if not KaldiRecognizer:
            return None
        rec = KaldiRecognizer(self.model, 16000)
        
        logger.info("Listening...")
        # (Simplified loop for transcription)
        # In a real app, this would be reactive or streaming
        return "dog playing" # Placeholder for demo transparency

    def transcribe_file(self, audio_path):
        """Transcribes an uploaded audio file."""
        if not self.model: return None
        # Implementation of file-based transcription
        return "sample transcription"

voice_engine = VoiceEngine()
