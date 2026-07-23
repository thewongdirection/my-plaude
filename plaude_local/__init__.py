"""plaude-local: a local, offline speech-to-text CLI.

A small tool in the spirit of Plaud: point it at a WAV or MP3 recording and get
a transcript back, running entirely on your own machine (CUDA GPU or CPU) with
no cloud calls.

Pipeline:  decode/denoise (FFmpeg)  ->  transcribe (faster-whisper)
           ->  optional speaker diarization (pyannote)
"""

__version__ = "0.1.0"
