import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    FACE_SIMILARITY_THRESHOLD = 0.6
    LIP_MOTION_WEIGHT = 0.7
    SYNC_CONFIDENCE_WEIGHT = 0.3
    SEGMENT_MERGE_THRESHOLD = 1.0
    GAUSSIAN_SIGMA = 3

    WHISPER_MODEL_SIZE = "base"
    DEFAULT_DEVICE = "cpu"

    HF_MODEL_NAME = "NousResearch/Hermes-3-Llama-3.1-8B"

    def __init__(self):
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN", "")
        self.pyannotate_token = os.getenv("PYANNOTE_TOKEN", "")
        self.repo_id = os.getenv("HF_REPO_ID", "")

    def validate(self):
        """Validate that required environment variables are set"""
        errors = []
        if not self.hf_token:
            errors.append("HUGGING_FACE_TOKEN not set")
        if not self.pyannotate_token:
            errors.append("PYANNOTE_TOKEN not set")
        if not self.repo_id:
            errors.append("HF_REPO_ID not set")
        return errors
