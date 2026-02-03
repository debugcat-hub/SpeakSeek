import os
import time
import requests
import pandas as pd
from huggingface_hub import upload_file, login

class AudioProcessor:
    def __init__(self, status_callback=None):
        self.status_callback = status_callback or (lambda msg: None)

    def update_status(self, message):
        self.status_callback(message)
        print(f"[STATUS] {message}")

    def diarize_audio(self, audio_path, hf_token, pyannotate_token, repo_id):
        """Perform speaker diarization using PyAnnotate API"""
        if not hf_token or not pyannotate_token or not repo_id:
            raise ValueError("Missing required tokens or repo_id for diarization")

        self.update_status("Uploading audio to Hugging Face...")

        try:
            login(hf_token)
        except Exception as e:
            raise RuntimeError(f"Hugging Face login failed: {str(e)}")

        try:
            filename = os.path.basename(audio_path)
            url = upload_file(
                path_or_fileobj=audio_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset"
            )
            public_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
            self.update_status(f"Audio uploaded to: {public_url}")
        except Exception as e:
            raise RuntimeError(f"Audio upload failed: {str(e)}")

        self.update_status("Starting diarization...")
        payload = {"url": public_url}
        headers = {
            "Authorization": f"Bearer {pyannotate_token}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                "https://api.pyannote.ai/v1/diarize",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            job_id = data.get("jobId")

            if not job_id:
                raise ValueError("No job ID returned from diarization API")

            self.update_status(f"Diarization job started: {job_id}")

            max_attempts = 60
            attempt = 0
            while attempt < max_attempts:
                time.sleep(10)
                attempt += 1

                response = requests.get(
                    f"https://api.pyannote.ai/v1/jobs/{job_id}",
                    headers=headers,
                    timeout=30
                )
                response.raise_for_status()
                job_data = response.json()
                job_status = job_data.get("status")

                self.update_status(f"Diarization status: {job_status} (attempt {attempt}/{max_attempts})")

                if job_status == "succeeded":
                    output = job_data.get("output", {})
                    if "diarization" in output:
                        df = pd.DataFrame(output["diarization"])
                        output_file = "diarization_output.csv"
                        df.to_csv(output_file, index=False)
                        self.update_status("Diarization completed successfully")
                        return output_file
                    else:
                        raise ValueError("Unexpected diarization output format")
                elif job_status in ["failed", "canceled"]:
                    error_msg = job_data.get("error", "Unknown error")
                    raise ValueError(f"Diarization job {job_status}: {error_msg}")

            raise TimeoutError("Diarization job timed out after 10 minutes")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Diarization API request failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Diarization failed: {str(e)}")
