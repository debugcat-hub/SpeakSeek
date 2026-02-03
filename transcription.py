import os
import requests
from faster_whisper import WhisperModel

def transcribe_clips(clip_paths, output_dir, model_size="base", device="cpu"):
    """Transcribe all video clips using faster-whisper"""
    if not clip_paths:
        return {}

    os.makedirs(output_dir, exist_ok=True)

    try:
        model = WhisperModel(model_size, device=device, compute_type="int8")
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model: {str(e)}")

    transcripts = {}
    for clip_path in clip_paths:
        if not os.path.exists(clip_path):
            print(f"[WARNING] Clip not found: {clip_path}")
            continue

        if clip_path.lower().endswith((".mp3", ".wav", ".mp4")):
            try:
                segments, _ = model.transcribe(clip_path)
                transcript = " ".join(segment.text for segment in segments)

                file_name = os.path.basename(clip_path)
                output_path = os.path.join(output_dir, f"{file_name}.txt")

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(transcript)

                transcripts[clip_path] = transcript
                print(f"[INFO] Transcribed: {file_name}")
            except Exception as e:
                print(f"[WARNING] Failed to transcribe {clip_path}: {str(e)}")

    return transcripts

def ask_question(transcripts, question, hf_token, model_name="NousResearch/Hermes-3-Llama-3.1-8B"):
    """Use Hugging Face API to answer questions about the transcripts"""
    if not transcripts:
        return "No transcripts available. Please run the pipeline first."

    if not question or not question.strip():
        return "Please ask a question."

    if not hf_token:
        return "Hugging Face token is required for chat functionality."

    combined_text = ""
    for clip_path, transcript in transcripts.items():
        clip_name = os.path.basename(clip_path)
        combined_text += f"\n\nClip: {clip_name}\nTranscript:\n{transcript}"

    system_prompt = (
        "You are an analyzer that receives transcripts of video clips where speakers are talking.\n"
        "You can answer questions like:\n"
        "- What did the speaker say in clip X about topic Y?\n"
        "- Where did the speaker talk about deadlines or specific topics?\n"
        "- What is the speaker's opinion or mood in the clips?\n"
        "- Which clip contains discussion about X?\n"
        "Always mention which clip(s) contain the relevant information."
    )

    full_prompt = (
        f"<|system|>\n{system_prompt}</s>\n"
        f"<|user|>\nTranscripts:{combined_text}\n\n{question}</s>\n"
        f"<|assistant|>"
    )

    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 512,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No response generated")
            else:
                return "No response generated"
        elif response.status_code == 503:
            return "Model is loading. Please try again in a moment."
        else:
            return f"Error {response.status_code}: {response.text}"
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"
