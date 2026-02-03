import os
import subprocess
import cv2
import numpy as np
import pandas as pd
import shutil
import re
import librosa
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

from face_analyzer import FaceAnalyzer
from models import load_syncnet_model, calculate_sync_confidence
from config import Config

class VideoProcessor:
    def __init__(self, status_callback=None, device='cpu'):
        self.status_callback = status_callback or (lambda msg: None)
        self.face_analyzer = FaceAnalyzer()
        self.syncnet = load_syncnet_model(device=device)
        self.device = device
        self.config = Config()

    def update_status(self, message):
        self.status_callback(message)
        print(f"[STATUS] {message}")

    def extract_keyframes(self, video_path, output_dir):
        """Extract keyframes from video"""
        self.update_status("Extracting keyframes...")
        os.makedirs(output_dir, exist_ok=True)

        try:
            cmd = f"""
            ffmpeg -i "{video_path}" \
            -vf "select=eq(pict_type\\,I),showinfo" -vsync vfr \
            "{os.path.join(output_dir, 'frame_%04d.jpg')}" 2>&1 | \
            grep "showinfo" | \
            awk -F 'pts_time:' '{{print $2}}' | \
            awk -F ' ' '{{print $1}}' > "{os.path.join(output_dir, 'timestamps.txt')}"
            """
            subprocess.run(cmd, shell=True, executable='/bin/bash', check=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
            timestamps_file = os.path.join(output_dir, 'timestamps.txt')

            if not os.path.exists(timestamps_file) or os.path.getsize(timestamps_file) == 0:
                raise ValueError("Failed to extract frame timestamps")

            timestamps = pd.read_csv(timestamps_file, header=None, names=['timestamp'])

            if len(frame_files) != len(timestamps):
                min_len = min(len(frame_files), len(timestamps))
                frame_files = frame_files[:min_len]
                timestamps = timestamps.head(min_len)

            frame_data = pd.DataFrame({
                'frame_file': frame_files,
                'timestamp': timestamps['timestamp']
            })

            frame_data['start'] = frame_data['timestamp']
            frame_data['end'] = frame_data['timestamp'].shift(-1)
            frame_data = frame_data.dropna(subset=['end'])

            return frame_data
        except Exception as e:
            raise RuntimeError(f"Keyframe extraction failed: {str(e)}")

    def match_faces(self, video_path, reference_img, output_base):
        """Match faces in video against reference image"""
        self.update_status("Initializing face recognition...")

        keyframe_dir = os.path.join(output_base, "keyframes")
        matched_frames_dir = os.path.join(output_base, "matched_frames")
        os.makedirs(keyframe_dir, exist_ok=True)
        os.makedirs(matched_frames_dir, exist_ok=True)

        frame_data = self.extract_keyframes(video_path, keyframe_dir)
        timestamp_file = os.path.join(output_base, "frame_timestamps.csv")
        intervals_file = os.path.join(output_base, "frame_intervals.csv")
        frame_data.to_csv(timestamp_file, index=False)
        frame_data.to_csv(intervals_file, index=False)

        ref_img = cv2.imread(reference_img)
        if ref_img is None:
            raise FileNotFoundError(f"Reference image not found: {reference_img}")

        ref_faces = self.face_analyzer.app.get(ref_img)
        if len(ref_faces) == 0:
            raise ValueError("No face found in reference image")
        ref_embedding = ref_faces[0].embedding

        self.update_status("Matching faces...")
        threshold = self.config.FACE_SIMILARITY_THRESHOLD
        matches_found = 0

        for filename in tqdm(sorted(os.listdir(keyframe_dir))):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                frame_path = os.path.join(keyframe_dir, filename)
                try:
                    frame_embedding = self.face_analyzer.get_embedding(frame_path)
                    similarity = self.face_analyzer.cosine_similarity(ref_embedding, frame_embedding)

                    if similarity > threshold:
                        matches_found += 1
                        shutil.copy2(frame_path, os.path.join(matched_frames_dir, filename))
                except Exception as e:
                    print(f"[WARNING] Skipping frame {filename}: {str(e)}")

        self.update_status(f"Found {matches_found} face matches")
        return matched_frames_dir, intervals_file

    def extract_audio(self, video_path, output_dir):
        """Extract audio from video"""
        self.update_status("Extracting audio...")
        os.makedirs(output_dir, exist_ok=True)
        audio_path = os.path.join(output_dir, "full_audio.wav")

        video = None
        try:
            video = VideoFileClip(video_path)
            if video.audio is None:
                raise ValueError("Video has no audio track")
            video.audio.write_audiofile(audio_path, logger=None)
            return audio_path
        except Exception as e:
            raise RuntimeError(f"Audio extraction failed: {str(e)}")
        finally:
            if video is not None:
                video.close()

    def analyze_lip_motion(self, reference_img, matched_frames_dir, audio_path, output_dir):
        """Analyze lip motion and sync confidence"""
        self.update_status("Analyzing lip motion and sync confidence...")
        os.makedirs(output_dir, exist_ok=True)

        lip_indices = self.face_analyzer.get_lip_indices(reference_img)
        audio, sr = librosa.load(audio_path, sr=16000)

        scores = []
        for fname in tqdm(sorted(os.listdir(matched_frames_dir))):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            fpath = os.path.join(matched_frames_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue

            try:
                frame_idx = int(re.search(r'frame_(\d+)', fname).group(1))
                timestamp = frame_idx / 30
            except:
                timestamp = 0.0

            lip_score = self.face_analyzer.analyze_lip_motion_in_frame(img, lip_indices)

            sync_conf = 0.5
            try:
                start_sample = max(0, int((timestamp - 0.25) * sr))
                end_sample = min(len(audio), int((timestamp + 0.25) * sr))
                audio_clip = audio[start_sample:end_sample]

                if len(audio_clip) < 0.5 * sr:
                    padding = int(0.5 * sr) - len(audio_clip)
                    audio_clip = np.pad(audio_clip, (0, padding), mode='constant')

                sync_conf = calculate_sync_confidence(self.syncnet, img, audio_clip, device=self.device)
                sync_conf = (sync_conf + 1) / 2
            except Exception as e:
                print(f"[WARNING] SyncNet error in {fname}: {str(e)}")

            composite_score = (self.config.LIP_MOTION_WEIGHT * lip_score +
                             self.config.SYNC_CONFIDENCE_WEIGHT * sync_conf)

            scores.append({
                "frame": fname,
                "timestamp": timestamp,
                "lip_score": lip_score,
                "sync_confidence": sync_conf,
                "composite_score": composite_score,
            })

        if scores:
            df = pd.DataFrame(scores)
            df = df.sort_values("composite_score", ascending=False)
            output_file = os.path.join(output_dir, "lip_motion_scores.csv")
            df.to_csv(output_file, index=False)
            return output_file, df
        else:
            raise ValueError("No valid lip motion data collected")

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.face_analyzer.cleanup()
        except Exception as e:
            print(f"[WARNING] Cleanup error: {str(e)}")
