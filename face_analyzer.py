import cv2
import numpy as np
from insightface.app import FaceAnalysis
import mediapipe as mp

class FaceAnalyzer:
    def __init__(self):
        try:
            self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0)
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
            print("[INFO] FaceAnalyzer initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FaceAnalyzer: {str(e)}")

    def get_embedding(self, img_path):
        """Get face embedding from image"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
            faces = self.app.get(img)
            if len(faces) == 0:
                raise ValueError("No faces detected")
            return faces[0].embedding
        except Exception as e:
            raise ValueError(f"Error getting face embedding: {str(e)}")

    def cosine_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def get_lip_indices(self, img_path):
        """Get lip landmark indices from face image"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.mp_face_mesh.process(rgb)

            if not result.multi_face_landmarks:
                raise ValueError("No face landmarks detected")

            landmarks = result.multi_face_landmarks[0].landmark
            coords = [(i, int(p.x * w), int(p.y * h)) for i, p in enumerate(landmarks)]
            lip_landmarks = list(range(61, 76)) + list(range(78, 88))
            lip_coords = [c for c in coords if c[0] in lip_landmarks]

            if not lip_coords:
                raise ValueError("No lip landmarks detected")

            return {
                "top": min(lip_coords, key=lambda x: x[2])[0],
                "bottom": max(lip_coords, key=lambda x: x[2])[0],
                "left": min(lip_coords, key=lambda x: x[1])[0],
                "right": max(lip_coords, key=lambda x: x[1])[0]
            }
        except Exception as e:
            raise ValueError(f"Error getting lip indices: {str(e)}")

    def analyze_lip_motion_in_frame(self, img, lip_indices):
        """Analyze lip motion in a single frame"""
        try:
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = self.mp_face_mesh.process(rgb)

            if not result.multi_face_landmarks:
                return 0

            landmarks = result.multi_face_landmarks[0].landmark
            top = landmarks[lip_indices['top']]
            bottom = landmarks[lip_indices['bottom']]
            left = landmarks[lip_indices['left']]
            right = landmarks[lip_indices['right']]

            lip_height = abs((bottom.y - top.y) * h)
            lip_width = abs((right.x - left.x) * w)
            lip_score = lip_height + 0.5 * lip_width

            return lip_score
        except Exception as e:
            print(f"[WARNING] Lip motion analysis failed: {str(e)}")
            return 0

    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'mp_face_mesh'):
                self.mp_face_mesh.close()
        except Exception as e:
            print(f"[WARNING] Cleanup error: {str(e)}")
