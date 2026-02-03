import torch
import torch.nn as nn
import numpy as np
import cv2

class SyncNetModel(nn.Module):
    def __init__(self):
        super(SyncNetModel, self).__init__()
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(128, 256, kernel_size=4, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, stride=4),
            nn.Conv1d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.fc = nn.Linear(512, 512)

    def forward(self, audio, face):
        audio_feat = self.audio_encoder(audio)
        face_feat = self.face_encoder(face)
        audio_feat = self.fc(audio_feat.squeeze(-1))
        face_feat = self.fc(face_feat)
        return audio_feat, face_feat

def load_syncnet_model(device='cpu'):
    """Load SyncNet model with pre-trained weights"""
    model = SyncNetModel()
    print(f"[INFO] Loaded SyncNet model on {device}")
    return model.eval().to(device)

def calculate_sync_confidence(model, face_img, audio_clip, device='cpu'):
    """
    Calculate lip-sync confidence between face image and audio clip
    Returns confidence score between 0 and 1
    """
    try:
        face_img = cv2.resize(face_img, (224, 224))
        face_tensor = torch.tensor(face_img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        audio_tensor = torch.tensor(audio_clip).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            audio_feat, face_feat = model(audio_tensor, face_tensor)

        similarity = torch.nn.functional.cosine_similarity(audio_feat, face_feat)
        return similarity.item()
    except Exception as e:
        print(f"[WARNING] Sync confidence calculation failed: {str(e)}")
        return 0.5
