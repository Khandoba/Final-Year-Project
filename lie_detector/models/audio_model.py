import os
import torch
import torch.nn as nn
import numpy as np

try:
    import librosa
except ImportError:
    print("Warning: librosa not found. Run: pip install librosa")
    librosa = None

N_MFCC = 40
TARGET_LEN = 100

class AudioModel(nn.Module):
    def __init__(self):
        """
        Audio Model using librosa 40-coefficient MFCCs as features.
        MFCCs capture vocal tract shape per frame - richer than openSMILE LLDs
        for detecting stress and deception patterns in speech.
        """
        super(AudioModel, self).__init__()

        self.input_size = N_MFCC  # 40 MFCC coefficients per frame

        # Regularized LSTM - smaller size + high dropout to prevent overfitting on small dataset
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=64, num_layers=1,
                            batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def forward(self, feats):
        if len(feats.shape) == 2:
            feats = feats.unsqueeze(1)
        lstm_out, _ = self.lstm(feats)
        score = self.classifier(lstm_out[:, -1, :])
        return score

    def load_weights(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            self.eval()
            print(f"AudioModel loaded weights from {path}")
        else:
            print(f"AudioModel weights not found at {path}, using random initialization.")

    def extract_features(self, audio_path):
        """
        Extracts 40-coefficient MFCC sequence from an audio file using librosa.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            np.ndarray: shape (n_frames, 40), or None on failure.
        """
        if librosa is None:
            print("librosa not available. Cannot extract features.")
            return None

        if not os.path.exists(audio_path):
            print(f"Error: Audio file {audio_path} not found.")
            return None

        try:
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            # Check if it's just silence/background noise
            rms = np.mean(librosa.feature.rms(y=y))
            if rms < 0.005:  # Threshold for silence
                return None
                
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=512)
            return mfcc.T  # shape: (n_frames, N_MFCC)
        except Exception as e:
            print(f"librosa error: {e}")
            return None

    def predict_deception(self, audio_path):
        """
        Predicts deception probability based on MFCC audio features.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            float: Deception probability (0 to 1).
        """
        self.eval()
        features = self.extract_features(audio_path)

        if features is None or len(features) == 0:
            return 0.5  # Neutral fallback

        try:
            if np.isnan(features).any():
                return 0.5

            if len(features) != TARGET_LEN:
                old_indices = np.linspace(0, len(features) - 1, num=len(features))
                new_indices = np.linspace(0, len(features) - 1, num=TARGET_LEN)
                new_seq = np.zeros((TARGET_LEN, features.shape[1]))
                for i in range(features.shape[1]):
                    new_seq[:, i] = np.interp(new_indices, old_indices, features[:, i])
                features = new_seq

            tensor = torch.from_numpy(features).float().unsqueeze(0)  # (1, 100, 40)
            with torch.no_grad():
                score = self.forward(tensor)
                prob = torch.sigmoid(score).item()
            return prob

        except RuntimeError as e:
            print(f"Torch model error: {e}")
            return 0.5
