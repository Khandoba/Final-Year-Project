import os
import subprocess
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class VisionModel(nn.Module):
    def __init__(self, openface_path=None):
        """
        Initializes the OpenFace wrapper and Neural Network for Vision-based deception detection.
        
        Args:
            openface_path (str): The absolute path to FeatureExtraction.exe. 
                                 If None, assumes it is in the system PATH or the current directory.
        """
        super(VisionModel, self).__init__()
        
        self.openface_path = openface_path if openface_path else "FeatureExtraction.exe"
        self.output_dir = os.path.join(os.getcwd(), "openface_output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Best-performing config: 5 AUs, hidden=64, dropout=0.3
        self.lstm = nn.LSTM(input_size=5, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = self.classifier(lstm_out[:, -1, :])
        return out

    def load_weights(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            self.eval()
            print(f"VisionModel loaded weights from {path}")
        else:
            print(f"VisionModel weights not found at {path}, using random initialization.")

    def extract_features(self, video_path):
        """
        Runs OpenFace FeatureExtraction on the given video.
        """
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found.")
            return None

        cmd = [
            self.openface_path,
            "-f", video_path,
            "-out_dir", self.output_dir,
            "-aus", # Extract Action Units
            "-pose" # Extract Head Pose
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            csv_path = os.path.join(self.output_dir, f"{base_name}.csv")
            
            if os.path.exists(csv_path):
                return csv_path
            else:
                print("OpenFace did not generate the expected CSV file.")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"OpenFace execution failed: {e}")
            return None
        except FileNotFoundError:
             print(f"OpenFace executable not found at '{self.openface_path}'. Please verify the path.")
             return None

    def predict_deception(self, video_path):
        """
        Predicts deception probability based on extracted visual features (AUs).
        """
        csv_path = self.extract_features(video_path)
        if not csv_path:
            return 0.5
            
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            
            target_aus = ['AU04_c', 'AU15_c', 'AU45_c', 'AU12_c', 'AU20_c']
            
            # We want all frames, fill missing AUs with 0
            au_data = df[target_aus].fillna(0).values
            
            target_len = 100
            if len(au_data) == 0:
                return 0.5
            elif len(au_data) != target_len:
                old_indices = np.linspace(0, len(au_data)-1, num=len(au_data))
                new_indices = np.linspace(0, len(au_data)-1, num=target_len)
                new_seq = np.zeros((target_len, au_data.shape[1]))
                for i in range(au_data.shape[1]):
                    new_seq[:, i] = np.interp(new_indices, old_indices, au_data[:, i])
                au_data = new_seq
            
            self.eval()
            tensor_feats = torch.from_numpy(au_data).float().unsqueeze(0) # Shape: (1, 100, 5)
            with torch.no_grad():
                score = self.forward(tensor_feats)
                prob = torch.sigmoid(score).item()
                
            return prob
            
        except Exception as e:
            print(f"Error processing Vision CSV: {e}")
            return 0.5
