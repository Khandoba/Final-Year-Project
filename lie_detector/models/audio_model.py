import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

try:
    import opensmile
except ImportError:
    print("Warning: opensmile package not found. Please install via 'pip install opensmile'.")

class AudioModel(nn.Module):
    def __init__(self):
        """
        Initializes the Audio Model using openSMILE features and a simple neural network.
        eGeMAPS is specifically designed for affective computing and stress/emotion detection.
        """
        super(AudioModel, self).__init__()
        
        # openSMILE eGeMAPSv02 feature set outputs 88 features at the functional level
        self.input_size = 88
        
        self.fc1 = nn.Linear(self.input_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
        try:
            # We want functionally summarized features over the entire segment (chunk)
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        except NameError:
            self.smile = None

    def forward(self, feats):
        x = self.relu(self.fc1(feats))
        score = self.fc2(x)
        return score

    def extract_features(self, audio_path):
        """
        Extracts openSMILE features from an audio file.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            np.ndarray: A 1D array of extracted acoustic features, or None on failure.
        """
        if not self.smile:
            print("openSMILE initialization failed. Cannot extract features.")
            return None
            
        if not os.path.exists(audio_path):
            print(f"Error: Audio file {audio_path} not found.")
            return None
            
        try:
            # Output is a pandas DataFrame, where columns are feature names.
            df = self.smile.process_file(audio_path)
            
            # Since feature_level is Functionals, there should be exactly one row representing the entire clip.
            # Convert that row to a numpy array.
            features = df.iloc[0].values
            return features
            
        except Exception as e:
            print(f"openSMILE processing error: {e}")
            return None

    def predict_deception(self, audio_path):
        """
        Predicts deception probability based on extracted audio features.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            float: Deception probability (0 to 1).
        """
        self.eval()
        features = self.extract_features(audio_path)
        
        if features is None or np.isnan(features).any():
            return 0.5 # Neutral fallback
            
        try:
            # For demonstration, normally the model weights would be trained.
            # We pass the features through the untrained initialized weights to get a random result,
            # simulating the API of a real trained model.
            mfcc_tensor = torch.from_numpy(features).float().unsqueeze(0) # Shape: (1, 88)
            with torch.no_grad():
                score = self.forward(mfcc_tensor)
                prob = torch.sigmoid(score).item()
            return prob
            
        except RuntimeError as e:
            print(f"Torch model error: {e}. Check if feature size {len(features)} matches expected {self.input_size}.")
            return 0.5
