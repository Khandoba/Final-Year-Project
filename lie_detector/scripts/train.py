import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Add parent directory to path so we can import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vision_model import VisionModel
from models.audio_model import AudioModel

def train_models(epochs=100, lr=0.01):
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'training_data.csv'))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    
    if not os.path.exists(data_path):
        print(f"Training data not found at {data_path}. Run prepare_data.py first.")
        return

    print("Loading training data...")
    df = pd.read_csv(data_path)
    
    if len(df) == 0:
        print("Training dataset is empty.")
        return
        
    print(f"Total samples loaded: {len(df)}")
        
    # Split into Train (75%) and Test (25%)
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
    print(f"Training on {len(train_df)} samples. Testing on {len(test_df)} samples.")
    
    # ------------------------------------------------------------------
    # Data Preparation Helper
    # ------------------------------------------------------------------
    target_aus = ['AU04_c', 'AU15_c', 'AU45_c', 'AU12_c', 'AU20_c']
    audio_cols = [f'audio_{i}' for i in range(88)]
    
    def get_tensors(dataframe):
        labels = torch.tensor(dataframe['Label'].values, dtype=torch.float32).unsqueeze(1)
        vision_feats = torch.tensor(dataframe[target_aus].values, dtype=torch.float32)
        audio_feats = torch.tensor(dataframe[audio_cols].values, dtype=torch.float32)
        return labels, vision_feats, audio_feats

    train_labels, train_vision, train_audio = get_tensors(train_df)
    test_labels, test_vision, test_audio = get_tensors(test_df)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    vision_model = VisionModel()
    audio_model = AudioModel()
    
    criterion = nn.BCEWithLogitsLoss()
    vision_optimizer = optim.Adam(vision_model.parameters(), lr=lr)
    audio_optimizer = optim.Adam(audio_model.parameters(), lr=lr)

    print("\n--- Starting Training ---")
    
    for epoch in range(epochs):
        # Vision Training
        vision_model.train()
        vision_optimizer.zero_grad()
        v_out = vision_model(train_vision)
        v_loss = criterion(v_out, train_labels)
        v_loss.backward()
        vision_optimizer.step()
        
        # Audio Training
        audio_model.train()
        audio_optimizer.zero_grad()
        a_out = audio_model(train_audio)
        a_loss = criterion(a_out, train_labels)
        a_loss.backward()
        audio_optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Vision Loss: {v_loss.item():.4f} | Audio Loss: {a_loss.item():.4f}")

    # ------------------------------------------------------------------
    # Evaluation on Test Set
    # ------------------------------------------------------------------
    print("\n--- Test Set Evaluation ---")
    vision_model.eval()
    audio_model.eval()
    
    with torch.no_grad():
        # Vision
        v_test_out = vision_model(test_vision)
        v_test_probs = torch.sigmoid(v_test_out)
        v_test_preds = (v_test_probs >= 0.5).float()
        v_accuracy = (v_test_preds == test_labels).float().mean().item()
        
        # Audio
        a_test_out = audio_model(test_audio)
        a_test_probs = torch.sigmoid(a_test_out)
        a_test_preds = (a_test_probs >= 0.5).float()
        a_accuracy = (a_test_preds == test_labels).float().mean().item()
        
    print(f"Vision Model Accuracy: {v_accuracy * 100:.2f}%")
    print(f"Audio Model Accuracy: {a_accuracy * 100:.2f}%")

    # ------------------------------------------------------------------
    # Save Weights
    # ------------------------------------------------------------------
    os.makedirs(models_dir, exist_ok=True)
    v_path = os.path.join(models_dir, 'vision_model.pth')
    a_path = os.path.join(models_dir, 'audio_model.pth')
    
    torch.save(vision_model.state_dict(), v_path)
    torch.save(audio_model.state_dict(), a_path)
    
    print("\nTraining complete. Weights saved to:")
    print(f" - {v_path}")
    print(f" - {a_path}")

if __name__ == "__main__":
    train_models()
