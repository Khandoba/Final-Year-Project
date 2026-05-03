import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vision_model import VisionModel
from models.audio_model import AudioModel
from models.fusion_model import FusionModel

def train_models(epochs=200, lr=0.01):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

    vision_path = os.path.join(data_dir, 'vision_seqs.npy')
    audio_path = os.path.join(data_dir, 'audio_seqs.npy')
    labels_path = os.path.join(data_dir, 'labels.npy')

    if not os.path.exists(vision_path) or not os.path.exists(audio_path):
        print("Sequence data not found. Run prepare_data_seq.py / reextract scripts first.")
        return

    print("Loading sequence data...")
    vision_seqs = np.load(vision_path)
    audio_seqs = np.load(audio_path)
    labels = np.load(labels_path)

    print(f"Total samples: {len(labels)}")
    print(f"Vision shape: {vision_seqs.shape}")
    print(f"Audio shape:  {audio_seqs.shape}")

    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=42)
    print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")

    def make_tensors(idx):
        v = torch.tensor(vision_seqs[idx], dtype=torch.float32)
        a = torch.tensor(audio_seqs[idx], dtype=torch.float32)
        l = torch.tensor(labels[idx], dtype=torch.float32).unsqueeze(1)
        return v, a, l

    train_v, train_a, train_l = make_tensors(train_idx)
    test_v, test_a, test_l = make_tensors(test_idx)

    vision_model = VisionModel()
    audio_model = AudioModel()

    criterion = nn.BCEWithLogitsLoss()
    vision_optimizer = optim.Adam(vision_model.parameters(), lr=lr, weight_decay=1e-4)
    audio_optimizer = optim.Adam(audio_model.parameters(), lr=lr, weight_decay=1e-4)

    vision_scheduler = optim.lr_scheduler.ReduceLROnPlateau(vision_optimizer, mode='min', factor=0.5, patience=15)
    audio_scheduler = optim.lr_scheduler.ReduceLROnPlateau(audio_optimizer, mode='min', factor=0.5, patience=15)

    print(f"\n--- Training Vision & Audio LSTMs for {epochs} epochs ---")

    for epoch in range(epochs):
        # Vision
        vision_model.train()
        vision_optimizer.zero_grad()
        v_noisy = train_v + 0.05 * torch.randn_like(train_v)
        v_out = vision_model(v_noisy)
        v_loss = criterion(v_out, train_l)
        v_loss.backward()
        vision_optimizer.step()
        vision_scheduler.step(v_loss.detach())

        # Audio
        audio_model.train()
        audio_optimizer.zero_grad()
        a_noisy = train_a + 0.05 * torch.randn_like(train_a)
        a_out = audio_model(a_noisy)
        a_loss = criterion(a_out, train_l)
        a_loss.backward()
        audio_optimizer.step()
        audio_scheduler.step(a_loss.detach())

        if (epoch + 1) % 20 == 0:
            v_lr = vision_optimizer.param_groups[0]['lr']
            a_lr = audio_optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] | V Loss: {v_loss.item():.4f} (lr={v_lr:.5f}) | A Loss: {a_loss.item():.4f} (lr={a_lr:.5f})")

    # --- Train Fusion Model ---
    print("\n--- Training Learned Fusion Model ---")
    vision_model.eval()
    audio_model.eval()

    # Generate probability scores for all training samples
    with torch.no_grad():
        v_train_probs = torch.sigmoid(vision_model(train_v)).squeeze(1)
        a_train_probs = torch.sigmoid(audio_model(train_a)).squeeze(1)
        # Text prob placeholder (0.5 = neutral, since we don't have text features in dataset)
        t_train_probs = torch.full((len(train_idx),), 0.5)
        fusion_train_x = torch.stack([v_train_probs, a_train_probs, t_train_probs], dim=1)

        v_test_probs = torch.sigmoid(vision_model(test_v)).squeeze(1)
        a_test_probs = torch.sigmoid(audio_model(test_a)).squeeze(1)
        t_test_probs = torch.full((len(test_idx),), 0.5)
        fusion_test_x = torch.stack([v_test_probs, a_test_probs, t_test_probs], dim=1)

    fusion_model = FusionModel()
    fusion_optimizer = optim.Adam(fusion_model.parameters(), lr=0.005)

    for epoch in range(300):
        fusion_model.train()
        fusion_optimizer.zero_grad()
        f_out = fusion_model(fusion_train_x)
        f_loss = criterion(f_out, train_l)
        f_loss.backward()
        fusion_optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Fusion Epoch [{epoch+1}/300] Loss: {f_loss.item():.4f}")

    # --- Evaluation ---
    print("\n--- Test Set Evaluation ---")
    vision_model.eval()
    audio_model.eval()
    fusion_model.eval()

    with torch.no_grad():
        v_preds = (torch.sigmoid(vision_model(test_v)) >= 0.5).float()
        v_acc = (v_preds == test_l).float().mean().item()

        a_preds = (torch.sigmoid(audio_model(test_a)) >= 0.5).float()
        a_acc = (a_preds == test_l).float().mean().item()

        f_preds = (torch.sigmoid(fusion_model(fusion_test_x)) >= 0.5).float()
        f_acc = (f_preds == test_l).float().mean().item()

    print(f"Vision Model Accuracy:  {v_acc * 100:.2f}%")
    print(f"Audio Model Accuracy:   {a_acc * 100:.2f}%")
    print(f"Fusion Model Accuracy:  {f_acc * 100:.2f}%  (combined)")

    # Save weights
    os.makedirs(models_dir, exist_ok=True)
    torch.save(vision_model.state_dict(), os.path.join(models_dir, 'vision_model.pth'))
    torch.save(audio_model.state_dict(), os.path.join(models_dir, 'audio_model.pth'))
    torch.save(fusion_model.state_dict(), os.path.join(models_dir, 'fusion_model.pth'))

    print("\nWeights saved:")
    print(f"  - {models_dir}/vision_model.pth")
    print(f"  - {models_dir}/audio_model.pth")
    print(f"  - {models_dir}/fusion_model.pth")

if __name__ == "__main__":
    train_models()
