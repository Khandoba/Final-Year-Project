import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.fusion_model import FusionModel

class MU3DDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        # Assuming Text is dropped for now, we pad it with 0.5 (neutral)
        self.features = df[['VisionProb', 'AudioProb']].values
        # Pad with neutral text probability to match Linear(3,1)
        self.features = torch.tensor(self.features, dtype=torch.float32)
        neutral_text = torch.full((self.features.shape[0], 1), 0.5)
        self.features = torch.cat((self.features, neutral_text), dim=1)
        
        self.labels = torch.tensor(df['Veracity'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def main():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mu3d_extracted_features.csv')
    
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Please run extract_mu3d.py first.")
        return

    print("Loading dataset...")
    full_dataset = MU3DDataset(csv_path)
    
    # Split
    if len(full_dataset) < 2:
        print("Not enough data to split.")
        return
        
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    model = FusionModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 20
    print(f"Beginning training for {epochs} epochs on {len(train_dataset)} samples...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_features.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                test_loss += loss.item() * batch_features.size(0)
                preds = torch.sigmoid(outputs).round()
                all_preds.extend(preds.numpy())
                all_labels.extend(batch_labels.numpy())
                
        test_loss /= len(test_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {acc:.4f}")

    # Save model
    save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_fusion_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Training complete! Model saved to {save_path}")

if __name__ == "__main__":
    main()
