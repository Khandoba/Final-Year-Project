import torch
import torch.nn as nn

class FusionModel(nn.Module):
    """
    A small learned fusion network that combines the probability scores
    from the Vision, Audio, and Text models into a final deception probability.
    
    Instead of a fixed weighted average, this learns the optimal combination
    from the training data.
    
    Input:  [vision_prob, audio_prob, text_prob]  (3 scalars)
    Output: 1 deception probability
    """
    def __init__(self):
        super(FusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        """x: tensor of shape (batch, 3)"""
        return self.net(x)

    def predict(self, vision_prob, audio_prob, text_prob):
        """Returns a fused probability (float 0-1) given three modality scores."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor([[vision_prob, audio_prob, text_prob]], dtype=torch.float32)
            score = self.forward(x)
            return torch.sigmoid(score).item()

    def load_weights(self, path):
        import os
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            self.eval()
            print(f"FusionModel loaded weights from {path}")
        else:
            print(f"FusionModel weights not found at {path}, using default weighted average.")
