import os
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vision_model import VisionModel
from models.audio_model import AudioModel

def evaluate_models():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

    vision_path = os.path.join(data_dir, 'vision_seqs.npy')
    audio_path = os.path.join(data_dir, 'audio_seqs.npy')
    labels_path = os.path.join(data_dir, 'labels.npy')

    if not os.path.exists(vision_path) or not os.path.exists(audio_path):
        print("Dataset not found. Please run data preparation scripts first.")
        return

    print("Loading sequence data...")
    vision_seqs = np.load(vision_path)
    audio_seqs = np.load(audio_path)
    labels = np.load(labels_path)

    # We only care about evaluating on the Test set (the 25% unseen data)
    indices = np.arange(len(labels))
    _, test_idx = train_test_split(indices, test_size=0.25, random_state=42)
    
    test_v = torch.tensor(vision_seqs[test_idx], dtype=torch.float32)
    test_a = torch.tensor(audio_seqs[test_idx], dtype=torch.float32)
    test_l = labels[test_idx]

    print("Loading trained models...")
    vision_model = VisionModel()
    try:
        vision_model.load_state_dict(torch.load(os.path.join(models_dir, 'vision_model.pth'), weights_only=True))
        vision_model.eval()
    except Exception as e:
        print(f"Could not load vision model: {e}")
        return

    audio_model = AudioModel()
    try:
        audio_model.load_state_dict(torch.load(os.path.join(models_dir, 'audio_model.pth'), weights_only=True))
        audio_model.eval()
    except Exception as e:
        print(f"Could not load audio model: {e}")
        return

    print(f"\nEvaluating independently on Test Set ({len(test_idx)} unseen samples)...\n")

    with torch.no_grad():
        v_probs = torch.sigmoid(vision_model(test_v)).squeeze(1).numpy()
        a_probs = torch.sigmoid(audio_model(test_a)).squeeze(1).numpy()
        
        v_preds = (v_probs >= 0.5).astype(int)
        a_preds = (a_probs >= 0.5).astype(int)
        
        # Simulate our Rule-Based Agent Fusion logic
        f_preds = []
        for v_val, a_val in zip(v_probs, a_probs):
            v_deceptive = v_val >= 0.5
            a_deceptive = a_val >= 0.5
            
            if v_deceptive and a_deceptive:
                f_preds.append(1) # Deceptive consensus
            elif not v_deceptive and not a_deceptive:
                f_preds.append(0) # Truthful consensus
            else:
                # Conflicting. Check Aggressive Deception rule (>0.75)
                if v_val > 0.75 or a_val > 0.75:
                    f_preds.append(1) # Override to deceptive
                else:
                    # Pure conflict, defaults to average threshold
                    f_preds.append(int((v_val + a_val) / 2 >= 0.5))

    def print_metrics(name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix: [[True Neg, False Pos], [False Neg, True Pos]]
        cm = confusion_matrix(y_true, y_pred)
        tn, fp = cm[0] if len(cm) > 1 else (cm[0][0], 0)
        fn, tp = cm[1] if len(cm) > 1 else (0, 0)
        
        print(f"=============================")
        print(f"      {name} Model ")
        print(f"=============================")
        print(f"Accuracy:  {acc*100:5.1f}%")
        print(f"Precision: {prec*100:5.1f}% (When it says 'Lie', how often is it right?)")
        print(f"Recall:    {rec*100:5.1f}% (Out of all real lies, how many did it catch?)")
        print(f"F1-Score:  {f1*100:5.1f}%")
        print("\nConfusion Matrix:")
        print(f"  Truthful correctly identified (TN): {tn}")
        print(f"  Truthful wrongly flagged (FP):      {fp}  <-- False Alarm")
        print(f"  Lies missed by model (FN):          {fn}  <-- Got fooled")
        print(f"  Lies correctly identified (TP):     {tp}\n")

    print_metrics("VISION", test_l, v_preds)
    print_metrics("AUDIO", test_l, a_preds)
    print_metrics("AGENT FUSION", test_l, f_preds)

if __name__ == "__main__":
    evaluate_models()
