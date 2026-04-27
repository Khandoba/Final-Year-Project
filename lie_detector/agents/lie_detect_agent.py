from models.vision_model import VisionModel
from models.audio_model import AudioModel
from models.text_model import TextModel
from models.fusion_model import FusionModel
import torch
import os

class LieDetectAgent:
    def __init__(self, openface_path=None):
        """
        Initializes the Agent responsible for analyzing modalities and performing ReAct reasoning.
        """
        self.vision_model = VisionModel(openface_path=openface_path)
        self.audio_model = AudioModel()
        self.text_model = TextModel()
        
        self.fusion_model = FusionModel()
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_fusion_model.pth')
        self.fusion_model.load_weights(model_path)
        
        self.thoughts = []

    def analyze(self, image_or_video=None, audio_file=None, text=None):
        """
        Performs analysis on the provided inputs and reasons about deception probability.
        
        Args:
            image_or_video (str): Path to image or video chunk.
            audio_file (str): Path to audio chunk.
            text (str): Spoken transcription.
            
        Returns:
            dict: Decision out, confidence, explanation, and individual scores.
        """
        self.thoughts = []
        scores = {}

        # 1. Vision Analysis
        if image_or_video is not None:
            vision_prob = self.vision_model.predict_deception(image_or_video)
            scores['vision'] = vision_prob
            self.thoughts.append(f"Vision analysis (OpenFace Action Units): model returned probability {vision_prob:.2f} for deception.")
            
            if vision_prob > 0.7:
                self.thoughts.append("Thought: Visual cues suggest stress, discomfort, or deceit.")
            elif vision_prob < 0.3:
                self.thoughts.append("Thought: Facial expression appears normal/relaxed.")

        # 2. Audio Analysis
        if audio_file is not None:
            audio_prob = self.audio_model.predict_deception(audio_file)
            scores['audio'] = audio_prob
            self.thoughts.append(f"Audio analysis (openSMILE eGeMAPS): model returned probability {audio_prob:.2f} for deception.")
            
            if audio_prob > 0.7:
                self.thoughts.append("Thought: Acoustic features (pitch, voice quality) indicate high stress.")
            elif audio_prob < 0.3:
                self.thoughts.append("Thought: Voice features do not show significant stress indicators.")

        # 3. Text Analysis
        if text is not None and len(text.strip()) > 0:
            text_prob = self.text_model.predict_deception(text)
            scores['text'] = text_prob
            self.thoughts.append(f"Text analysis (NLP): model returned probability {text_prob:.2f} for deception.")
            
            if text_prob > 0.7:
                self.thoughts.append("Thought: Linguistic analysis finds cues of deception or inconsistency in wording.")
            elif text_prob < 0.3:
                self.thoughts.append("Thought: Linguistic content appears consistent.")

        if not scores:
            return {"decision": "No data", "confidence": 0.0, "explanation": "No valid modalities could be analyzed."}

        # 4. Neural Fusion Reasoning
        vision_val = scores.get('vision', 0.5)
        audio_val = scores.get('audio', 0.5)
        text_val = scores.get('text', 0.5)
        
        feats = torch.tensor([[vision_val, audio_val, text_val]], dtype=torch.float32)
        with torch.no_grad():
            output = self.fusion_model(feats)
            avg_score = torch.sigmoid(output).item()
            
        self.thoughts.append(f"Trained Fusion probability = {avg_score:.2f}.")

        if avg_score >= 0.5:
            decision = "Deceptive"
            conf = avg_score
        else:
            decision = "Truthful"
            conf = 1 - avg_score

        self.thoughts.append(f"Action: Based on combined analysis, decision = {decision}.")

        # Handling conflicting signals
        if 0.4 < avg_score < 0.6 and len(scores) > 1:
            spread = max(scores.values()) - min(scores.values())
            if spread > 0.5:
                self.thoughts.append("Thought: Modalities disagree significantly (e.g., relaxed face but stressed voice). Flagging for human review.")
                decision += " (Ambiguous/Conflicting)"

        explanation = "\n".join(self.thoughts)
        
        return {
            "decision": decision, 
            "confidence": float(conf), 
            "explanation": explanation, 
            "scores": scores,
            "fused_probability": avg_score # The raw 0-1 probability
        }
