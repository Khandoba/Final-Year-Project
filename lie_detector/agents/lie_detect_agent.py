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
        v_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vision_model.pth')
        if os.path.exists(v_path):
            self.vision_model.load_state_dict(torch.load(v_path, weights_only=True))
            self.vision_model.eval()

        self.audio_model = AudioModel()
        a_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'audio_model.pth')
        if os.path.exists(a_path):
            self.audio_model.load_state_dict(torch.load(a_path, weights_only=True))
            self.audio_model.eval()
            
        self.text_model = TextModel()
        
        self.fusion_model = FusionModel()
        f_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'fusion_model.pth')
        self.fusion_model.load_weights(f_path)
        
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
            self.thoughts.append(f"Audio analysis (librosa MFCCs): model returned probability {audio_prob:.2f} for deception.")
            
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

        # 4. Rule-Based Fusion Reasoning
        vision_val = scores.get('vision', 0.5)
        audio_val = scores.get('audio', 0.5)
        
        # Get individual decisions based on a 0.5 threshold
        vision_is_deceptive = vision_val >= 0.5
        audio_is_deceptive = audio_val >= 0.5

        # Check for agreement or conflict
        if vision_is_deceptive and audio_is_deceptive:
            decision = "Deceptive"
            conf = max(vision_val, audio_val)
            self.thoughts.append("Action: Both modalities agree. Decision = Deceptive.")
        elif not vision_is_deceptive and not audio_is_deceptive:
            decision = "Truthful"
            conf = 1 - min(vision_val, audio_val)
            self.thoughts.append("Action: Both modalities agree. Decision = Truthful.")
        else:
            # They disagree!
            decision = "Conflicting"
            # In a conflict, the confidence is low
            conf = 0.5
            
            # Explain the conflict in the thoughts
            if vision_is_deceptive:
                self.thoughts.append("Thought: Modalities disagree! Vision detects deception, but Audio seems truthful.")
            else:
                self.thoughts.append("Thought: Modalities disagree! Audio detects stress/deception, but Vision seems relaxed.")
                
            # "Aggressive Deception" override rule:
            # If one model is EXTREMELY confident about deception (>0.75), override the conflict
            if vision_val > 0.75 or audio_val > 0.75:
                decision = "Deceptive (High Alert)"
                conf = max(vision_val, audio_val)
                self.thoughts.append("Action: One modality is extremely confident about deception. Overriding conflict to flag as Deceptive.")

        # Determine aggregate deception score for the UI bar
        # Averaging them gives a smooth, realistic bar (e.g., 30% or 40% deception)
        deception_prob = (vision_val + audio_val) / 2.0
        
        # If the Aggressive Override triggered, ensure the bar reflects the high deception
        if decision.startswith("Deceptive") and deception_prob < 0.5:
            deception_prob = max(vision_val, audio_val)

        explanation = "\n".join(self.thoughts)
        
        return {
            "decision": decision, 
            "confidence": float(conf), 
            "deception_prob": float(deception_prob),
            "explanation": explanation, 
            "scores": scores
        }
