import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import transformers

# Suppress expected load warnings
transformers.logging.set_verbosity_error()

class TextModel:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        Initializes the Natural Language Processing component for analyzing spoken deception.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        except Exception as e:
            print(f"Failed to load HuggingFace model '{model_name}': {e}. Ensure transformers and internet connection are available.")
            self.tokenizer = None
            self.model = None

    def predict_deception(self, text):
        """
        Predicts deception probability based on text content.
        
        Args:
            text (str): Extracted transcription from speech.
            
        Returns:
            float: Deception probability (0 to 1).
        """
        if not self.model or not self.tokenizer:
            return 0.5
            
        if not text or len(text.strip()) == 0:
            return 0.5 # No text to analyze

        self.model.eval()
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Softmax to get probabilities for class 0 (Truth) and class 1 (Deception)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            deception_prob = float(probs[1])
            
        return deception_prob
