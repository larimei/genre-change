import torch
from audiocraft.models import MusicGen

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        if self.model is None:
            print("Loading model", self.model_path)
            torch.cuda.empty_cache()
            self.model = MusicGen.get_pretrained(self.model_path)
        return self.model
