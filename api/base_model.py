from PIL import Image
import numpy as np
import torch
from model import Net
from process.transformation import get_transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



WEIGHTS = {
    'HIPER' : 'artifacts/hiper_others_2023-05-12-18-06-48/4_fold_max_acc_checkpoint.pth.tar',
    'MEMBR' : 'artifacts/membran_others_2023-05-12-20-31-18/0_fold_max_acc_checkpoint.pth.tar',
    'NORM' : 'artifacts/normal_others_2023-05-12-15-45-47/4_fold_max_acc_checkpoint.pth.tar',
    'SCLER' : 'artifacts/sclero_others_2023-05-12-13-27-54/4_fold_max_acc_checkpoint.pth.tar',
    'PODOC' : 'artifacts/podoc_others_2023-05-17-08-46-23/2_fold_max_acc_checkpoint.pth.tar',
    'CRESC' : 'artifacts/cresc_others_2023-05-17-13-22-44/2_fold_max_acc_checkpoint.pth.tar',
}

class Model:
    def __init__(self, model_type) -> None:
        self._model = None
        self._weights_path = WEIGHTS[model_type]
        self._transform = get_transform()
        self._load_model()
        self._load_model_weights()

    def _load_model(self) -> None:
        self._model = Net(net_version="b0", num_classes=2).to(DEVICE)
        print("model loaded")


    def process(self, image: Image.Image)->np.ndarray:
        image_processed = self._transform(image)
        image_processed = image_processed.unsqueeze(0)
        return image_processed

    def predict(self, image: np.ndarray):
        self._model.eval()

        with torch.no_grad():
            
            output = self._model(image.to(DEVICE))
            scores = torch.sigmoid(output)
            predictions = (scores>0.5).float()
            _, pred = torch.min(predictions, 1)

        return pred.item()
    
    def _load_model_weights(self):
        checkpoint = torch.load(self._weights_path, map_location=torch.device(DEVICE))
        self._model.load_state_dict(checkpoint['state_dict'])
