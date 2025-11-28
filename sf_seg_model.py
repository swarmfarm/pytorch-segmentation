import numpy as np
from typing import Tuple

import torch

from swarmfarm_computer_vision.mlops.inference.segmentation.model.seg_model import SegModel, create_confidence

from models import segmentation


"""
Wraps a pytorch-segmentation model in an AP SegModel, for use in AP inference functions.
"""

class SwarmfarmSegModel(SegModel):

    def __init__(
            self, 
            arch: str,
            num_classes,
            checkpoint_file,
            mean,
            std,
            device: str = "cuda",
            ):
        model = segmentation.__dict__[arch](num_classes=num_classes, aux_loss=False, pretrained=False)
        model.to(device)
        model.eval()

        if checkpoint_file:
            checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'])

        self.model = model
        self.mean = mean
        self.std = std
        self.device = device
    
    def inference_image(self, img: np.array) -> Tuple[np.array, np.array, np.array]:

        # Normalise image for input to the model.
        img = (img.astype(float) / 255 - self.mean) / self.std

        with torch.no_grad():
            # Convert to tensor of shape (1, 3, height, width).
            img = torch.Tensor(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

            output = self.model(img)
            logits = output['out'][0]
            logits = logits.cpu().numpy()

            # Get class predictions.
            pred = logits.argmax(axis=0)
            
            # TODO: Compute softmax to generate confidence.
            confidence = None

            return pred, confidence, logits

    def get_class_names(self):
        # TODO: Add this when needed for Detectron2 vis.
        return []

    def get_class_colours(self):
        # TODO: Add this when needed for Detectron2 vis.
        return []
    