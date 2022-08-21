from typing import Optional

import torch
from PIL import Image
from torchvision import models, transforms
import numpy as np
import cv2


class FullBodySegmentaion(object):
    def __init__(
        self,
        device: Optional[torch.device] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        output_shape=(192, 256),
    ):
        self.device = device
        self.output_shape = output_shape
        self.img_transforms = transforms.Compose(
            [
                transforms.Resize((520)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        self.coco_classes_list = [
            "__background__",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

    def predict_numpy(self, image):
        image = Image.fromarray(image)
        return self.predict_pil(image=image)

    def predict_path(self, image_path):
        image = Image.open(image_path)
        return self.predict_pil(image=image)

    def predict_pil(self, image):
        image_tensor = self.img_transforms(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        output = self.model(image_tensor)
        output = torch.argmax(output["out"].squeeze(0), dim=0).cpu().numpy()
        output = np.where(output == 15, 255, 0).astype(np.uint8)
        if self.output_shape:
            output = cv2.resize(output, self.output_shape, cv2.INTER_AREA)
        return output
