import torch
import cv2
import numpy as np
from typing import Tuple, Dict, List

def imagePreProcessMiDaS(img: np.ndarray, transform, device) -> torch.Tensor:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    return input_batch


def runMiDaS(input_batch: torch.Tensor, model, img_shape: Tuple[int, int]) -> torch.Tensor:
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        return prediction

def JSONifyMiDaS(prediction: torch.Tensor) -> dict:
    return {
        "depth_map": prediction.cpu().numpy().tolist()
    }


def executeMiDaS(img: np.ndarray, model, transform, device) -> dict:
    input_batch = imagePreProcessMiDaS(img, transform, device)
    prediction = runMiDaS(input_batch, model, img.shape[:2])
    JSON = JSONifyMiDaS(prediction)
    return JSON