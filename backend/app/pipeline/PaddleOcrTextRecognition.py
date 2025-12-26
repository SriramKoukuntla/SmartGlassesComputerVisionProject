"""
OCR pipeline for extracting text from images using PaddleOCR.
"""
from typing import List, Dict, Any
import cv2
import numpy as np
from paddleocr import PaddleOCR


def runPaddleOcr(img, ocr_model: PaddleOCR) -> List[Dict]:
    result = ocr_model.predict(input=img)
    return result

def parsePaddleOcrResult(paddleocr_result: List[Dict]):
    formatted = []

    for res in paddleocr_result:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        polys = res.get("rec_polys", [])

        for text, score, poly in zip(texts, scores, polys):
            poly = np.array(poly).astype(int).tolist()  # ensure clean [[x,y],...]
            formatted.append([text, float(score), poly])

    return formatted

def PaddleOcrResultToJSON(paddleocr_result):
    return {
        "text_detections": paddleocr_result,
        "count": len(paddleocr_result),
        "full_text": " ".join([det[0] for det in paddleocr_result])
    }

def executePaddleOcr(img, ocr_model: PaddleOCR):
    result = runPaddleOcr(img, ocr_model)
    detections = parsePaddleOcrResult(result)
    JSON = PaddleOcrResultToJSON(detections)
    return JSON