from paddleocr import PaddleOCR
import numpy as np
import time


print(f"Initializing PaddleOCR model ")
ocr = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True)
print("PaddleOCR model initialized successfully!")

result = ocr.predict(input="image.png")

formatted = []

for res in result:
    texts = res.get("rec_texts", [])
    scores = res.get("rec_scores", [])
    polys = res.get("rec_polys", [])

    for text, score, poly in zip(texts, scores, polys):
        poly = np.array(poly).astype(int).tolist()  # ensure clean [[x,y],...]
        formatted.append([text, float(score), poly])

print(formatted)


# for i in range(6):
#     # Run OCR inference on a sample image 
#     start_time = time.time()
#     result = ocr.predict(input="image.png")
#     end_time = time.time()
#     latency = end_time - start_time
#     print(f"OCR prediction latency: {latency:.4f} seconds ({latency * 1000:.2f} ms)")


# # Visualize the results and save the JSON results
# i = 0;
# for res in result:
#     print(f"Result {i}:")
#     res.print()
#     res.save_to_img("output")
#     res.save_to_json("output")
#     i += 1