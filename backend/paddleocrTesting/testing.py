from paddleocr import PaddleOCR
import time


print(f"Initializing PaddleOCR model ")
ocr = PaddleOCR(
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_textline_orientation=True)
print("PaddleOCR model initialized successfully!")

for i in range(6):
    # Run OCR inference on a sample image 
    start_time = time.time()
    result = ocr.predict(input="image.png")
    end_time = time.time()
    latency = end_time - start_time
    print(f"OCR prediction latency: {latency:.4f} seconds ({latency * 1000:.2f} ms)")

# Visualize the results and save the JSON results
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")