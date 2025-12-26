import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt

# Load model
print("Loading MiDaS model...")
# MIDAS_MODEL_NAME = "DPT_Hybrid"
MIDAS_MODEL_NAME = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_NAME)
print("MiDaS model loaded successfully!")

#Move model to GPU or CPU
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("MiDaS model moved to GPU")
else:
    device = torch.device("cpu")
    print("MiDaS model moved to CPU")
midas.to(device)

#Set model to evaluation mode
midas.eval()


#Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if MIDAS_MODEL_NAME == "DPT_Large" or MIDAS_MODEL_NAME == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

#Load image and convert to RGB
img = cv2.imread("image.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_batch = transform(img).to(device)

#Run inference
with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

#Show output
plt.imshow(output)
plt.show()

