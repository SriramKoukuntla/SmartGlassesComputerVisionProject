import torch
import cv2
import matplotlib.pyplot as plt



MidasModels = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]
MIDAS_MODEL_NAME = MidasModels[2]

def load_MiDaS_model():
    print(f"Loading MiDaS model...")
    midas = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_NAME)
    print("MiDaS model loaded successfully!")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("MiDaS model moved to GPU")
    else:
        device = torch.device("cpu")
        print("MiDaS model moved to CPU")
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    print("Loading MiDaS preprocessing pipeline...")
    if  MIDAS_MODEL_NAME == "DPT_Large" or MIDAS_MODEL_NAME == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    print("MiDaS preprocessing pipeline loaded successfully!")

    return midas, transform, device


midas, transform, midas_device = load_MiDaS_model()

img = cv2.imread("image.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(midas_device)

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

plt.imshow(output)

plt.show()  # Added to actually display the plot
