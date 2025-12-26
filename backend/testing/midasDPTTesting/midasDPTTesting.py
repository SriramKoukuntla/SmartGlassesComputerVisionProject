import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

def load_image_from_url(url):
    """
    Load an image from a URL.
    
    Args:
        url: URL of the image to load
    
    Returns:
        RGB image as numpy array
    """
    print(f"Downloading image from {url}...")
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image from URL: {url}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image loaded successfully. Shape: {img.shape}")
    return img


def test_model_latency(model_name, img, num_iterations=5):
    """
    Test latency for a given MiDaS model.
    
    Args:
        model_name: Name of the model ("DPT_Hybrid", "DPT_Large", or "MiDaS_small")
        img: RGB image as numpy array
        num_iterations: Number of iterations to run for latency testing
    
    Returns:
        Dictionary with timing statistics
    """
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading {model_name} model...")
    load_start = time.time()
    midas = torch.hub.load("intel-isl/MiDaS", model_name)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Move model to GPU or CPU
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"{model_name} model moved to GPU")
    else:
        device = torch.device("cpu")
        print(f"{model_name} model moved to CPU")
    midas.to(device)
    
    # Set model to evaluation mode
    midas.eval()
    
    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_name == "DPT_Large" or model_name == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    
    # Warm-up run (not counted in latency)
    print("Running warm-up inference...")
    input_batch = transform(img).to(device)
    with torch.no_grad():
        _ = midas(input_batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Latency testing
    print(f"Running {num_iterations} iterations for latency testing...")
    inference_times = []
    
    for i in range(num_iterations):
        # Prepare input
        input_batch = transform(img).to(device)
        
        # Measure inference time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            prediction = midas(input_batch)
            
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        print(f"  Iteration {i+1}/{num_iterations}: {inference_time*1000:.2f} ms")
    
    # Calculate statistics
    inference_times_ms = [t * 1000 for t in inference_times]
    stats = {
        'model_name': model_name,
        'load_time': load_time,
        'inference_times': inference_times,
        'inference_times_ms': inference_times_ms,
        'mean': np.mean(inference_times_ms),
        'std': np.std(inference_times_ms),
        'min': np.min(inference_times_ms),
        'max': np.max(inference_times_ms),
        'median': np.median(inference_times_ms)
    }
    
    print(f"\n{model_name} Statistics:")
    print(f"  Mean latency: {stats['mean']:.2f} ms")
    print(f"  Std deviation: {stats['std']:.2f} ms")
    print(f"  Min latency: {stats['min']:.2f} ms")
    print(f"  Max latency: {stats['max']:.2f} ms")
    print(f"  Median latency: {stats['median']:.2f} ms")
    
    return stats, prediction.cpu().numpy()


def main():
    """Main function to test all models."""
    print("MiDaS Model Latency Testing")
    print("="*60)
    
    # Check device
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_info}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load test image from URL
    image_url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    try:
        img = load_image_from_url(image_url)
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return
    
    # Test all models
    results = {}
    outputs = {}
    
    # Test DPT_Hybrid
    try:
        stats_hybrid, output_hybrid = test_model_latency("DPT_Hybrid", img, num_iterations=5)
        results['DPT_Hybrid'] = stats_hybrid
        outputs['DPT_Hybrid'] = output_hybrid
    except Exception as e:
        print(f"Error testing DPT_Hybrid: {e}")
        results['DPT_Hybrid'] = None
        outputs['DPT_Hybrid'] = None
    
    # Test DPT_Large
    try:
        stats_large, output_large = test_model_latency("DPT_Large", img, num_iterations=5)
        results['DPT_Large'] = stats_large
        outputs['DPT_Large'] = output_large
    except Exception as e:
        print(f"Error testing DPT_Large: {e}")
        results['DPT_Large'] = None
        outputs['DPT_Large'] = None
    
    # Test MiDaS_small
    try:
        stats_small, output_small = test_model_latency("MiDaS_small", img, num_iterations=5)
        results['MiDaS_small'] = stats_small
        outputs['MiDaS_small'] = output_small
    except Exception as e:
        print(f"Error testing MiDaS_small: {e}")
        results['MiDaS_small'] = None
        outputs['MiDaS_small'] = None
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    # Collect valid results
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) > 0:
        print("\nMean Latency Results:")
        for model_name, stats in valid_results.items():
            print(f"  {model_name}: {stats['mean']:.2f} ms")
        
        # Find fastest model
        if len(valid_results) > 1:
            sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['mean'])
            fastest_name, fastest_stats = sorted_models[0]
            slowest_name, slowest_stats = sorted_models[-1]
            
            speedup = (slowest_stats['mean'] / fastest_stats['mean'] - 1) * 100
            print(f"\n{fastest_name} is fastest at {fastest_stats['mean']:.2f} ms")
            print(f"{slowest_name} is slowest at {slowest_stats['mean']:.2f} ms")
            print(f"{fastest_name} is {speedup:.1f}% faster than {slowest_name}")
    
    print(f"\n{'='*60}")
    
    # Optionally show output visualization (uncomment if needed)
    valid_outputs = {k: v for k, v in outputs.items() if v is not None}
    if len(valid_outputs) > 0:
        num_models = len(valid_outputs)
        plt.figure(figsize=(6 * num_models, 5))
        for idx, (model_name, output) in enumerate(valid_outputs.items(), 1):
            plt.subplot(1, num_models, idx)
            plt.imshow(output)
            plt.title(f"{model_name} Output")
            plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
