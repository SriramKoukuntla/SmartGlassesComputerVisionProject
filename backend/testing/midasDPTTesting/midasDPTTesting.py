import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

def test_model_latency(model_name, num_iterations=5):
    """
    Test latency for a given MiDaS model.
    
    Args:
        model_name: Name of the model ("DPT_Hybrid" or "DPT_Large")
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
    
    # Load image and convert to RGB
    img = cv2.imread("image.png")
    if img is None:
        raise FileNotFoundError("image.png not found in current directory")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
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
    """Main function to test both models."""
    print("MiDaS Model Latency Testing")
    print("="*60)
    
    # Check device
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_info}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Test both models
    results = {}
    
    # Test DPT_Hybrid
    try:
        stats_hybrid, output_hybrid = test_model_latency("DPT_Hybrid", num_iterations=5)
        results['DPT_Hybrid'] = stats_hybrid
    except Exception as e:
        print(f"Error testing DPT_Hybrid: {e}")
        results['DPT_Hybrid'] = None
    
    # Test DPT_Large
    try:
        stats_large, output_large = test_model_latency("DPT_Large", num_iterations=5)
        results['DPT_Large'] = stats_large
    except Exception as e:
        print(f"Error testing DPT_Large: {e}")
        results['DPT_Large'] = None
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    if results['DPT_Hybrid'] and results['DPT_Large']:
        hybrid_mean = results['DPT_Hybrid']['mean']
        large_mean = results['DPT_Large']['mean']
        
        print(f"\nDPT_Hybrid mean latency: {hybrid_mean:.2f} ms")
        print(f"DPT_Large mean latency: {large_mean:.2f} ms")
        
        if hybrid_mean < large_mean:
            faster = "DPT_Hybrid"
            speedup = (large_mean / hybrid_mean - 1) * 100
        else:
            faster = "DPT_Large"
            speedup = (hybrid_mean / large_mean - 1) * 100
        
        print(f"\n{faster} is {speedup:.1f}% faster")
        print(f"Speed difference: {abs(hybrid_mean - large_mean):.2f} ms")
    
    print(f"\n{'='*60}")
    
    # Optionally show output visualization (uncomment if needed)
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(output_hybrid)
    # plt.title("DPT_Hybrid Output")
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(output_large)
    # plt.title("DPT_Large Output")
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
