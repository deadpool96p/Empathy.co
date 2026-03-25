# backend/test_gpu.py
import tensorflow as tf
import torch

def test_devices():
    print("=" * 60)
    print("DEVICE DIAGNOSTICS")
    print("=" * 60)
    
    # TensorFlow
    print(f"TensorFlow version: {tf.__version__}")
    tf_gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow GPUs found: {len(tf_gpus)}")
    for gpu in tf_gpus:
        print(f" - {gpu}")
        
    # PyTorch
    print("-" * 30)
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" - Device name: {torch.cuda.get_device_name(0)}")
        print(f" - Device count: {torch.cuda.device_count()}")
    
    print("=" * 60)

if __name__ == "__main__":
    test_devices()