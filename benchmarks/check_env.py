import tensorflow as tf
import os

def check_gpu_status():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s).")
        for i, gpu in enumerate(gpus):
            print(f"   [{i}] {gpu}")
    else:
        print("❌ No GPU found. Falling back to CPU.")

if __name__ == "__main__":
    check_gpu_status()
