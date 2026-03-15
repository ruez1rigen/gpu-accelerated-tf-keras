# gpu-accelerated-tf-keras 🏎️⚡

This repository explores performance optimization for deep learning models, focusing on maximizing GPU utilization and minimizing training time.

## 📊 Focus Areas
- **Custom Training Loops**: Bypassing high-level API overhead for granular performance control.
- **Mixed Precision Training**: Utilizing FP16 for faster computation on modern NVIDIA GPUs.
- **Distributed Strategies**: Implementing `MirroredStrategy` and `MultiWorkerMirroredStrategy`.

## 💻 Sample: Optimized Setup
```python
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision

# Enable Mixed Precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Multi-GPU Strategy
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.Sequential([
        layers.Dense(1024, activation='relu'),
        layers.Dense(10, activation='softmax', dtype='float32') # Ensure output is FP32
    ])
```

---
*Based on my work in GPU-accelerated AI research.*
