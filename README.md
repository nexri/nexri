# Nexri

Nexri provides specialized neural network layers for TensorFlow that extend standard Keras layers with advanced features.

## Installation

```bash
pip install nexri
```

## Features

### QPDense Layer

A Quadratic Penalty Dense layer with integrated batch normalization that offers:

- Quadratic penalty terms that modify the optimization landscape
- Integrated batch normalization for faster convergence
- Built on top of Keras Dense layer for maximum compatibility

The layer computes:

```
output = BatchNorm(2 * (inputs · kernel) - α * (sum(inputs²)) - α * (sum(kernel²)) + bias)
```

Where:
- α (alpha) is a trainable or fixed parameter that controls the strength of the quadratic penalty terms
- BatchNorm applies normalization to stabilize and accelerate training (optional)

## Usage

```python
from nexri import QPDense
import tensorflow as tf

# Create a simple model with QPDense layers
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    QPDense(128, activation='relu'),   # Using the class directly
    QPDense(10, activation='softmax', use_batch_norm=False)
])

# Compile and train as usual
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Use custom alpha values
custom_layer = QPDense(
    units=32,
    alfa_initial=0.01,
    alfa_trainable=True,
    activation='relu'
)
```

## Advanced Configuration

The QPDense layer accepts all parameters that a regular Dense layer accepts, plus these additional options:

- `alfa_initial`: Initial value for the alpha parameter (default: 0.0)
- `alfa_trainable`: Whether alpha should be trainable (default: False)
- `weight_mean`: Mean value for the kernel initializer (default: 0.5)
- `use_batch_norm`: Whether to apply batch normalization (default: True)
- `batch_norm_momentum`: Momentum for the batch normalization layer (default: 0.99)
- `batch_norm_epsilon`: Small float added to variance to avoid division by zero (default: 1e-3)

## License

MIT
