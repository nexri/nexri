# nexri QPDense Example

This repository contains example code for using the `QPDense` layer from the `nexri` library, which provides advanced dense layer implementations for TensorFlow/Keras.

## Overview

The `QPDense` layer extends the traditional dense layer with additional capabilities for model quantization and parameter optimization. This example demonstrates how to use `QPDense` with the MNIST dataset for handwritten digit recognition.

For complete documentation of the `QPDense` layer and all its parameters and specifications, please refer to the official PyPI page:
[https://pypi.org/project/nexri/0.1.0/](https://pypi.org/project/nexri/0.1.0/)

## Requirements

- Python 3.7+
- TensorFlow 2.x
- nexri library

## Installation

1. Install the required packages:

```bash
pip install tensorflow
pip install nexri
```

## Documentation

For detailed information about the `QPDense` layer including:
- All available parameters
- Implementation details
- Performance characteristics
- Advanced usage patterns

Please refer to the official documentation on PyPI:
[https://pypi.org/project/nexri/0.1.0/](https://pypi.org/project/nexri/0.1.0/)

## Usage

Run the example script:

```bash
python qpdense_example.py
```

This will:
1. Load the MNIST dataset
2. Create a simple neural network using `QPDense` layers
3. Train the model for 5 epochs
4. Evaluate the model on test data

## Example Code

The example demonstrates:
- How to import and use the `QPDense` layer
- How to set custom parameters like `weight_mean`
- Integration with standard TensorFlow/Keras components
- Basic model training and evaluation

## Results

The example model should achieve approximately 97-98% accuracy on the MNIST test set after just 5 epochs, demonstrating the effectiveness of the `QPDense` layer.

## Contributing

Feel free to submit issues or pull requests to improve this example.

## License

This example is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
