# Import nexRI advanced dense layer
from nexri import QPDense

# Import regular Tensorflow and Keras components
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Activation
from tensorflow.keras.utils import to_categorical

def main():
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize and reshape
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    # Prepare outputs
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    print("Building QPDense model...")
    # Create a simple model with QPDense layers
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Flatten(),
        QPDense(units=128, weight_mean=0.5),
        Activation('relu'),
        QPDense(units=10),
        Activation('softmax'),
    ])

    # Print model summary
    model.summary()

    # Compile and train as usual
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training model with QPDense layers...")
    # Train the model
    history = model.fit(train_images,
                      train_labels,
                      epochs=5,
                      batch_size=64,
                      validation_data=(test_images, test_labels))
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
