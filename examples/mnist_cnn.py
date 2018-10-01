#!/usr/bin/env python3

from math import ceil
import time

import dynet as dy

from dynn.layers.dense_layers import DenseLayer
from dynn.layers.convolution_layers import Conv2DLayer
from dynn.layers.pooling_layers import MaxPooling2DLayer
from dynn.layers.flow_layers import FlattenLayer
from dynn.layers.combination_layers import StackedLayers, ConcatenatedLayers
from dynn import activations

from dynn.data import mnist
from dynn.data.batching import NumpyBatchIterator

# For reproducibility
dy.reset_random_seed(31415)

# Model
# =====

# Master parameter collection
pc = dy.ParameterCollection()

# Network
network = StackedLayers(
    # First conv + maxpool layer
    ConcatenatedLayers(
        Conv2DLayer(pc, 1, 16, [5, 5], activation=activations.relu),
        Conv2DLayer(pc, 1, 16, [3, 3], activation=activations.relu),
        dim=-1,
    ),
    MaxPooling2DLayer(default_kernel_size=[2, 2], default_strides=[2, 2]),
    # Second conv + maxpool layer
    Conv2DLayer(pc, 32, 64, [5, 5], activation=activations.relu),
    MaxPooling2DLayer(default_kernel_size=[2, 2], default_strides=[2, 2]),
    # Flatten the resulting 3d tensor
    FlattenLayer(),
    # Final Multilayer perceptron
    DenseLayer(pc, 64*(28//4)**2, 128, activation=activations.relu),
    DenseLayer(pc, 128, 10, activation=activations.identity, dropout=0.1)
)

# Optimizer
trainer = dy.MomentumSGDTrainer(pc, learning_rate=0.01, mom=0.9)

# Data
# ====

# Download MNIST
mnist.download_mnist(".")

# Load the data
(train_x, train_y), (test_x, test_y) = mnist.load_mnist(".")

# Create the batch iterators
train_batches = NumpyBatchIterator(train_x, train_y, batch_size=64)
test_batches = NumpyBatchIterator(test_x, test_y, batch_size=64, shuffle=False)

# Training
# ========

# Start training
for epoch in range(5):
    # Time the epoch
    start_time = time.time()
    for x, y in train_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=False, update=True)
        # Inputs
        x = dy.inputTensor(x, batched=True)
        # Run the cnn
        logits = network(x)
        # Loss function
        nll = dy.mean_batches(dy.pickneglogsoftmax_batch(logits, y))
        # Backward pass
        nll.backward()
        # Update the parameters
        trainer.update()
        # Print the current loss from time to time
        if train_batches.just_passed_multiple(ceil(len(train_batches)/10)):
            print(
                f"Epoch {epoch+1}@{train_batches.percentage_done():.0f}%: "
                f"NLL={nll.value():.3f}"
            )
    # End of epoch logging
    print(f"Epoch {epoch+1}@100%: NLL={nll.value():.3f}")
    print(f"Took {time.time()-start_time:.1f}s")
    print("=" * 20)

# Testing
# =======

# Test
accuracy = 0
for x, y in test_batches:
    # Renew the computation graph
    dy.renew_cg()
    # Initialize layers
    network.init(test=True, update=False)
    # Inputs
    x = dy.inputTensor(x, batched=True)
    # Run the cnn
    logits = network(x)
    # Get prediction
    predicted = logits.npvalue().argmax(axis=0)
    # Accuracy
    accuracy += (predicted == y).sum()
# Average accuracy
accuracy /= test_batches.num_samples
# Print final result
print(f"Test accuracy: {accuracy*100:.2f}%")
