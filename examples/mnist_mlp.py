#!/usr/bin/env python3

from math import ceil
import time

import dynet as dy

import dynn
from dynn.layers.dense_layers import Affine
from dynn.layers.combination_layers import Sequential
from dynn.layers.flow_layers import FlattenLayer
from dynn.activations import relu

from dynn.data import mnist
from dynn.data import preprocess
from dynn.data.batching import NumpyBatchIterator

# For reproducibility
dynn.set_random_seed(31415)

# Model
# =====

# Hyper parameters
HIDDEN_DIM = 100

# Master parameter collection
pc = dy.ParameterCollection()

# Network
network = Sequential(
    # Flatten 28 x 28 x 1 image to a vector
    FlattenLayer(),
    # Add all them layers
    Affine(pc, 28**2, HIDDEN_DIM, activation=relu),
    Affine(pc, HIDDEN_DIM, HIDDEN_DIM, activation=relu),
    Affine(pc, HIDDEN_DIM, 10, dropout=0.1)
)

# Optimizer
trainer = dy.MomentumSGDTrainer(pc, learning_rate=0.01, mom=0.9)

# Data
# ====

# Download MNIST
mnist.download_mnist(".")

# Load the data
print("Loading MNIST data")
(train_x, train_y), (test_x, test_y) = mnist.load_mnist(".")

print("Normalizing pixel values")
train_x, test_x = preprocess.normalize([train_x, test_x])

# Create the batch iterators
print("Creating batches")
train_batches = NumpyBatchIterator(train_x, train_y, batch_size=64)
test_batches = NumpyBatchIterator(test_x, test_y, batch_size=64, shuffle=False)

# Training
# ========

print("Starting training")
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

# Save model
# ==========

pc.save("mnist_mlp.model")
