#!/usr/bin/env python3

from math import ceil
import time

import dynet as dy

from dynn.layers.functional_layers import LambdaLayer
from dynn.layers.dense_layers import Affine
from dynn.layers.residual_layers import ResidualLayer
from dynn.layers.convolution_layers import Conv2D
from dynn.layers.flow_layers import FlattenLayer
from dynn.layers.combination_layers import Sequential
from dynn.activations import relu, identity

from dynn.data import cifar10
from dynn.data.batching import NumpyBatchIterator

# For reproducibility
dy.reset_random_seed(31415)

# Model
# =====

# Master parameter collection
pc = dy.ParameterCollection()

# TODO: check pooling/conv dimensions
# Keep track of the number of pixels
n_pixels = 32 * 32
# Initial layer
res_layers = [Conv2D(pc, 3, 16, [3, 3], activation=relu)]
# Stacks of residual blocks
for n_res_block in range(4):
    # Stride
    stride = 1 if n_res_block == 0 else 2
    # Input and output channels (=#of kernels)
    in_channels = 16*(2**max(n_res_block-1, 0))
    num_kernels = 16 * (2 ** n_res_block)
    # Build the residual block
    block = Sequential(
        Conv2D(
            pc,
            in_channels,
            num_kernels,
            [3, 3],
            default_strides=[stride, stride],
            activation=relu,
            nobias=True,
        ),
        Conv2D(
            pc,
            num_kernels,
            num_kernels,
            [3, 3],
            activation=relu,
            nobias=True,
        ),
    )
    # Shortcut layer so that the input is the same shape as the output
    if stride > 1 or in_channels != num_kernels:
        shortcut = Conv2D(
            pc,
            in_channels,
            num_kernels,
            [1, 1],
            default_strides=[stride, stride],
            activation=identity,
            nobias=True,
        )
    else:
        shortcut = None
    # Full residual block
    res_block = Sequential(
        # Actual residual layer
        ResidualLayer(block, shortcut_transform=shortcut),
        # Activation
        LambdaLayer(relu),
    )
    # Add the residual block to the list of res_layers
    res_layers.append(res_block)
    # Divide the number of pixels according to the stride
    n_pixels //= stride**2

residual_stack = Sequential(*res_layers)

network = Sequential(
    residual_stack,
    # Flatten the resulting 3d tensor
    FlattenLayer(),
    # Final Multilayer perceptron
    Affine(pc, n_pixels * num_kernels, 128, activation=relu),
    Affine(pc, 128, 10, activation=identity, dropout=0.1)
)

# Optimizer
trainer = dy.MomentumSGDTrainer(pc, learning_rate=0.01, mom=0.9)

# Data
# ====

# Download MNIST
cifar10.download_cifar10("data")

# Load the data
(train_x, train_y), (test_x, test_y) = cifar10.load_cifar10("data")

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
