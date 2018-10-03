#!/usr/bin/env python3

from math import ceil
import time

import numpy as np
import dynet as dy

from dynn.layers.dense_layers import DenseLayer
from dynn.layers.embedding_layers import EmbeddingLayer
from dynn.layers.convolution_layers import Conv1DLayer
from dynn.layers.pooling_layers import MaxPooling1DLayer
from dynn.layers.combination_layers import StackedLayers, ConcatenatedLayers
from dynn.activations import relu, identity
from dynn.parameter_initialization import NormalInit

from dynn.data import sst
from dynn.data.dictionary import Dictionary
from dynn.data.batching import PaddedSequenceBatchIterator

# For reproducibility
dy.reset_random_seed(31415)
np.random.seed(51413)

# Data
# ====

# Download SST
sst.download_sst(".")

# Load the data
(
    (train_x, train_y),
    (dev_x, dev_y),
    (test_x, test_y),
) = sst.load_sst(".", terminals_only=True, binary=True)

# Learn the dictionary
dic = Dictionary.from_data(train_x)
dic.freeze()

# Numberize the data
train_x = dic.numberize(train_x)
dev_x = dic.numberize(dev_x)
test_x = dic.numberize(test_x)

# Create the batch iterators
train_batches = PaddedSequenceBatchIterator(
    train_x, train_y, dic, max_samples=64
)
dev_batches = PaddedSequenceBatchIterator(
    dev_x, dev_y, dic, max_samples=1, shuffle=False
)
test_batches = PaddedSequenceBatchIterator(
    test_x, test_y, dic, max_samples=1, shuffle=False
)

# Model
# =====

# Hyper-parameters
EMBED_DIM = 100
HIDDEN_DIM = 512
MAX_WIDTH = 4
N_CLASSES = 2

# Master parameter collection
pc = dy.ParameterCollection()


# Embeddings Layer
embeddings = EmbeddingLayer(pc, dic, EMBED_DIM, pad_mask=0.0)
# Convolutions
conv1d = [
    Conv1DLayer(pc, EMBED_DIM, HIDDEN_DIM//MAX_WIDTH, width, activation=relu)
    for width in range(1, MAX_WIDTH+1)
]
# Network
network = StackedLayers(
    embeddings,
    # Convolution layer
    ConcatenatedLayers(
        *conv1d,
        dim=-1,
    ),
    # Max pooling
    MaxPooling1DLayer(),
    # Softmax layer
    DenseLayer(pc, HIDDEN_DIM, N_CLASSES, activation=identity, dropout=0.5),
)

# Optimizer
trainer = dy.RMSPropTrainer(pc, learning_rate=0.001)


# Training
# ========

# Start training
for epoch in range(5):
    # Time the epoch
    start_time = time.time()
    for batch, y in train_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=False, update=True)
        # Run the cnn
        logits = network(batch.sequences)
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
    # Validate
    accuracy = 0
    for batch, y in dev_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=True, update=False)
        # Run the cnn
        logits = network(batch.sequences)
        # Get prediction
        predicted = logits.npvalue().argmax(axis=0)
        # Accuracy
        accuracy += (predicted == y).sum()
    # Average accuracy
    accuracy /= dev_batches.num_samples
    # Print final result
    print(f"Dev accuracy: {accuracy*100:.2f}%")

# Testing
# =======

# Test
accuracy = 0
for batch, y in test_batches:
    # Renew the computation graph
    dy.renew_cg()
    # Initialize layers
    network.init(test=True, update=False)
    # Run the cnn
    logits = network(batch.sequences)
    # Get prediction
    predicted = logits.npvalue().argmax(axis=0)
    # Accuracy
    accuracy += (predicted == y).sum()
# Average accuracy
accuracy /= test_batches.num_samples
# Print final result
print(f"Test accuracy: {accuracy*100:.2f}%")

